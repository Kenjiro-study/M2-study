# all_one_buyer.py
import os, dspy, re, math, random
from typing import Optional, Literal
from pydantic import BaseModel, Field  #########

from ..strategies import STRATEGIES, CATEGORY_CONTEXT
from .extractor import PriceExtractor
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

IntentType = Literal['intro', 'inquire', 'inform', 'init-price', 'counter-price', 'vague-price', 'insist', 'supplemental', 'thanks', 'agree', 'disagree']
StatusType = Literal['ACCEPTANCE', 'REJECTION', 'CONTINUE']

BUYER_INTENT_DEFINITION = """
    - intro: Select at the start of negotiations. Say 'Hi' or express interest very briefly. Keep it casual, like a text message.
    - inquire: Select this when you need to ask about details such as the condition of the product. Ask a quick, short question about condition/shipping. Use simple words. No formal grammar.
    - inform: Select when answering a question from the other person. Answer the seller's question with just the necessary info. Be blunt and efficient.
    - supplemental: Select to provide additional information (e.g., product benefits) when the partner's intent was *not* `inquire`. Briefly mention your budget or reason (e.g., 'student here'). Use this to gain sympathy, not as a formal offer.
    - init-price: Select to make the *first* price proposal. Throw out your first price offer casually. Just the number and a short phrase (e.g., 'How about $X?').
    - counter-price: Select to propose a *different* price after the partner has proposed an `init-price` or `counter-price`. Counter with a new price. Be direct and short. Do not write a long explanation.
    - insist: Select to re-state your *previous price* after the partner has made a `counter-price`. Repeat your price stubbornly. Say you can't go higher. Keep it short.
    - vague-price: Select to negotiate indirectly without stating a specific price (e.g., "That's a bit high..."). Ask for a discount without naming a price yet. Use phrases like 'Can you lower it?' or 'Too expensive'.
    - disagree: Select this to decline the deal and end negotiations. Reject the current price briefly. Say 'That's too high' or 'No thanks'. Don't be polite.
    - agree: Select this when you agree to trade at the price proposed by the other party. Say 'OK' or 'I'll take it' to the current price. Keep it very short.
    - thanks: Select to express your gratitude for reaching an agreement (Condition: Only if your partner's "partner_intent" is "agree" or "thanks”). Say 'Thanks' or 'Cool'. No formal appreciation needed.
"""
BUYER_LANGUAGE_SKILLS = """
    - Emphasis: Complain that the item isn't worth the asking price. Point out flaws or age to drive the price down aggressively.
    - Emotional Strategy: Act friendly or play the victim (e.g., 'I'm broke', 'It's for my kid'). Use emotional words or emojis to bond.
    - Compare the Market: Mention that others are selling it cheaper. Say 'I saw this for $X elsewhere' to pressure the seller.
    - Transaction Guarantee: Promise immediate payment. Say 'I pay right now' or 'Instant decision' to tempt the seller.
    - Create Urgency: Say you might buy something else if they don't decide now. 'Deciding between this and another one'.
    - Chat: Just reply normally like a text message. No special tactics, just short and lazy response.
"""

class NegotiationTurn(BaseModel):
    """
    LLMが1ターンで生成すべき全ての情報をまとめたPydanticモデル。
    """
    partner_intent: IntentType = Field(..., description="Intent classification of the other person's input text")
    partner_price: Optional[str] = Field(..., description="The price offered by the other party (or None)")
    next_intent: IntentType = Field(..., description="Your next strategic intent")
    offer_price: Optional[str] = Field(..., description="The next price you offer (only for init-price, counter-price, insist; otherwise None)")
    response: str = Field(..., description="Natural language response following strategy guidance")


# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """You are a professional negotiation assistant tasked with purchasing a product. Your goal is to negotiate the best possible price for the product and close the transaction as close to your 'target_price' as possible.
    Analyze the partner's input, decide on the next strategy, and generate a response all at once in the following order:
    1. Classify the partner's input ('partner_utterance') into 11 types of intents ('partner_intent')
    2. Extract price information ('partner_price') from the partner's input ('partner_utterance').
    3. Determine your next intent ('next_intent') based on 'conversation_history', 'partner_utterance', 'partner_intent', 'partner_role', and 'agent_role'.
    4. If the intent requires a price offer('init-price', 'counter-price' and 'insist'), determine the offer price ('offer_price')
    5. Generate a natural language response based on the atrategy based on determined intent('intent_definitions'), price('offer_price') and 'item_information', 'conversation_history', and 'partner_utterrance'. If the intent is one of the following regarding price offers: init-price, counter-price, insist, vague-price, randomly select one from 'language_skills' (Emphasis, Emotional Strategy, Compare the Market, Transaction Guarantee, Create Urgency, Chat) and generate a natural language response based on that skill.
    
    [RESPONSE CONSTRAINTS]
    - You must not exceed your 'budget', otherwise you should reject the offer and say you cannot afford it.
    - **The response MUST be natural and human-like.**
    - **The response MUST be short and concise, focusing on one main point**

    [GOAL]
    - Negotiate to obtain the product at the lowest possible price.
    - Use effective negotiation strategies to achieve the best deal.
    - [IMPORTANT] You must not exceed your 'budget', otherwise you should reject the offer and say you cannot afford it. 

    [GUIDELINES]
    1. Keep your responses natural and conversational.
    2. Respond with a single message only.
    3. Keep your response concise and to the point.
    4. Don’t reveal your internal thoughts or strategy.
    5. Do not show any bracket about unknown message, like [YourName]. Remember, this is a real conversation between a buyer and a seller.
    6. Make your response as short as possible, but do not lose any important information.
    """

    item_information: str = dspy.InputField(desc="Product name, category, list price, and detailed description for negotiation")
    conversation_history: str = dspy.InputField(desc="Previous chat history")
    partner_utterance: str = dspy.InputField(desc="The partner's statement to which you should respond.")
    partner_role = dspy.InputField(desc="The role of the partner (e.g., Buyer, Seller).")
    agent_role = dspy.InputField(desc="Your role (e.g., Buyer, Seller).")
    budget: Optional[float] = dspy.InputField(desc="Your budget. Do not exceed this budget under any circumstances.")
    target_price : Optional[float] = dspy.InputField(desc="Your target trading price")
    intent_definitions : str = dspy.InputField(desc="Definitions of 11 types of intent and their strategic role")
    language_skills : str = dspy.InputField(desc="The language skills used to make statements about price offers")

    negotiation_turn: NegotiationTurn = dspy.OutputField(
        desc="A structured object containing analysis, planning, and response."
    )

# 交渉中に自然言語の応答を生成する
class NegotiationGreeting(dspy.Signature):
    """You are a professional negotiation assistant aiming to purchase a product at the best possible price. Your task is to start the conversation naturally without revealing your role as a negotiation assistant. Please write a short and friendly message to the seller, following these GUIDELINES.

    [GUIDELINES]
    1. Expresses interest in the product and asks about the possibility of negotiating the price.
    2. Sounds natural, polite, and engaging Avoid over-explaining-- just say "Hello" to start and smoothly lead into your interest.
    3. Keep the message concise and focused on opening the negotiation.
    """

    item_information: str = dspy.InputField(desc="Product name, category, list price, and detailed description for negotiation")

    response: str = dspy.OutputField(desc="natural language response.")

class NegotiationJudge(dspy.Signature):
    """(You are an impartial judge evaluating the progress of a negotiation between a Buyer and a Seller.
    Analyze the Buyer's latest message in response to the Seller's latest message to determine the current status.

    [STATUS DEFINITIONS]
    1. ACCEPTANCE -- The buyer explicitly agrees to the proposed price/terms. The deal is closed.
    2. REJECTION -- The buyer explicitly withdraws from the negotiation or states they cannot proceed
    3. CONTINUE -- The negotiation is ongoing. Includes: expressing interest, asking questions, counter-offers, or ambiguous responses.

    In your analysis, consider:
    - Has the buyer explicitly accepted the offered price?
    - Has the buyer explicitly rejected the offer or indicated they are walking away?
    - Has the buyer said they cannot afford the price?- Is the buyer asking further questions or making a counter-offer?
    
    **Please output only a single word: ACCEPTANCE, REJECTION, or CONTINUE**)
    """

    buyer_latest_message: str = dspy.InputField(desc="Buyer's latest message.")
    seller_latest_message: str = dspy.InputField(desc="Seller's latest message.(If none, assume ’No response yet’)")

    reasoning: str = dspy.OutputField(desc="Step-by-step analysis. 1. Does the buyer mention a price? 2. Is it mere interest or a final decision? 3. Conclusion.")
    status: StatusType = dspy.OutputField(desc="Negotiation Status. Please output only a single word: ACCEPTANCE, REJECTION, or CONTINUE")

class AllinOneLLMBuyerAgent():
    """
    AgreeMate baseline negotiation system の Base Agent
    買い手側と売り手側の両方の子エージェントが実装するコア機能と抽象メソッドを定義します。
    """

    def __init__(
        self,
        target_price: float,
        list_price: float,
        category: str,
        item_info: dict[str, any],
        lm: dspy.LM,
    ):
        self.category_context = CATEGORY_CONTEXT[category]
        self.target_price = target_price
        self.list_price = list_price
        self.category = category
        self.is_buyer = True
        self.role = "buyer"
        self.item_info = item_info # 2025/9/18 追加
        self.lm = lm # 2025/7/15 追加
        self.strategy_name = "free"

        # 状態のトラッキング
        self.conversation_history = []
        self.price_history = [] # 自分の価格の履歴
        self.partner_price_history = [] # 相手の価格の履歴
        self.all_price_history = [] # 自分と相手双方の価格の履歴
        self.pertner_intent_history = [] # 相手のインテントの履歴
        self.last_action = None
        self.partner_data = None # 2025/9/17 追加
        self.num_turns = 0

        self.max_price = self.max_price_select()

        # predictor modules のセットアップ
        self.response_predictor = dspy.ChainOfThought(NegotiationResponse)
        self.greeting_predictor = dspy.ChainOfThought(NegotiationGreeting)
        self.status_predictor = dspy.Predict(NegotiationJudge)
    
    def round_three_digit(self, price: float):
        if price == 0.0:
            return 0.0
        exponent = math.floor(math.log10(abs(price)))
        ndigit = 2 - exponent
        return round(price, ndigit)
    
    def max_price_select(self) -> float:
        """最高価格の設定"""
        max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.5))
        max_price = self.round_three_digit(max_price)
        
        # 最高価格が定価を超えてしまう場合には定価に修正
        if max_price >= self.list_price:
            max_price = self.list_price

        return max_price

    def compute_utility(self, final_price: float) -> float:
        if final_price <= self.target_price:
            return 1.0
        elif final_price >= self.max_price:
            return 0.0

        final_diff = self.max_price - final_price
        target_diff = self.max_price - self.target_price
        utility = final_diff / target_diff
        return utility

    def update_state(self, message: dict[str, str]) -> dict:
        """
        LLM extraction を使用して交渉状態を更新する
        StateExtractor を使用して, メッセージから構造化された情報を取得する

        Args:
            message: dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        # 会話状態を更新する
        self.conversation_history.append(message)
        self.num_turns += 1

        # 新しい価格が検出されたら, 価格の状態を更新する
        if message['price'] is not None:
            self.price_history.append(message['price'])
            self.all_price_history.append(message['price'])
        #self.lm.inspect_history(n=1) ###############################

        # action 状態を更新する
        self.last_action = message['intent']

        return message
            
    def clean_generator_output(self, text: str) -> str:
        # \" を " に置換
        cleaned = text.replace('\"', '"')
        # "### Completed:" や "###" などのストップマーカーで分割し、本体部分を取得
        stop_markers = ["### Completed:", "###", "Completed:", "[[ ##"]
        for marker in stop_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]
        # 前後の空白文字(\n 等)を削除
        cleaned = cleaned.strip()
        # 全体が"で囲まれている場合、"を削除(例: "I appreciate〜" -> I appreciate〜)
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned.strip('"')
        # 5. 再度、前後の空白を削除
        cleaned = cleaned.strip()
    
        return cleaned

    def response_generation(self) -> dict:
        """自然言語の応答を生成する"""
        from ..utils.model_loader import MODEL_CONFIGS

        # プロンプト用に会話履歴をフォーマットする
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[:-1]
        ])
        
        # get prompt template を取得して入力する
        model_name = self.lm.model
        template = MODEL_CONFIGS[model_name].prompt_template
        item_prompt = template.format(
            item_name = self.item_info["item_name"],
            category = self.item_info["category"],
            list_price = self.item_info["list_price"],
            description = self.item_info["description"]
        )
        
        context = {
            "item_information": item_prompt,
            "conversation_history": history_text,
            "partner_utterance": self.partner_data['content'],
            "partner_role": self.partner_data['role'],
            "agent_role": self.role,
            "budget": self.max_price,
            "target_price":  self.target_price,
            "intent_definitions": BUYER_INTENT_DEFINITION,
            "language_skills": BUYER_LANGUAGE_SKILLS
        }
        response_prediction = self.response_predictor(**context)
        #dspy.settings.lm.inspect_history(n=1) ###############
        turn_data: NegotiationTurn = response_prediction.negotiation_turn

        response_prediction = {
            "partner_intent": turn_data.partner_intent,
            "partner_price": turn_data.partner_price,
            "next_intent": turn_data.next_intent,
            "offer_price": turn_data.offer_price,
            "response": self.clean_generator_output(turn_data.response) # レスポンス文字列のクリーニングは継続
        }


        return response_prediction
    
    def greeting_generation(self) -> dict:
        """挨拶を生成する"""
        from ..utils.model_loader import MODEL_CONFIGS

        model_name = self.lm.model
        template = MODEL_CONFIGS[model_name].prompt_template
        item_prompt = template.format(
            item_name = self.item_info["item_name"],
            category = self.item_info["category"],
            list_price = self.item_info["list_price"],
            description = self.item_info["description"]
        )

        context = {
            "item_information": item_prompt,
        }

        response_prediction = self.greeting_predictor(**context)
        response_prediction['response'] = self.clean_generator_output(response_prediction['response'])
        #self.lm.inspect_history(n=1) ##############

        return response_prediction
    
    def status_judge(self, buyer_latest_message) -> dict:
        if self.partner_data is None:
            context = {
                "buyer_latest_message": buyer_latest_message,
                "seller_latest_message": None
            }
        else:
            context = {
                "buyer_latest_message": buyer_latest_message,
                "seller_latest_message": self.partner_data['content']
            }
        status_prediction = self.status_predictor(**context)
        status_prediction['status'] = (status_prediction['status']).split('\n')[0].strip(" \n`")

        return status_prediction


    def step(self, partner_data, extractor) -> dict[str, str]:
        """
        交渉ステップを実行する: つまり行動を予測し, 応答を生成する

        Returns:
            応答メッセージのコンテンツと役割を含む辞書
        """
        print(f"{self.role}'s turn") ########
        # パートナーのデータを更新
        self.partner_data = partner_data

        # パートナー情報の更新
        if self.partner_data is not None:
            self.conversation_history.append(self.partner_data)
            self.pertner_intent_history.append(self.partner_data['intent'])
            if self.partner_data['price'] != None:
                self.partner_price_history.append(self.partner_data['price'])
                self.all_price_history.append(self.partner_data['price'])

        # ジェネレーター
        # 自然言語の応答を生成する
        with dspy.context(lm=self.lm):
            if self.partner_data == None:
                response_prediction = self.greeting_generation()
            else:
                response_prediction = self.response_generation()

            response = response_prediction['response']

            print(f"generator result: {response}") ########

            status_prediction = self.status_judge(response)

        print("status_prediction['status']: ", status_prediction['status']) ###########
        if status_prediction['status'] == "ACCEPTANCE":
            intent = "accept"
        elif status_prediction['status'] == "REJECTION":
            intent = "reject"
        else:
            intent = "unknown"
        #print("status: ", status_prediction['status']) ########

        with dspy.context(lm=extractor.lm):
            price_prediction = extractor.compiled_extractor(
                message_content=response
            )

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": response,
            "price": price_prediction["extracted_price"],
            "intent": intent
        }

        # acceptの場合, 交渉成立価格を記録に残すために自分が承諾したパートナーの最終提案価格を取得
        print("self.all_price_history: ", self.all_price_history)
        if message["intent"] == "accept":
            message["price"] = self.all_price_history[-1]

        # 自分自身の状態を更新する
        message = self.update_state(message)
        return message
    
def test_all_in_one_llm_buyer():
    """sinple_llm_buyer の機能をテストする"""
    import os

    # test LM のセットアップ
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    # buyer agent の作成
    buyer = AllinOneLLMBuyerAgent(
        strategy_name="length",
        target_price=100.0,
        category="electronics",
        max_price=120.0,
        lm=test_lm
    )

    # 初期化のテスト
    assert buyer.role == "buyer"
    assert buyer.max_price == 120.0

    # オファー処理のテスト
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    buyer.update_state(message)

    # counter-offer 生成のテスト
    response = buyer.step()
    assert response["role"] == "buyer"
    assert "content" in response

    print("✓ All buyer agent tests passed")
    return buyer

if __name__ == "__main__":
    buyer = test_all_in_one_llm_buyer()