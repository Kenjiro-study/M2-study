# simple_llm_seller.py
import os, dspy, re, math, random
from typing import Optional, Literal

from ..strategies import STRATEGIES, CATEGORY_CONTEXT
from .extractor import PriceExtractor
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

StatusType = Literal['ACCEPTANCE', 'REJECTION', 'CONTINUE']

# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """You are a professional sales assistant tasked with selling a product. Your goal is to negotiate the best possible price for the product and close the transaction as close to your 'target_price' as possible.

    [RESPONSE CONSTRAINTS]
    - You must not sell below the 'minimum_price'.
    - **The response MUST be natural and human-like.**
    - **The response MUST be short and concise, focusing on one main point**

    [GOAL]
    - Negotiate to sell the product at the highest possible price.
    - Use effective negotiation strategies to maximize your profit.
    - [IMPORTANT] You must not go below the "minimum_price." If you do, you must reject the offer and tell them you cannot go any lower.

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
    minimum_price: Optional[float] = dspy.InputField(desc="Your lowest trading price. Do not go below this minimum under any circumstances.")
    target_price : Optional[float] = dspy.InputField(desc="Your target trading price")

    response: str = dspy.OutputField(desc="natural language response.")

class NegotiationJudge(dspy.Signature):
    """You are an impartial judge evaluating the progress of a negotiation between a Buyer and a Seller.
    Analyze the Seller's latest message in response to the Buyer's latest message to determine the current status.

    [STATUS DEFINITIONS]
    1. ACCEPTANCE -- The seller explicitly agrees to the proposed price/terms. The deal is closed.
    2. REJECTION -- The seller explicitly withdraws from the negotiation or states they cannot proceed
    3. CONTINUE -- The negotiation is ongoing. Includes: expressing interest, asking questions, counter-offers, or ambiguous responses.

    In your analysis, consider:
    - Has the seller explicitly accepted the offered price?
    - Has the seller explicitly rejected the offer or indicated they are walking away?
    - Has the seller said they cannot afford the price?- Is the seller asking further questions or making a counter-offer?
    
    **Please output only a single word: ACCEPTANCE, REJECTION, or CONTINUE**
    """

    buyer_latest_message: str = dspy.InputField(desc="Buyer's latest message.(If none, assume ’No response yet’)")
    seller_latest_message: str = dspy.InputField(desc="Seller's latest message.")

    reasoning: str = dspy.OutputField(desc="Step-by-step analysis. 1. Does the buyer mention a price? 2. Is it mere interest or a final decision? 3. Conclusion.")
    status: StatusType = dspy.OutputField(desc="Negotiation Status. Please output only a single word: ACCEPTANCE, REJECTION, or CONTINUE")

class SimpleLLMSellerAgent():
    """
    AgreeMate baseline negotiation system の seller agent
    seller-specific の交渉行動と戦略の解釈を実装する
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
        self.is_buyer = False
        self.role = "seller"
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

        self.min_price = self.min_price_select()

        # predictor modules のセットアップ
        self.response_predictor = dspy.Predict(NegotiationResponse)
        self.status_predictor = dspy.Predict(NegotiationJudge)

    def round_three_digit(self, price: float):
        if price == 0.0:
            return 0.0
        exponent = math.floor(math.log10(abs(price)))
        ndigit = 2 - exponent
        return round(price, ndigit)
    
    def min_price_select(self) -> float:
        """最低価格の設定"""
        min_price = self.list_price * random.uniform(0.9, 0.5)        
        min_price = self.round_three_digit(min_price)

        return min_price
    
    def compute_utility(self, final_price: float) -> float:
        # 最終価格が目標価格より高ければ1.0
        if final_price >= self.target_price:
            return 1.0
        # 最終価格が最低価格より低ければ0.0
        elif final_price <= self.min_price:
            return 0.0

        final_diff = final_price - self.min_price
        target_diff = self.target_price - self.min_price
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
            "minimum_price": self.min_price,
            "target_price":  self.target_price
        }

        response_prediction = self.response_predictor(**context)
        response_prediction['response'] = self.clean_generator_output(response_prediction['response'])

        return response_prediction
    
    def status_judge(self, seller_latest_message) -> dict:
        if self.partner_data is None:
            context = {
                "buyer_latest_message": None,
                "seller_latest_message": seller_latest_message
            }
        else:
            context = {
                "buyer_latest_message": self.partner_data['content'],
                "seller_latest_message": seller_latest_message
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
            response_prediction = self.response_generation()
            response = response_prediction['response']

            print(f"generator result: {response}") ########

            status_prediction = self.status_judge(response)

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
        if message["intent"] == "accept":
            message["price"] = self.all_price_history[-1]

        # 自分自身の状態を更新する
        message = self.update_state(message)
        return message

def test_sinple_llm_seller():
    """sinple_llm_seller の機能をテストする"""
    import os

    # test LM のセットアップ
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")

    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    # seller agent の作成
    seller = SimpleLLMSellerAgent(
        strategy_name="length",
        target_price=100.0,
        category="electronics",
        min_price=80.0,
        initial_price=120.0,
        lm=test_lm
    )

    # 初期化のテスト
    assert seller.role == "seller"
    assert seller.min_price == 80.0
    assert seller.initial_price == 120.0

    # 最初のオファーのテスト
    response = seller.step()
    assert response["role"] == "seller"
    #assert "120" in response["content"] # should include initial price

    # オファー処理のテスト
    message = {
        "role": "buyer",
        "content": "I can offer $90"
    }
    seller.update_state(message)

    # counter-offer 生成のテスト
    response = seller.step()
    assert response["role"] == "seller"
    assert "content" in response

    print("✓ All seller agent tests passed")
    return seller

if __name__ == "__main__":
    seller = test_sinple_llm_seller()