# buyer.py
import dspy, random, math
from typing import Optional

from .base_agent import BaseAgent, PriceNegotiationManager, InfoNegotiationManager, NegotiationPhase
from .extractor import PriceExtractor
from ..strategies import BUYER_INTENT_CONTEXT, BUYER_LANGUAGE_SKILLS

# 交渉中に自然言語の応答を生成する
class NegotiationResponse(dspy.Signature):
    """You are a BUYER. First, reason step-by-step about how to construct your response. Then, generate the final response based on your reasoning.

    [REASONING GUIDELINES]
    1. What is the overall STRATEGY?
    2. Is there an `offer_price`? If yes, how will I include this specific price in my response sentence?
    3. Based on the strategy and the price, what is the most effective and concise message?

    [PRICE HANDLING RULES (EXTREMELY IMPORTANT)]
    1.  **Check if `offer_price` is provided (is not None).**
    2.  **IF `offer_price` IS PROVIDED (e.g., $1895.0):**
        - Your response *MUST* include this exact price.
        - This applies to strategies like 'init-price', 'counter-price', 'insist', 'accept'.
    3.  **IF `offer_price` IS NONE (e.g., for 'inform', 'inquire', 'vague-price'):**
        - Your response *MUST NOT*, under any circumstances, include *any* specific monetary value or price number (e.g., "$1895.0", "1895").
        - **DO NOT mention the 'List Price'** from `item_information`, even if the partner asks about the price.
        - If the partner asks for the price, respond vaguely (e.g., "We can discuss the price," "What do you have in mind?") or state that you will propose a price soon.

    [RESPONSE CONSTRAINTS]
    - **The response MUST be natural and human-like.**
    - **The response MUST be short and concise, focusing on one main point**
    """
    item_information: str = dspy.InputField(desc="Product name, category, list price, and detailed description for negotiation")
    conversation_history: str = dspy.InputField(desc="Previous chat history")
    partner_utterance: dict = dspy.InputField(desc="The partner's statement to which we should respond. This includes information on price, role, intended meaning of the statement, and the content of the statement.")
    strategy: str = dspy.InputField(desc="Response strategy. Please generate a response based on this information.")
    language_skill: str = dspy.InputField(desc="Language skills complement strategy")

    offer_price: str = dspy.InputField(desc="The *exact* price string (e.g., '$1895.0') to include in the response. If 'NONE', do not mention any price.")

    response: str = dspy.OutputField(desc="natural language response following strategy guidance")

class BuyerAgent(BaseAgent):
    """
    AgreeMate baseline negotiation system の Buyer agent
    buyer-specific の交渉行動と戦略解釈を実装する
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float, 
        category: str,
        item_info: dict[str, any],
        lm: dspy.LM = None
    ):
        """
        buyer agent を初期化する

        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: Buyer の目標購入金額
            category: 商品のカテゴリー
            max_price: 最大許容価格 (デフォルト値は target より 10%高い価格)
            lm: 応答生成用の DSPy 言語モデル
        """
        super().__init__(
            strategy_name=strategy_name,
            target_price=target_price,
            list_price=list_price,
            category=category,
            is_buyer=True,
            item_info = item_info, # 2025/9/18 追加
            lm=lm
        )

        self.strategy_name = strategy_name 
        self.max_price = self.max_price_select()
        self.all_keys = list(BUYER_LANGUAGE_SKILLS.keys())
        self.keys_to_pick = []
        self.price_gap = self.max_price - self.target_price
        self.accept_line = self.accept_line_select()

        # predictor modules のセットアップ
        self.response_predictor = dspy.Predict(NegotiationResponse)
        self.price_intent_predictor = dspy.Predict(PriceNegotiationManager)
        self.info_intent_predictor = dspy.Predict(InfoNegotiationManager)

    def accept_line_select(self) -> float:
        if self.strategy_name == "fair":
            accept_line = self.target_price + ((self.list_price - self.target_price) * random.uniform(0.4, 0.1))
        elif self.strategy_name == "utility":
            accept_line = self.target_price + ((self.list_price - self.target_price) * random.uniform(0.2, 0.1))
        elif self.strategy_name == "length":
            accept_line = self.target_price + ((self.list_price - self.target_price) * random.uniform(0.2, 0.1))
        else:
            raise ValueError("Invalid strategy name")
        #print("buyer_accept_line: ", accept_line)
        return accept_line

    def max_price_select(self) -> float:
        """性格ごとの最高価格の設定"""
        if self.strategy_name == "fair":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.7))
        elif self.strategy_name == "utility":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.3))
        elif self.strategy_name == "length":
            max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.5))
        else:
            raise ValueError("Invalid strategy name")
        
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

    def _analyze_state(self) -> dict:
        """buyer の視点から現在の交渉状況を分析する"""
        if self.current_price is None:
            return {
                'price_sentiment': 'unknown',
                'bargaining_power': 'strong',
                'recommended_flexibility': 0.1 # start conservative
            }

        analysis = self.state_analyzer(
            current_price=self.current_price,
            target_price=self.target_price,
            strategy_name=self.strategy["name"],
            category=self.category,
            num_turns=self.num_turns
        )

        return {
            'price_sentiment': analysis.price_sentiment,
            'bargaining_power': analysis.bargaining_power,
            'recommended_flexibility': analysis.recommended_flexibility
        }
    
    def get_price_manager_context(self) -> dict:
        """価格交渉用の context を取得する"""
        context = super().get_manager_context()
        context.update({
            "agent_strategy": self.strategy['price_buyer_style'],
        })
        return context
    
    def get_info_manager_context(self) -> dict:
        """情報交換用の context を取得する"""
        context = super().get_manager_context()
        context.update({
            "agent_strategy": self.strategy['info_buyer_style'],
        })
        return context
    
    def fair_manager(self) -> dict:
        if self.partner_data != None and self.partner_data['price'] != None:
            if self.accept_line >= self.partner_data['price']:
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 5 or (self.price_history and self.price_history[-1] == self.max_price and self.last_action == "insist"):
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.price_intent_predictor(**self.get_price_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip(" \n`")

        # insistと予測されたが, まだ自分が価格提案を行っていない場合, またはinsistが連続した場合はcounter-priceに変更
        if ((intent == "insist") and (not self.price_history)) or ((intent == "insist") and (self.last_action == "insist")):
            intent = "counter-price"
        # parser等のミスでthanksが連発したら価格交渉に戻す(基本thanksは交渉終了時のみ)
        elif (intent == "thanks") and (self.last_action == "thanks"):
            intent = "counter-price"

        # 価格の設定
        price =  None  
        if intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1] + (self.price_gap * random.uniform(0.2, 0.45))
            price = self.round_three_digit(price)

            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price

            if price > self.accept_line:
                price = self.round_three_digit(self.accept_line)
                self.accept_line = price * random.uniform(1.05, 1.2)
                if self.accept_line > self.max_price:
                    self.accept_line = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }
    
    def utility_manager(self) -> dict:
        if self.partner_data != None and self.partner_data['price'] != None:
            if self.accept_line >= self.partner_data['price']:
                return{
                    "intent": "agree",
                    "price": None
                }
            elif self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 5 or (self.price_history and self.price_history[-1] == self.max_price and self.last_action == "insist"):
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.price_intent_predictor(**self.get_price_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip(" \n`")

        # insistと予測されたが, まだ自分が価格提案を行っていない場合, またはinsistが連続した場合はcounter-priceに変更
        if ((intent == "insist") and (not self.price_history)) or ((intent == "insist") and (self.last_action == "insist")):
            intent = "counter-price"
        # parser等のミスでthanksが連発したら価格交渉に戻す(基本thanksは交渉終了時のみ)
        elif (intent == "thanks") and (self.last_action == "thanks"):
            intent = "counter-price"

        # 価格の設定
        price =  None  
        if intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1] + (self.price_gap * random.uniform(0.05, 0.35))

            price = self.round_three_digit(price)

            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price
            
            if price > self.accept_line:
                price = self.round_three_digit(self.accept_line)
                self.accept_line = price * random.uniform(1.05, 1.2)
                if self.accept_line > self.max_price:
                    self.accept_line = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }
    
    def length_manager(self) -> dict:
        if self.partner_data != None and self.partner_data['price'] != None:
            if self.accept_line >= self.partner_data['price']:
                return{
                    "intent": "agree",
                    "price": None
                }
            elif (self.pertner_intent_history.count("counter-price") + self.pertner_intent_history.count("insist") == 5) or (self.price_history and self.price_history[-1] == self.max_price and self.last_action == "insist"):
                return{
                    "intent": "disagree",
                    "price": None
                }
            elif len(self.price_history) >= 1 and self.price_history[-1] == self.max_price:
                return{
                    "intent": "insist",
                    "price": self.max_price
                }
        
        prediction = self.price_intent_predictor(**self.get_price_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip(" \n`")

        # insistと予測されたが, まだ自分が価格提案を行っていない場合, またはinsistが連続した場合はcounter-priceに変更
        if ((intent == "insist") and (not self.price_history)) or ((intent == "insist") and (self.last_action == "insist")):
            intent = "counter-price"
        # parser等のミスでthanksが連発したら価格交渉に戻す(基本thanksは交渉終了時のみ)
        elif (intent == "thanks") and (self.last_action == "thanks"):
            intent = "counter-price"

        # 価格の設定
        price =  None  
        if intent == "counter-price":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1] + (self.price_gap * random.uniform(0.1, 0.25))

            price = self.round_three_digit(price)
            
            # 最高価格を超えていたら最高価格に設定
            if price >= self.max_price:
                price = self.max_price

            if price > self.accept_line:
                price = self.round_three_digit(self.accept_line)
                self.accept_line = price * random.uniform(1.05, 1.2)
                if self.accept_line > self.max_price:
                    self.accept_line = self.max_price

        elif intent == "insist":
            if not self.price_history:
                price = self.target_price
            else:
                price = self.price_history[-1]

        return {
            "intent": intent,
            "price": price
        }
    
    def info_manager(self) -> dict:
        prediction = self.info_intent_predictor(**self.get_info_manager_context())
        intent = (prediction.next_intent).split('\n')[0].strip(" \n`")

        price =  None  
        if intent == "init-price":
            price = self.target_price

        return {
            "intent": intent,
            "price": price
        }

    def predict_action_manager(self) -> dict:
        self.update_negotiation_phase()
        print("current_phase: ", self.current_phase)
        if self.last_action == "agree":
            return{
                "intent": "accept",
                "price": None
            }
        elif self.last_action == "disagree":
            return{
                "intent": "reject",
                "price": None
            }
        elif self.partner_data and self.partner_data['intent'] == "inquire":
            return{
                "intent": "inform",
                "price": None
            }
        elif self.current_phase == NegotiationPhase.GREETING:
            return{
                "intent": "intro",
                "price": None
            }
        elif self.current_phase == NegotiationPhase.INFO_EXCHANGE:
            prediction = self.info_manager()
        elif self.current_phase == NegotiationPhase.INIT_PRICE:
            return{
                "intent": "init-price",
                "price": self.target_price
            }
        elif self.current_phase == NegotiationPhase.PRICE_NEGOTIATION:
            if self.strategy_name == "fair":
                prediction = self.fair_manager()
            elif self.strategy_name == "utility":
                prediction = self.utility_manager()
            elif self.strategy_name == "length":
                prediction = self.length_manager()
            else:
                raise ValueError("Invalid strategy name")
        else:
            raise ValueError("Unexpected condition")

        return prediction
    
    def select_language_skill(self, intent:str):
        if intent in ["init-price", "counter-price","vague-price", "insist"]:
            if not self.keys_to_pick:
                self.keys_to_pick = self.all_keys.copy()
                random.shuffle(self.keys_to_pick)
            selected_key = self.keys_to_pick.pop()
            sentence = BUYER_LANGUAGE_SKILLS[selected_key]
        else:
            sentence = ""
        
        return sentence
    
    def response_generation(self, intent: str, price: float | None = None) -> dict:
        context = super().response_generation(intent, price)
        strategy = BUYER_INTENT_CONTEXT[intent]
        context["strategy"] = strategy
        context["language_skill"] = self.select_language_skill(intent)
        response_prediction = self.response_predictor(**context)

        response_prediction['response'] = self.clean_generator_output(response_prediction['response'])

        return response_prediction


def test_buyer_agent():
    """Test buyer agent の機能をテストする"""
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
    buyer = BuyerAgent(
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
    buyer = test_buyer_agent()