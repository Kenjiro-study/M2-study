# human.py
import os, dspy, random, math
from typing import Dict

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

from ..strategies import STRATEGIES, CATEGORY_CONTEXT

class HumanAgent:
    """
    AgreeMate baseline negotiation system の Human Agent
    人間が買い手側と売り手側のどちらかのエージェントの役割を果たす場合の機能と抽象メソッドを定義します。
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float,
        category: str,
        is_buyer: bool,
        item_info: dict[str, any],
    ):
        if strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.strategy = STRATEGIES[strategy_name]
        self.category_context = CATEGORY_CONTEXT[category]
        self.target_price = target_price
        self.list_price = list_price
        self.category = category
        self.is_buyer = is_buyer
        self.role = "buyer" if is_buyer else "seller"
        self.item_info = item_info # 2025/9/18 追加
        self.min_price = self.min_price_select()
        self.max_price = self.max_price_select()

        # 状態のトラッキング
        self.conversation_history = []
        self.price_history = []
        self.partner_price_history = [] # 相手の価格の履歴
        self.all_price_history = [] # 自分と相手双方の価格の履歴
        self.pertner_intent_history = [] # 相手のインテントの履歴
        self.last_action = None
        self.partner_data = None
        self.num_turns = 0
        

        # パーサー用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = "archive/da_system/agents/parser/model/roberta_fold_1/checkpoint-82304"
        self.parser = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=12)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    
    def round_three_digit(self, price: float):
        if price == 0.0:
            return 0.0
        exponent = math.floor(math.log10(abs(price)))
        ndigit = 2 - exponent
        return round(price, ndigit)

    def min_price_select(self) -> float:
        min_price = self.list_price * random.uniform(0.9, 0.5)
        min_price = self.round_three_digit(min_price)

        return min_price
    
    def max_price_select(self) -> float:
        max_price = self.target_price + ((self.list_price - self.target_price) * random.uniform(1.0, 0.3))
        max_price = self.round_three_digit(max_price)
        
        # 最高価格が定価を超えてしまう場合には定価に修正
        if max_price >= self.list_price:
            max_price = self.list_price

        return max_price

    def compute_utility(self, final_price: float) -> float:
        if self.is_buyer == True:
            if final_price <= self.target_price:
                return 1.0
            elif final_price >= self.max_price:
                return 0.0
            else:
                final_diff = self.max_price - final_price
                target_diff = self.max_price - self.target_price
                utility = final_diff / target_diff
                return utility
        else:
            if final_price >= self.target_price:
                return 1.0
            elif final_price <= self.min_price:
                return 0.0
            else:
                final_diff = final_price - self.min_price
                target_diff = self.target_price - self.min_price
                utility = final_diff / target_diff
                return utility
            
    def parse_dialogue(self, text):
        parser = self.parser.to(self.device)
        with torch.no_grad():
            parser.eval()
            pre_text = (self.conversation_history[-1])['content'] if self.conversation_history else "[PAD]"
            inputs = self.tokenizer(pre_text, text, max_length=512, truncation=True, return_tensors="pt")
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            outputs = parser(**inputs)

            logits = outputs.logits # ロジットの取得
            probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
            predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
            predicted_class = parser.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換
        
        self.num_turns += 1
        return predicted_class

    def update_state(self, message: Dict[str, str], extractor) -> Dict:
        """
        LLM extraction を使用して交渉状態を更新する
        StateExtractor を使用して, メッセージから構造化された情報を取得する

        Args:
            message: Dict containing 'role' and 'content' of message
        """
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Invalid message format")

        price = None

        # action 状態を更新する
        # 人間が交渉を受け入れたり断ったりする時はacceptかrejectと入力する
        if message['content'] == "accept":
            self.last_action = "accept"
            price = self.all_price_history[-1]
        elif message['content'] == "reject":
            self.last_action = "reject"
        else:
            self.last_action = self.parse_dialogue(message['content'])
        
        # 自分のインテントが価格交渉に関するものの場合, 価格を抽出
        if self.last_action in ["init-price", "counter-price", "insist"]:
            with dspy.context(lm=extractor.lm):
                price_prediction = extractor.compiled_extractor(
                    message_content=message['content']
                )
            price = price_prediction["extracted_price"]
            if price == None:
                self.last_action = "unknown"
            elif (not self.price_history) and (not self.partner_price_history) and (self.last_action in ["counter-price", "insist"]):
                self.last_action = "init-price"
            elif (self.price_history) and (self.price_history[-1] == price) and (self.last_action in ["init-price", "counter-price"]):
                self.last_action = "insist"
            elif (self.partner_price_history or self.price_history) and (self.last_action == "init-price"):
                self.last_action = "counter-price"

        # 新しい価格が検出されたら, 価格の状態を更新する
        if price is not None:
            self.price_history.append(price)
            self.all_price_history.append(price)
    
        message.update({
            "price": price,
            "intent": self.last_action,
        })

        # 会話状態を更新する
        self.conversation_history.append(message)
        #self.num_turns += 1

        return message

    def step(self, partner_data, extractor) -> Dict[str, str]:
        """
        交渉ステップを実行する: つまり行動を予測し, 応答を生成する

        Returns:
            応答メッセージのコンテンツと役割を含む辞書
        """
        print(f"{self.role}'s turn") ########
        # パートナーのデータを更新
        self.partner_data = partner_data

        # パーサー
        # まずはここで相手の発言のインテントを予測し, 必要であれば価格を抽出する
        if self.partner_data is not None:
            self.partner_data['intent'] = self.parse_dialogue(self.partner_data['content'])
            self.partner_data['price'] = None

            # 相手のインテントが価格交渉に関するものの場合, 価格を抽出
            if self.partner_data['intent'] in ["init-price", "counter-price", "insist"]:
                with dspy.context(lm=extractor.lm):
                    price_prediction = extractor.compiled_extractor(
                        message_content=self.partner_data['content']
                    )
                self.partner_data['price'] = price_prediction["extracted_price"]
                if self.partner_data['price'] == None:
                    self.partner_data['intent'] = "unknown"
                elif (not self.price_history) and (not self.partner_price_history) and (self.partner_data['intent'] in ["counter-price", "insist"]):
                    self.partner_data['intent'] = "init-price"
                elif (self.partner_price_history) and (self.partner_price_history[-1] == self.partner_data['price']) and (self.partner_data['intent'] in ["init-price", "counter-price"]):
                    self.partner_data['intent'] = "insist"
                elif (self.partner_price_history or self.price_history) and (self.partner_data['intent'] == "init-price"):
                    self.partner_data['intent'] = "counter-price"

            # パートナー情報の更新
            self.conversation_history.append(self.partner_data)
            self.pertner_intent_history.append(self.partner_data['intent'])
            #self.num_turns += 1 # ターンを一つ進める
            if self.partner_data['price'] != None:
                self.partner_price_history.append(self.partner_data['price'])
                self.all_price_history.append(self.partner_data['price'])
            print(f"parser result: {self.partner_data['intent']}(price={self.partner_data['price']})") ########


        # 自然言語の応答を入力する
        user_response = input(f"Your turn! Please your message as a {self.role}: ")

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": user_response
        }

        # 自分自身の状態を更新する
        message = self.update_state(message, extractor)

        return message

