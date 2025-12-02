# base_agent.py
import os, dspy, re, math

from ..strategies import STRATEGIES, CATEGORY_CONTEXT
from .extractor import PriceExtractor
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

PRICE_RELATED_INTENTS = ["init-price", "counter-price", "insist"]

class PriceNegotiationManager(dspy.Signature):
    """As a price negotiation agent, considering the dialogue history, the partner's last utterance, roles, and your own strategy, select the single most strategic "intent" to take next. Currently in the price negotiation phase.

    [THOUGHT PROCESS]
    1.  First, analyze the current dialogue history and the partner's most recent `partner_intent`.
    2.  Next, consider your own `agent_role` (e.g., Buyer or Seller) and `agent_strategy`.
    3.  Finally, strictly follow the "Intent Selection Criteria" below to select the single most appropriate intent label.
    
    [INTENT SELECTION CRITERIA (top priority)]
    - counter-price: Select to propose a *different* price after the partner has proposed an `init-price` or `counter-price`.
    - vague-price: Select to negotiate indirectly without stating a specific price (e.g., "That's a bit high...").
        (Condition: Select this *only if* concrete price negotiation is deadlocked.)
    - insist: Select to re-state your *previous price* after the partner has made a `counter-price`. 
    - supplemental: Select to provide additional information (e.g., product benefits) when the partner's intent was *not* `inquire`.
    - thanks: Select to express your gratitude for reaching an agreement.
        (Condition: Only if your partner's "partner_intent" is "agree" or "thanks")"""
    
    # --- 入力フィールド ---
    dialogue_history = dspy.InputField(desc="The past dialogue history with intent labels for each utterance.")
    partner_utterance = dspy.InputField(desc="The partner's most recent utterance to respond to.")
    partner_intent = dspy.InputField(desc="The intent label of the partner's most recent utterance.")
    partner_role = dspy.InputField(desc="The role of the partner (e.g., Buyer, Seller).")
    agent_role = dspy.InputField(desc="Your role (e.g., Buyer, Seller).")
    agent_strategy = dspy.InputField(desc="Your strategy for selecting an intent. This is a guideline; the 'INTENT SELECTION CRITERIA' above take precedence.")

    # --- 出力フィールド ---
    next_intent = dspy.OutputField(
        desc="The intent label for the agent's next action. Choose exactly one from the following 4 types: "
             "counter-price, vague-price, insist, supplemental, thanks"
    )

class InfoNegotiationManager(dspy.Signature):
    """As a price negotiation agent, considering the dialogue history, the partner's last utterance, roles, and your own strategy, select the single most strategic "intent" to take next. Currently in the information exchange phase.

    [THOUGHT PROCESS]
    1.  First, analyze the current dialogue history and the partner's most recent `partner_intent`.
    2.  Next, consider your own `agent_role` (e.g., Buyer or Seller) and `agent_strategy`.
    3.  Finally, strictly follow the "Intent Selection Criteria" below to select the single most appropriate intent label.
    
    [INTENT SELECTION CRITERIA (top priorityy)]
    - inquire: Select this when you need to ask about details such as the condition of the product.
    - init-price: Select to make the *first* price proposal.
    - supplemental: Select to provide additional information (e.g., product benefits) when the partner's intent was *not* `inquire`."""
    
    # --- 入力フィールド ---
    dialogue_history = dspy.InputField(desc="The past dialogue history with intent labels for each utterance.")
    partner_utterance = dspy.InputField(desc="The partner's most recent utterance to respond to.")
    partner_intent = dspy.InputField(desc="The intent label of the partner's most recent utterance.")
    partner_role = dspy.InputField(desc="The role of the partner (e.g., Buyer, Seller).")
    agent_role = dspy.InputField(desc="Your role (e.g., Buyer, Seller).")
    agent_strategy = dspy.InputField(desc="Your strategy for selecting an intent. This is a guideline; the 'INTENT SELECTION CRITERIA' above take precedence.")

    # --- 出力フィールド ---
    next_intent = dspy.OutputField(
        desc="The intent label for the agent's next action. Choose exactly one from the following 3 types: "
             "inquire, init-price, supplemental"
    )

#class AddPriceInfo(dspy.Signature):
    """Change `response` to a negotiation statement asking for the price shown in `price`.
    """
    # --- 入力フィールド ---
    #response = dspy.InputField(desc="Sentences to be corrected")
    #price = dspy.InputField(desc="The price you want to include in the response")

    # --- 出力フィールド ---
    #revised_response = dspy.OutputField(desc="A revised response with the correct price information")

class NegotiationPhase:
    GREETING = "GREETING"                   # 挨拶フェーズ
    INFO_EXCHANGE = "INFO_EXCHANGE"         # 情報交換フェーズ
    INIT_PRICE = "INIT_PRICE"               # 価格提案フェーズ
    PRICE_NEGOTIATION = "PRICE_NEGOTIATION" # 価格交渉フェーズ

class BaseAgent:
    """
    AgreeMate baseline negotiation system の Base Agent
    買い手側と売り手側の両方の子エージェントが実装するコア機能と抽象メソッドを定義します。
    """

    def __init__(
        self,
        strategy_name: str,
        target_price: float,
        list_price: float,
        category: str,
        is_buyer: bool,
        item_info: dict[str, any],
        lm: dspy.LM,
    ):
        """
        negotiation agent を初期化する

        Args:
            strategy_name: STRATEGIES の戦略名
            target_price: このエージェントの目標価格
            category: 商品のカテゴリー (electronics, vehicles, etc)
            is_buyer: buyer (True) であるか seller (False) であるか
            lm: 応答生成のための DSPy 言語モデル 
        """
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
        self.lm = lm # 2025/7/15 追加

        # 状態のトラッキング
        self.conversation_history = []
        self.price_history = [] # 自分の価格の履歴
        self.partner_price_history = [] # 相手の価格の履歴
        self.all_price_history = [] # 自分と相手双方の価格の履歴
        self.pertner_intent_history = [] # 相手のインテントの履歴
        self.last_action = None
        self.partner_data = None # 2025/9/17 追加
        self.num_turns = 0
        self.current_phase = "" # 2025/10/30 追加
        

        # パーサー用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = "archive/da_system/agents/parser/model/roberta_fold_1/checkpoint-82304"
        self.parser = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=12)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        # predictor modules のセットアップ
        self.price_intent_predictor = dspy.Predict(PriceNegotiationManager)
        self.info_intent_predictor = dspy.Predict(InfoNegotiationManager)
        #self.response_modifier = dspy.Predict(AddPriceInfo)

        # すべてのモジュールで提供された言語モデルを使用するように DSPy を構成する
        #dspy.settings.configure(lm=lm)
    
    def round_three_digit(self, price: float):
        if price == 0.0:
            return 0.0
        exponent = math.floor(math.log10(abs(price)))
        ndigit = 2 - exponent
        return round(price, ndigit)

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
    
    def parse_partner_dialogue(self):
        """パートナーの発言を分析する"""
        parser = self.parser.to(self.device)
        with torch.no_grad():
            parser.eval()
            pre_text = (self.conversation_history[-1])['content'] if self.conversation_history else "[PAD]"
            inputs = self.tokenizer(pre_text, self.partner_data['content'], max_length=512, truncation=True, return_tensors="pt")
            inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
            outputs = parser(**inputs)

            logits = outputs.logits # ロジットの取得
            probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
            predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
            predicted_class = parser.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換

        self.num_turns += 1 # ターンを一つ進める

        return predicted_class

    def update_negotiation_phase(self):
        # 一度価格交渉フェーズに入ったら戻らない
        if self.current_phase == NegotiationPhase.PRICE_NEGOTIATION:
            return
        
        # 0 or 1ターン目で相手のintentがintroの場合、GREETINGフェーズに移動
        if self.num_turns == 0 or (self.num_turns == 1 and self.partner_data and self.partner_data['intent'] != "init-price"):
            self.current_phase = NegotiationPhase.GREETING
            return
        
        # 挨拶(intro)が終わったら INFO_EXCHANGEフェーズに移動
        if self.partner_data['intent'] != "init-price" and (self.last_action == "intro" or self.num_turns == 2 or self.num_turns == 3) :
            self.current_phase = NegotiationPhase.INFO_EXCHANGE
            return
        
        # 価格提案がまだない場合のみINIT_PRICEに遷移できる
        if (not self.partner_price_history) and (not self.price_history):
            # fairの場合, 遅くとも会話4往復目からINIT-PRICEフェーズに移動
            if self.strategy["name"] == "fair" and (self.num_turns == 6 or self.num_turns == 7):
                self.current_phase = NegotiationPhase.INIT_PRICE
                return
            # utilityの場合, 遅くとも会話3往復目からINIT-PRICEフェーズに移動
            elif self.strategy["name"] == "utility" and (self.num_turns == 4 or self.num_turns == 5):
                self.current_phase = NegotiationPhase.INIT_PRICE
                return
            # lengthの場合, 遅くとも会話5往復目からINIT-PRICEフェーズに移動
            elif self.strategy["name"] == "length" and (self.num_turns == 6 or self.num_turns == 9):
                self.current_phase = NegotiationPhase.INIT_PRICE
                return

        # init-priceを検知した場合, PRICE_NEGOTIATIONフェーズに移動
        if self.partner_data['intent'] == "init-price" or self.last_action == "init-price":
            self.current_phase = NegotiationPhase.PRICE_NEGOTIATION
            return

        return
    
    def get_manager_context(self) -> dict:
        """予測の context を取得する"""
        if self.partner_data == None:
            return {
                "dialogue_history": self.conversation_history,
                "partner_utterance": "",
                "partner_intent": "",
                "partner_role": "",
                "agent_role": self.role,
            }
        else:
            return {
                "dialogue_history": self.conversation_history,
                "partner_utterance": self.partner_data['content'],
                "partner_intent": self.partner_data['intent'],
                "partner_role": self.partner_data['role'],
                "agent_role": self.role,
            }

    def predict_action_manager(self) -> dict:
        """交渉における次の intent を予測する"""
        prediction = self.price_intent_predictor(**self.get_manager_context())
        #self.lm.inspect_history(n=1) ###############################
        return {
            "rationale": prediction.rationale,
            "next_intent": prediction.next_intent,
        }
    
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

    def response_generation(self, intent: str, price: float | None = None) -> dict:
        """自然言語の応答を生成する"""
        from ..utils.model_loader import MODEL_CONFIGS

        # プロンプト用に会話履歴をフォーマットする
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[:-1]
        ])
        
        # get prompt template を取得して入力する
        model_name = self.lm.model.split('/')[-1] # 2025/7/15 model_name → model に変更
        template = MODEL_CONFIGS[model_name].prompt_template
        item_prompt = template.format(
            item_name = self.item_info["item_name"],
            category = self.item_info["category"],
            list_price = self.item_info["list_price"],
            description = self.item_info["description"]
        )
        # 価格関連のインテントでない場合は定価の情報をプロンプトから消す(これで余計に価格情報を提示するのを防ぐ)
        if intent not in PRICE_RELATED_INTENTS:
            item_prompt = re.sub(r"List Price: \d+\.\d+\n?", "", item_prompt, flags=re.IGNORECASE)
        
        offer_price = f"${price}" ######
        
        context = {
            "item_information": item_prompt,
            "conversation_history": history_text,
            "partner_utterance": self.partner_data,
            "offer_price": offer_price #######
        }

        return context

    def step(self, partner_data, extractor) -> dict[str, str]:
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
            self.partner_data['intent'] = self.parse_partner_dialogue()
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
            if self.partner_data['price'] != None:
                self.partner_price_history.append(self.partner_data['price'])
                self.all_price_history.append(self.partner_data['price'])
            print(f"parser result: {self.partner_data['intent']}(price={self.partner_data['price']})") ########

        # マネージャー
        # 次に自分の応答のインテントを考え, 戦略を決定する
        with dspy.context(lm=self.lm):
            prediction = self.predict_action_manager()
            print(f"manager result: {prediction['intent']}(price={prediction['price']})") ########

        # ジェネレーター
        # 自然言語の応答を生成する
            response_prediction = self.response_generation(
                intent=prediction["intent"], 
                price=prediction["price"]
            )
            #dspy.settings.lm.inspect_history(n=1) ###############

        print(f"generator result: {response_prediction['response']}") ########

        # メッセージを作成する
        message = {
            "role": self.role,
            "content": response_prediction["response"],
            "price": prediction["price"],
            "intent": prediction["intent"]
        }

        # acceptの場合, 交渉成立価格を記録に残すために自分が承諾したパートナーの最終提案価格を取得
        if message["intent"] == "accept":
            message["price"] = self.all_price_history[-1]

        # 自分自身の状態を更新する
        message = self.update_state(message)
        print("self.last_action: ", self.last_action) ########
        print("self.all_price_history: ", self.all_price_history) ########
        return message


def test_base_agent():
    """BaseAgent の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    #test_lm = dspy.LM(
        #model="openai/llama3.1", # llama3.1という名前だが一応llama-3.1-8Bらしい
        #api_base="http://localhost:11434/v1",
        #api_key="",
        #cache_dir=pretrained_dir
    #)
    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    agent = BaseAgent(
        strategy_name="length",
        target_price=100.0,
        category="electronics",
        is_buyer=True,
        lm=test_lm,
    )
    assert agent.role == "buyer"
    assert agent.strategy["name"] == "length"

    # 状態更新のテスト
    message = {
        "role": "seller",
        "content": "I can offer it for $150"
    }
    agent.update_state(message)
    assert len(agent.conversation_history) == 1
    assert agent.num_turns == 1

    # step のテスト
    response = agent.step()
    assert "role" in response
    assert "content" in response
    assert response["role"] == "buyer"

    print("✓ All base agent tests passed")
    return agent

if __name__ == "__main__":
    agent = test_base_agent()