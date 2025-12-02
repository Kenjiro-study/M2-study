from .buyer import BuyerAgent
from .generator.search_generator import get_serach_generator
from .generator.price_tracker import PriceTracker
from .generator.tokenizer import tokenize
from .generator.get_template import extract_template

class SearchBaseBuyerAgent(BuyerAgent):
    """
    BuyerAgentを継承し、応答生成部分（ジェネレーター）だけを変更したエージェント
    """

    def __init__(self, *args, **kwargs):
        # まずは親クラス（BuyerAgent）の初期化処理をそのまま呼び出す
        super().__init__(*args, **kwargs)

        self.price_tracker = PriceTracker("archive/da_system/agents/generator/price_tracker.pkl")

    def get_generator_context(self) -> dict:
        if self.partner_data is None:
            return {
                "title": self.item_info["item_name"],
                "description": self.item_info["description"],
                "category": self.item_info["category"],
                "list_price": self.list_price,
                "my_price": self.price_history[-1] if self.price_history else self.target_price,
                "partner_price": None, # 相手の今の提案価格(提案が来るまではNone)
                "my_role": self.role,
                "partner_intent": None,
            }
        else:
            return {
                "title": self.item_info["item_name"],
                "description": self.item_info["description"],
                "category": self.item_info["category"],
                "list_price": self.list_price,
                "my_price": self.price_history[-1] if self.price_history else self.target_price,
                "partner_price": self.partner_data["price"], # 相手の今の提案価格(提案が来るまではNone)
                "my_role": self.role,
                "partner_intent": self.partner_data["intent"],
            }
    
    def response_generation(self, intent: str, price: float | None = None) -> dict:
        """
        このメソッドだけをオーバーライドする。
        stepメソッドがこの新しいメソッドを自動的に呼び出してくれる。
        """
        kb = self.get_generator_context()

        if self.partner_data is None:
            kb["partner_template"] = ""
        else:
            # 相手の発話をテンプレートに変換してkbに追加
            tokens = self.price_tracker.link_entity(tokenize(self.partner_data["content"]), kb=kb, scale=False)
            partner_template = extract_template(tokens, kb)
            kb["partner_template"] = partner_template

        # 検索ベースのジェネレーター作成
        search_generator =  get_serach_generator(kb)

        response_prediction = search_generator.template_message(intent, price=price)
        print(response_prediction)
            
        # BaseAgentのstepメソッドが期待する形式で応答を返す
        return response_prediction