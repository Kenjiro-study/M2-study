import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .utterance import LogicalForm as LF
from .utterance import Utterance
from .get_template import extract_template
from .price_tracker import PriceTracker
from .tokenizer import tokenize, detokenize

class SearchGenerator:

    def __init__(self, templates, kb, sample_temperature=10.0):
        self.vectorizer = TfidfVectorizer()
        self.templates = templates
        self.kb = kb
        self.sample_temperature = sample_temperature
        self.used_templates = set()
        self.vectorizer = TfidfVectorizer()
        self.build_tfidf()

    def build_tfidf(self):
        documents = self.templates['context'].values
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
    
    def _add_filter(self, locs, cond):
        locs.append(locs[-1] & cond)

    def _select_filter(self, locs):
        for loc in locs[::-1]:
            if np.sum(loc) > 0:
                return loc
        return locs[0]

    def get_filter(self, used_templates=None, category=None, role=None, context_tag=None, tag=None, **kwargs):
        if used_templates:
            locs = (~self.templates.id.isin(used_templates)) # used_templatesに書かれているidのテンプレート以外のテンプレート(つまり使ってないテンプレート)をlocに格納
            if np.sum(locs) <= 0:
                locs = self.templates.id.notnull()
        else:
            locs = self.templates.id.notnull() # 使用済みテンプレートがなかったり, 全てテンプレートが使用済みだった場合は, 全てのテンプレートを使用可能として返す
        locs = [locs]
        self._add_filter(locs, self.templates.role == role)
        self._add_filter(locs, self.templates.category == category)
        if tag:
            self._add_filter(locs, self.templates.tag == tag)
        if context_tag:
            self._add_filter(locs, self.templates.context_tag == context_tag)
        return self._select_filter(locs)

    # 「取得する」という意味の関数. おそらくこれがジェネレータ本体
    def retrieve(self, context, used_templates=None, topk=20, T=1., **kwargs):
        # なんかめっちゃムズイけどいつか解読しなきゃいけない気がしなくもない
        loc = self.get_filter(used_templates=used_templates, **kwargs)
        if loc is None:
            return None
        #print("loc: ", loc)

        if isinstance(context, list):
            # リストにトークンごとに分けられている文を一つの文章に戻す
            # ['i', 'dont', 'exactly', 'have', '{listing_price}', ',', 'would', 'you', 'accept', '{price}', '?']
            # i dont exactly have {listing_price}, would you accept {price}?
            context = detokenize(context)
        features = self.vectorizer.transform([context])
        scores = self.tfidf_matrix * features.T
        scores = scores.todense()[loc]
        scores = np.squeeze(np.array(scores), axis=1)
        ids = np.argsort(scores)[::-1][:topk]

        candidates = self.templates[loc]
        candidates = candidates.iloc[ids]
        rows = self.templates[loc]
        rows = rows.iloc[ids]
        logp = rows['logp'].values

        return self.sample(logp, candidates, T)

    # なんかソフトマックスとか使ってるから, 多分ここで確率出してランダムに文章を決定してる気がする
    def sample(self, scores, templates, T=1.):
        probs = self.softmax(scores, T=T) # 確率に変換
        template_id = np.random.multinomial(1, probs).argmax() # その確率分布に従ってくじを引く
        template = templates.iloc[template_id] # くじで当たったidのテンプレートを使用する
        return template

    def softmax(self, scores, T=1.):
        exp_scores = np.exp(scores / T)
        return exp_scores / np.sum(exp_scores)

    def retrieve_response_template(self, tag, **kwargs):
        context_tag = self.kb["partner_intent"] if self.kb["partner_intent"] != 'unknown' else None # 相手のインテント
        context = self.kb["partner_template"] # 相手の発言のテンプレート
        template = self.retrieve(context, tag=tag, context_tag=context_tag, used_templates=self.used_templates, T=self.sample_temperature, **kwargs)
        if template is None:
            return None
        #print("template: ", template)
        self.used_templates.add(template['id'])
        template = template.to_dict()
        #print("template: ", template)
        template['source'] = 'rule'
        #print("template: ", template)
        return template

    def fill_template(self, template, price=None):
        return template.format(title=self.kb["title"], price=(price or ''), listing_price=self.kb["list_price"], partner_price=(self.kb["partner_price"] or ''), my_price=(self.kb["my_price"] or ''))

    def template_message(self, intent, price=None):
        template = self.retrieve_response_template(intent, category=self.kb["category"], role=self.kb["my_role"])
        if '{price}' in template['template']:
            price = price or self.kb["my_price"]
        else:
            price = None
        lf = LF(intent, price=price)
        text = self.fill_template(template['template'], price=price)
        #utterance = Utterance(raw_text=text, logical_form=lf, template=template)
        return {
            "response": text,
            "intent": intent,
            "price": price
        }
    
def get_template():
    file_path = 'archive/da_system/agents/generator/template.csv'
    return pd.read_csv(file_path)

def get_serach_generator(kb):
    template = get_template()
    search_generator =  SearchGenerator(template, kb)
    return search_generator

def test_search_generator():
    template = get_template()
    action = "unknown" # マネージャーから取得した自分のインテント
    price = None # マネージャーから取得した自分の提案価格
    kb = {
        "title": "Verizon Car Charger with Dual Output Micro USB and LED Light",
        "description": "Charge two devices simultaneously on the go. This vehicle charger with an additional USB port delivers enough power to charge two devices at once. The push-button activated LED connector light means no more fumbling in the dark trying to connect your device. Auto Detect IC Technology automatically detects the device type and its specific charging needs for improved compatibility. And the built-in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly.",
        "category": "phone",
        "list_price": 10.0,
        "my_price": 6.0, # 自分の今の提案価格(提案を開始するまでは目標価格)
        "partner_price": None, # 相手の今の提案価格(提案が来るまではNone)
        "my_role": "buyer",
        "partner_intent": "init-price",
    }
    model_path = "archive/da_system/agents/generator/price_tracker.pkl"
    price_tracker = PriceTracker(model_path)
    partner_text = "Hello, this charger is great as it can charge 2 devices at the same time. The price is only $10"

    tokens = price_tracker.link_entity(tokenize(partner_text), kb=kb, scale=False) # craigslistbargain/core/price-tracker.pyで価格を検出してトークナイズ
    #print("tokens: ", tokens)
    partner_template = extract_template(tokens, kb) # タイトルやprice-trackerを使用して発話の一部をプレースホルダに置き換えてテンプレートを作成
    #print("partner_template: ", partner_template)
    
    kb["partner_template"] = partner_template

    search_generator =  SearchGenerator(template, kb)

    response = search_generator.template_message(action, price=price)
    #print("response: ", response)

if __name__ == "__main__":
    manager = test_search_generator()