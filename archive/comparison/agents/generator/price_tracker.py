import math
import re
from itertools import chain

from .util import read_pickle
from .entity import Entity, CanonicalEntity
from .tokenizer import tokenize


class PriceScaler(object):
    @classmethod
    def get_price_range(cls, kb):
        '''
        目標価格とボトムライン価格を返す
        '''
        t = kb["my_target"]  # 1
        role = kb["my_role"]

        if role == 'seller':
            b = t * 0.7
        else:
            b = kb["list_price"]

        return b, t

    @classmethod
    def get_parameters(cls, b, t):
        '''
        線形マッピングのパラメータ(傾き, 定数)を返す
        '''
        assert (t - b) != 0
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        return w, c

    @classmethod
    # TODO: これは canonical entities に対して操作されるため, 一貫している必要がある
    def unscale_price(cls, kb, price):
        p = PriceTracker.get_price(price)
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        assert w != 0
        p = (p - c) / w
        p = int(p)
        if isinstance(price, Entity):
            return price._replace(canonical=price.canonical._replace(value=p))
        else:
            return price._replace(value=p)

    @classmethod
    def _scale_price(cls, kb, p):
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        p = w * p + c
        # 2桁に離散化する
        p = float('{:.2f}'.format(p))
        return p

    @classmethod
    def scale_price(cls, kb, price):
        """bottomline=0 , target=1 となるように価格を調整する

        Args:
            price (Entity)
        """
        p = PriceTracker.get_price(price)
        p = cls._scale_price(kb, p)
        return price._replace(canonical=price.canonical._replace(value=p))


class PriceTracker(object):
    def __init__(self, model_path):
        self.model = read_pickle(model_path) # model変数を出力用のpklファイル(第二引数)で初期化

    @classmethod
    def get_price(cls, token):
        try:
            return token.canonical.value
        except:
            try:
                return token.value
            except:
                return None

    @classmethod
    def process_string(cls, token):
        token = re.sub(r'[\$\,]', '', token)
        try:
            if token.endswith('k'):
                token = str(float(token.replace('k', '')) * 1000)
        except ValueError:
            pass
        return token

    def is_price(self, left_context, right_context):
        if left_context in self.model['left'] and right_context in self.model['right']:
            return True
        else:
            return False

    def get_kb_numbers(self, kb):
        title = tokenize(re.sub(r'[^\w0-9\.,]', ' ', kb["title"]))
        description = tokenize(re.sub(r'[^\w0-9\.,]', ' ', ' '.join(kb["description"])))
        numbers = set()
        for token in chain(title, description):
            try:
                numbers.add(float(self.process_string(token)))
            except ValueError:
                continue
        return numbers
    
    def dollar_bond(self, tokens):
        flag = 0
        new_tokens = []
        for i in range(len(tokens)):
            new_token = tokens[i]
            if flag == 1:
                flag = 0
                continue
            if new_token == "$":
                new_token = new_token + tokens[i+1]
                flag = 1
            new_tokens.append(new_token)
        return new_tokens

    def link_entity(self, raw_tokens, kb=None, scale=True, price_clip=None):
        tokens = self.dollar_bond(raw_tokens)
        tokens = ['<s>'] + tokens + ['</s>']
        entity_tokens = []
        if kb:
            kb_numbers = self.get_kb_numbers(kb)
            list_price = kb["list_price"]
        for i in range(1, len(tokens)-1):
            token = tokens[i]
            try:
                number = float(self.process_string(token))
                has_dollar = lambda token: token[0] == '$' or token[-1] == '$'
                # コンテキストの確認
                if not has_dollar(token) and \
                        not self.is_price(tokens[i-1], tokens[i+1]):
                    number = None
                # "infinity" が数値として認識されないようにする
                elif number == float('inf') or number == float('-inf'):
                    number = None
                # 価格が妥当かどうか確認する
                elif kb:
                    if not has_dollar(token):
                        if number > 1.5 * list_price:
                            number = None
                        # おそらく spec number
                        if number != list_price and number in kb_numbers:
                            number = None
                    if number is not None and price_clip is not None:
                        scaled_price = PriceScaler._scale_price(kb, number)
                        if abs(scaled_price) > price_clip:
                            number = None
            except ValueError:
                number = None
            if number is None:
                new_token = token
            else:
                assert not math.isnan(number)
                if scale:
                    scaled_price = PriceScaler._scale_price(kb, number)
                else:
                    scaled_price = number
                new_token = Entity(surface=token, canonical=CanonicalEntity(value=scaled_price, type='price'))
            entity_tokens.append(new_token)
        return entity_tokens