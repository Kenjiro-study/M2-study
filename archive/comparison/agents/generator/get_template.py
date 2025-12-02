from nltk.corpus import stopwords
from .tokenizer import tokenize
from .entity import is_entity
from .price_tracker import PriceTracker

stopwords = set(stopwords.words('english')) # 頻繁に現れるため検索処理から除外すべき単語(stopwords)の設定
stopwords.update(['may', 'might', 'rent', 'new', 'brand', 'low', 'high', 'now', 'available']) # 交渉対話用にstopwordsを追加

def is_price_token(token):
    return is_entity(token)

def parse_price(price_token, kb):
    price = price_token.canonical.value
    if price == kb["list_price"]:
        type_ = 'listing_price'
    elif price == kb["my_price"]:
        type_ = 'my_price'
    elif price == kb["partner_price"]:
        type_ = 'partner_price'
    else:
        type_ = 'price'
    canonical = price_token.canonical._replace(type=type_)
    price_token = price_token._replace(canonical=canonical)
    return price_token

def parse_prices(tokens, kb):
    tokens = [parse_price(token, kb) if is_price_token(token) else token for token in tokens]
    return tokens

def parse_title(tokens, kb):
    title = kb["title"].lower().split() # データのscenario.kbs.item.title(要は交渉する商品のタイトル)の文を小文字化してトークナイズ
    new_tokens = []
    for token in tokens:
        if token in title and not token in stopwords: # 発話内のトークンがtitleに含まれているかつ, ストップワードに含まれていない場合
            if len(new_tokens) == 0 or new_tokens[-1] != '{title}': # 発話内の最初のトークンか, 一つ前のトークンがプレースホルダーに置き換えられていない場合
                new_tokens.append('{title}') # {title}プレースホルダに置き換え
        else:
            new_tokens.append(token) # 条件を満たさない場合は置き換えずにそのまま
    return new_tokens

def extract_template(tokens, kb):
        tokens = parse_prices(tokens, kb) # 価格を検出してトークナイズ
        #print("tokens: ", tokens) #####
        tokens = ['{%s}' % token.canonical.type if is_price_token(token) else token for token in tokens] # 価格の部分を{price}のプレースホルダに置き換える
        #print("tokens: ", tokens) #####
        tokens = parse_title(tokens, kb) # 発話内における商品説明のタイトルにも出てくる単語を{title}のプレースホルダに置き換える
        return tokens

def test_get_template():
    model_path = "archive/da_system/agents/generator/price_tracker.pkl"
    price_tracker = PriceTracker(model_path)
    #partner_text = "Hello, this charger is great as it can charge 2 devices at the same time. The price is only $10"
    partner_text = "yes."
    kb = {
        "title": "Verizon Car Charger with Dual Output Micro USB and LED Light",
        "description": "Charge two devices simultaneously on the go. This vehicle charger with an additional USB port delivers enough power to charge two devices at once. The push-button activated LED connector light means no more fumbling in the dark trying to connect your device. Auto Detect IC Technology automatically detects the device type and its specific charging needs for improved compatibility. And the built-in indicator light illuminates red to let you know the charger is receiving power and the power socket is working properly.",
        "list_price": 10.0,
        "my_price": 6.0,
        "partner_price": None,
        "my_role": "buyer"
    }

    tokens = price_tracker.link_entity(tokenize(partner_text), kb=kb, scale=False) # craigslistbargain/core/price-tracker.pyで価格を検出してトークナイズ
    print("tokens: ", tokens)
    template = extract_template(tokens, kb) # タイトルやprice-trackerを使用して発話の一部をプレースホルダに置き換えてテンプレートを作成
    print("template: ", template)


if __name__ == "__main__":
    manager = test_get_template()