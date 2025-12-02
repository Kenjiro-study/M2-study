# extractor.py
import os, dspy
from typing import Dict, List, Optional

# 交渉メッセージから構造化された状態情報を抽出する
class GetPrice(dspy.Signature):
    """Accurately extract the proposed price information written in the negotiation message. Do not extract prices from statements that negate the other party's price, but only extract prices when you are proposing a price."""
    message_content: str = dspy.InputField(desc="This is the message from which you want to extract the price. We will extract the proposed price from this sentence.")

    extracted_price: Optional[float] = dspy.OutputField(desc="The price proposed in the message_content, if any")
    reasoning: str = dspy.OutputField(desc="explanation of extraction")
    score: float = dspy.OutputField(desc="Confidence score (0.0 to 1.0) for the extraction.") # ← score フィールドを追加

def _create_train_examples():
    # 学習データを返すヘルパー関数
    return[     
        dspy.Example(message_content="was hoping for something around 8,000 dollars but it's really no use to me so i'd be willing to go lower", extracted_price=8000, reasoning="The seller offers $8000, but says there's room for negotiation and is willing to go lower.").with_inputs("message_content"),
        dspy.Example(message_content="That's too low, but I can get it down to 5k.", extracted_price=5000, reasoning="The seller is offering a counter price. The k in 5k stands for thousand.").with_inputs("message_content"),
        dspy.Example(message_content="I've seen that item on sale for $80.", extracted_price=80, reasoning="Buyers implicitly suggest their desired price by mentioning the price at other stores.").with_inputs("message_content"),
        dspy.Example(message_content="This battery is $550 brand new. It has a large capacity and is very easy to use.", extracted_price=550, reasoning="The seller is implicitly communicating their asking price by mentioning the price when the item was new.").with_inputs("message_content"),
        dspy.Example(message_content="Sorry, I don't have any money right now, so I can't give you more than $2,150.", extracted_price=2150, reasoning="Buyers communicate their maximum asking price with reasons").with_inputs("message_content")
    ]

def price_extraction_metric(example, prediction, _trace=None):
    """
    予測された extracted_price が、教師データの正解 (example.extracted_price) と
    一致しているかを評価するメトリック。
    """
    # 価格が None の場合も考慮して、単純な一致比較を行う
    return prediction.extracted_price == example.extracted_price

class PriceExtractor:
    """
    AgreeMate baseline negotiation system の Human Agent
    人間が買い手側と売り手側のどちらかのエージェントの役割を果たす場合の機能と抽象メソッドを定義します。
    """
    _compiled_extractor = None

    @classmethod
    def _get_compiled_extractor(cls, lm: dspy.LM):
        if cls._compiled_extractor is None:
            from dspy.teleprompt import BootstrapFewShot

            extractor = dspy.ChainOfThought(GetPrice)
            train_examples = _create_train_examples()
            
            optimizer = BootstrapFewShot(max_bootstrapped_demos=5, metric=price_extraction_metric)
            # コンパイルを実行
            with dspy.context(lm=lm):
                cls._compiled_extractor = optimizer.compile(student=extractor, trainset=train_examples)
        return cls._compiled_extractor

    def __init__(
        self,
        lm: dspy.LM,
    ):
        """
        price extractor を初期化する

        Args:
            lm: 価格抽出のための DSPy 言語モデル 
        """
        self.lm = lm

        # すべてのモジュールで提供された言語モデルを使用するように DSPy を構成する
        #dspy.settings.configure(lm=lm)

        # predictor modules のセットアップ
        #self.price_extractor = dspy.ChainOfThought(GetPrice) # 未コンパイルの抽出器
        self.compiled_extractor =  self._get_compiled_extractor(lm=self.lm)


def test_extractor():
    import pandas as pd
    """PriceExtractor の機能をテストする"""
    baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agreemate_dir = os.path.dirname(baseline_dir)
    pretrained_dir = os.path.join(agreemate_dir, "models", "pretrained")
    
    test_lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
        cache_dir=pretrained_dir,
    )

    agent = PriceExtractor(
        lm=test_lm,
    )
    df = pd.read_csv('archive/da_system/agents/parser/data/cb_dataset_3500~3999.csv')
    df = df.drop("Unnamed: 0", axis=1)
    df = df.head(1000)
    df = df[df["meta_text"].isin(["counter-price", "init-price", "insist"])]
    print(df)

    df_text = df["text"]

    compiled = []
    #nomal = []

    with dspy.context(lm=agent.lm):
        for i, item in enumerate(df_text):
        
            print(i)
            compiled_extraction = agent.compiled_extractor(
                message_content=item,
            )
            #extraction = agent.price_extractor(
                #message_content=item,
            #)
            compiled.append(compiled_extraction.extracted_price)
            #nomal.append(extraction.extracted_price)

    df["compiled"] = compiled
    #df["nomal"] = nomal
    df.to_csv('output5.csv', index=False, encoding='utf-8')

    """
        with dspy.context(lm=agent.lm):
        while True:
            user_input = input("入力(exitで終了): ")

            if user_input == "exit":
                break
            else:
                compiled_extraction = agent.compiled_extractor(
                    message_content=user_input,
                )
                #extraction = agent.price_extractor(
                    #message_content=user_input,
                #)
                print(f"compiled → price: {compiled_extraction.extracted_price}, reason: {compiled_extraction.reasoning}")
                #print(f"normal → price: {extraction.extracted_price}, reason: {extraction.reasoning}")

"""

if __name__ == "__main__":
    agent = test_extractor()