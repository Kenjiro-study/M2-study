# scenario_manager.py
"""
AgreeMate system の Craigslist Bargains データセットからの交渉シナリオを管理する

このモジュールは, CSVファイルからの交渉シナリオの読み込みと準備を行う
各シナリオには, 次のような実際の買い手と売り手の交渉データが含まれている
- 商品の詳細 (title, description, category)
- 価格情報 (list price, buyer target, seller target)
- 品質指標 (completeness, confidence)

マネージャーは, シナリオが適切にロードされ, 交渉実験で使用するために検証されていることを
確認し, 元のデータセットとの整合性を維持しながら, 実験のために簡単にアクセスできるように
する
"""
import logging
from typing import Dict, List, Optional
import pandas as pd

from .utils.data_loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NegotiationScenario:
    """
    データセットから抽出した完全な交渉シナリオを示す

    CSVファイルからの実際のシナリオデータを使用したことで, 買い手と売り手のエージェントが
    交渉を行うために必要な全ての情報が含まれている
    """

    def __init__(self, row: pd.Series):
        """
        元となるデータセットからシナリオを初期化する
        train, validation, test.csvの情報を取得している！

        Args:
            row: CSV から取得したシナリオデータを含むpandasシリーズ
        """
        # core identification
        self.scenario_id = row['scenario_id']
        self.split_type = row['split_type']
        self.category = row['category']

        # item information
        self.title = row['title']
        self.description = row['description']

        # price information
        self.list_price = float(row['list_price'])
        self.buyer_target = float(row['buyer_target'])
        self.seller_target = float(row['seller_target'])

        # quality metrics
        self.price_delta_pct = float(row['price_delta_pct'])
        self.relative_price = float(row['relative_price'])
        self.data_completeness = float(row['data_completeness'])
        self.price_confidence = bool(row['price_confidence'])
        self.has_images = bool(row['has_images'])

        # 価格の関係を検証する
        if not (self.buyer_target <= self.list_price and 
                self.seller_target <= self.list_price):
            raise ValueError(f"Invalid price relationships in scenario {self.scenario_id}")

    def get_buyer_context(self) -> Dict:
        """買い手の視点からシナリオのcontextを取得する"""
        return {
            'role': 'buyer',
            'scenario_id': self.scenario_id,
            'category': self.category,
            'item': {
                'title': self.title,
                'description': self.description,
                'list_price': self.list_price
            },
            'target_price': self.buyer_target
        }

    def get_seller_context(self) -> Dict:
        """売り手の視点からシナリオのcontextを取得する"""
        return {
            'role': 'seller',
            'scenario_id': self.scenario_id,
            'category': self.category,
            'item': {
                'title': self.title,
                'description': self.description,
                'list_price': self.list_price
            },
            'target_price': self.seller_target
        }


class ScenarioManager:
    """
    データセットからの交渉シナリオの読み込みと選択を管理する
    """

    def __init__(self, data_loader: DataLoader):
        """
        シナリオマネージャーを初期化する

        Args:
            data_loader: CSV ファイルにアクセスするための DataLoader インスタンス
        """
        self.data_loader = data_loader

        # 全ての分割データ(train, validation, test)をロードする
        self.train_df, self.test_df, self.val_df = self.data_loader.load_splits()
        logger.info(f"Loaded {len(self.train_df)} training, {len(self.test_df)} test, "
                   f"and {len(self.val_df)} validation scenarios")

        # 作成されたシナリオを追跡する
        self.scenarios: Dict[str, NegotiationScenario] = {}

    def create_evaluation_batch(
        self,
        split: str = 'test',
        size: Optional[int] = None,
        balanced_categories: bool = True,
        category: Optional[str] = None
    ) -> List[NegotiationScenario]:
        """
        評価用のシナリオのバッチを作成する

        Args:
            split: 使用するデータセットの分割 ('train', 'test', 'val')
            size: return するシナリオの数 (default: 全てのsplit)
            balanced_categories: カテゴリーのバランスを確保するかどうか
            category: 使用する Optional specific カテゴリー

        Returns:
            NegotiationScenario インスタンスのリスト
        """
        # 適切なデータフレームを取得
        if split == 'train':
            df = self.train_df
        elif split == 'test':
            df = self.test_df
        elif split == 'val':
            df = self.val_df
        else:
            raise ValueError(f"Unknown split: {split}")

        # categoryが指定されている場合は, そのカテゴリーでフィルタリングする
        if category:
            df = df[df['category'] == category]
            if len(df) == 0:
                raise ValueError(f"No scenarios found for category: {category}")

        # バランスの取れたカテゴリー選択を行う
        if balanced_categories and not category:
            scenarios = []
            categories = df['category'].unique()

            if size:
                per_category = size // len(categories)
                remainder = size % len(categories)
            else:
                # カテゴリーのバランスを保ちながら, あらゆるシナリオを活用する
                min_per_cat = df['category'].value_counts().min()
                per_category = min_per_cat
                remainder = 0

            for cat in categories:
                cat_df = df[df['category'] == cat]
                count = per_category + (1 if remainder > 0 else 0)
                remainder -= 1

                # sample scenarios
                cat_scenarios = cat_df.sample(n=min(count, len(cat_df)))
                for _, row in cat_scenarios.iterrows():
                    scenario = NegotiationScenario(row)
                    self.scenarios[scenario.scenario_id] = scenario
                    scenarios.append(scenario)

            return scenarios

        else:
            # 単純な random sampling
            if size:
                df = df.sample(n=min(size, len(df)))

            scenarios = []
            for _, row in df.iterrows():
                scenario = NegotiationScenario(row)
                self.scenarios[scenario.scenario_id] = scenario
                scenarios.append(scenario)

            return scenarios

    def get_scenario(self, scenario_id: str) -> NegotiationScenario:
        """
        ID を使用して特定のシナリオを取得する

        Args:
            scenario_id: シナリオ識別子

        Returns:
            NegotiationScenario インスタンス
        """
        # キャッシュされたシナリオが利用可能な場合は return
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id]

        # データフレーム内のシナリオを見つける
        for df in [self.train_df, self.test_df, self.val_df]:
            scenario_df = df[df['scenario_id'] == scenario_id]
            if len(scenario_df) == 1:
                scenario = NegotiationScenario(scenario_df.iloc[0])
                self.scenarios[scenario_id] = scenario
                return scenario

        raise ValueError(f"Scenario not found: {scenario_id}")


def test_scenario_manager():
    """ScenarioManager 機能をテストする"""
    # data loader で初期化する
    data_loader = DataLoader()
    manager = ScenarioManager(data_loader)

    # 基本的なシナリオ作成をテストする
    test_batch = manager.create_evaluation_batch(
        split='test',
        size=10,
        balanced_categories=True
    )
    assert len(test_batch) == 10

    # カテゴリーのバランスを確認する
    categories = set(s.category for s in test_batch)
    assert len(categories) > 1

    # 個々のシナリオの読み込みをテストする
    scenario = manager.get_scenario(test_batch[0].scenario_id)
    assert scenario.scenario_id == test_batch[0].scenario_id
    assert scenario.buyer_target <= scenario.list_price
    assert scenario.seller_target <= scenario.list_price

    # context 生成をテストする
    buyer_context = scenario.get_buyer_context()
    seller_context = scenario.get_seller_context()
    assert buyer_context['role'] == 'buyer'
    assert seller_context['role'] == 'seller'
    assert buyer_context['target_price'] == scenario.buyer_target
    assert seller_context['target_price'] == scenario.seller_target

    print("✓ All scenario manager tests passed")
    return manager

if __name__ == "__main__":
    manager = test_scenario_manager()