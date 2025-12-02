# data_loader.py
import os, warnings
from typing import Dict, List, Tuple
import pandas as pd


class DataLoader:
    """
    AgreeMate baseline system 用のデータ読み込み及び処理ユーティリティ
    交渉データセットの読み込みと前処理を行う
    """
    def __init__(self):
        """データセットのディレクトリへのパスを使用して DataLoader を初期化する"""
        baseline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agreemate_dir = os.path.dirname(baseline_dir)
        self.data_dir = os.path.join(agreemate_dir, "data", "craigslist_bargains")
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.metadata = None

        # データセットのインフォメーションをロードする
        self._load_dataset_info()

    def _load_dataset_info(self):
        """dataset_info.json からデータセットのメタデータを読み込む"""
        try:
            info_path = os.path.join(self.data_dir, "dataset_info.json")
            self.metadata = pd.read_json(info_path)
        except Exception as e:
            print(f"Warning: Could not load dataset info: {str(e)}")
            self.metadata = None

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        全てのデータ分割 (train, test, validation) を読み込む

        Returns:
            (train_df, test_df, val_df) のタプル
        """
        self.train_data = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.val_data = pd.read_csv(os.path.join(self.data_dir, "validation.csv"))

        return self.train_data, self.test_data, self.val_data

    def get_category_stats(self) -> Dict:
        """カテゴリーの分布と統計を取得する"""
        if self.train_data is None:
            self.load_splits()

        stats = {}
        for category in self.train_data['category'].unique():
            cat_data = self.train_data[self.train_data['category'] == category]
            stats[category] = {
                'count': len(cat_data),
                'price_stats': {
                    'min': cat_data['list_price'].min(),
                    'max': cat_data['list_price'].max(),
                    'mean': cat_data['list_price'].mean(),
                    'median': cat_data['list_price'].median()
                },
                'avg_price_delta': cat_data['price_delta_pct'].mean()
            }
        return stats

    def create_negotiation_pair(
        self, 
        row: pd.Series
    ) -> Dict[str, Dict]:
        """
        交渉シナリオの buyer と seller のインフォメーションを作成する

        Args:
            row: シナリオデータを含む DataFrame の行

        Returns:
            buyer と seller のインフォメーションを含む辞書
        """
        return {
            'scenario_id': row['scenario_id'],
            'category': row['category'],
            'item': {
                'title': row['title'],
                'description': row['description'],
                'list_price': row['list_price']
            },
            'buyer': {
                'target_price': row['buyer_target'],
                'relative_price': row['relative_price'],
            },
            'seller': {
                'target_price': row['seller_target'],
                'price_delta_pct': row['price_delta_pct']
            }
        }

    def get_batch(
        self,
        split: str = 'train',
        batch_size: int = 32,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        交渉シナリオのバッチを取得する

        Args:
            split: 使用するデータセット分割 ('train', 'test', 'val')
            batch_size: return するシナリオの数
            shuffle: データをシャッフルするかどうか

        Returns:
            シナリオの辞書のリスト
        """
        # 適切なデータセットを取得
        if split == 'train':
            data = self.train_data
        elif split == 'test':
            data = self.test_data
        elif split == 'val':
            data = self.val_data
        else:
            raise ValueError(f"Unknown split: {split}")

        if data is None:
            self.load_splits()
            data = getattr(self, f"{split}_data")

        # shuffleが指定されていた場合, データをシャッフル
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        # バッチを取得
        batch = data.head(batch_size)

        # 交渉ペアに変換
        scenarios = [self.create_negotiation_pair(row) for _, row in batch.iterrows()]
        
        return scenarios

    def get_category_price_bounds(self, category: str) -> Dict[str, float]:
        """特定のカテゴリの価格統計を取得する"""
        if self.metadata is not None:
            try:
                price_ranges = self.metadata['train']['categories']['price_ranges'][category]
                return {
                    'min': price_ranges['min'],
                    'max': price_ranges['max'],
                    'mean': price_ranges['mean'],
                    'median': price_ranges['median']
                }
            except KeyError:
                warnings.warn(f"No price range data found for category: {category}")

        # データから計算への fallback
        if self.train_data is None:
            self.load_splits()

        cat_data = self.train_data[self.train_data['category'] == category]
        return {
            'min': cat_data['list_price'].min(),
            'max': cat_data['list_price'].max(),
            'mean': cat_data['list_price'].mean(),
            'median': cat_data['list_price'].median()
        }


def test_data_loader():
    """data loader 機能の基本テストをする"""
    loader = DataLoader()

    # data loading のテスト
    train, test, val = loader.load_splits()
    assert len(train) > 0, "Train data empty"
    assert len(test) > 0, "Test data empty"
    assert len(val) > 0, "Validation data empty"

    # 予想される columns を確認する
    expected_columns = [
        'scenario_id', 'split_type', 'category', 'list_price',
        'buyer_target', 'seller_target', 'title', 'description',
        'price_delta_pct', 'relative_price', 'title_token_count',
        'description_length', 'data_completeness', 'price_confidence',
        'has_images'
    ]
    for col in expected_columns:
        assert col in train.columns, f"Missing column: {col}"

    # batch creation のテスト
    batch = loader.get_batch(batch_size=2)
    assert len(batch) == 2, "Incorrect batch size"

    # category stats のテスト
    stats = loader.get_category_stats()
    assert len(stats) > 0, "No category stats generated"

    print("✓ All data loader tests passed")
    return loader

if __name__ == "__main__":
    loader = test_data_loader()