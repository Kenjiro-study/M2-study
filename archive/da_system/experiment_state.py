# experiment_state.py
"""
AgreeMate における, 実験状態, チェックポイント, 結果の永続性を管理する
実験の進行状況を追跡し, 必要に応じて実験のリカバリーを可能にする
"""
import os, json, logging
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from .config import ExperimentConfig
from .negotiation_runner import NegotiationMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ExperimentState:
    """実験実行における現在の状態"""
    experiment_name: str
    config: ExperimentConfig
    start_time: datetime
    scenarios_total: int
    scenarios_completed: int = 0
    scenarios_failed: int = 0
    last_checkpoint: Optional[datetime] = None

@dataclass
class ModelPairMetrics:
    """特定のモデルペアの組み合わせのための Metrics"""
    buyer_model: str
    seller_model: str
    buyer_hcv: int = 0
    seller_hcv: int = 0
    buyer_hcv_rate: float = 0.0
    seller_hcv_rate: float = 0.0
    num_negotiations: int = 0
    deal_rate: float = 0.0
    avg_turns: float = 0.0
    avg_buyer_utility: float = 0.0
    avg_seller_utility: float = 0.0
    avg_duration: float = 0.0
    avg_fairness: float = 0.0


class ExperimentTracker:
    """
    実験の進行状況をトラッキングし, results persistence を管理する
    チェックポイントの設定と実験のリカバリーを処理する
    """

    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        config: ExperimentConfig
    ):
        """
        experiment tracker を初期化する

        Args:
            output_dir: 出力ファイルのためのディレクトリ
            experiment_name: 一意の実験識別子
            config: 実験設定
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, 'results')
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')

        # ディレクトリを作成
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # 状態を初期化
        self.state = ExperimentState(
            experiment_name=experiment_name,
            config=config,
            start_time=datetime.now(),
            scenarios_total=config.num_scenarios
        )

        # results tracking を初期化
        self.completed_negotiations: dict[str, NegotiationMetrics] = {}
        self.failed_negotiations: dict[str, dict] = {}
        self.model_pair_metrics: dict[str, ModelPairMetrics] = {}        

    def record_completion(
        self,
        scenario_id: str,
        metrics: NegotiationMetrics,
        combination: dict,
    ):
        """成功した交渉について, それが完了したことを記録する"""
        self.completed_negotiations[scenario_id] = metrics
        self.state.scenarios_completed += 1

        # モデルペアの metrics を更新する
        pair_key = f"{combination['buyer_model']}:{combination['buyer_strategy']}:{combination['buyer_agent']}_{combination['seller_model']}:{combination['seller_strategy']}:{combination['seller_agent']}"
        if pair_key not in self.model_pair_metrics:
            self.model_pair_metrics[pair_key] = ModelPairMetrics(
                buyer_model= f"{combination['buyer_model']}:{combination['buyer_strategy']}:{combination['buyer_agent']}",
                seller_model= f"{combination['seller_model']}:{combination['seller_strategy']}:{combination['seller_agent']}"
            )

        pair_metrics = self.model_pair_metrics[pair_key]
        pair_metrics.num_negotiations += 1
        if metrics.buyer_hcv == True:
            pair_metrics.buyer_hcv += 1
        elif metrics.seller_hcv == True:
            pair_metrics.seller_hcv += 1

        pair_metrics.avg_fairness += (metrics.fairness - pair_metrics.avg_fairness) / pair_metrics.num_negotiations # 平均公平性
        pair_metrics.avg_turns += (metrics.turns_taken - pair_metrics.avg_turns) / pair_metrics.num_negotiations # 平均ターン数
        pair_metrics.avg_buyer_utility += (metrics.buyer_utility - pair_metrics.avg_buyer_utility) / pair_metrics.num_negotiations # 平均buyer効用
        pair_metrics.avg_seller_utility += (metrics.seller_utility - pair_metrics.avg_seller_utility) / pair_metrics.num_negotiations # 平均seller効用

        pair_metrics.avg_duration += (metrics.compute_duration() - pair_metrics.avg_duration) / pair_metrics.num_negotiations # (平均交渉時間)

    def record_failure(
        self,
        scenario_id: str,
        error: Exception,
        context: dict
    ):
        """交渉の失敗を記録する"""
        self.failed_negotiations[scenario_id] = {
            'error': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.state.scenarios_failed += 1

    def save_checkpoint(self):
        """実験のチェックポイントを保存する"""
        timestamp = datetime.now()
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"checkpoint_{timestamp:%Y%m%d_%H%M%S}.json"
        )

        checkpoint = {
            'state': asdict(self.state),
            'completed_negotiations': {
                k: asdict(v) for k, v in self.completed_negotiations.items()
            },
            'failed_negotiations': self.failed_negotiations,
            'model_pair_metrics': {
                k: asdict(v) for k, v in self.model_pair_metrics.items()
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
            
        self.state.last_checkpoint = timestamp
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントから実験状態をロードする"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # 状態を復元する
        self.state = ExperimentState(**checkpoint['state'])
        self.completed_negotiations = {
            k: NegotiationMetrics(**v)
            for k, v in checkpoint['completed_negotiations'].items()
        }
        self.failed_negotiations = checkpoint['failed_negotiations']
        self.model_pair_metrics = {
            k: ModelPairMetrics(**v)
            for k, v in checkpoint['model_pair_metrics'].items()
        }

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def save_final_results(self):
        """最終的な実験結果を保存する"""
        results_path = os.path.join(
            self.results_dir,
            f"{self.state.experiment_name}_results.json"
        )

        # summary DataFrame を作成する
        summary_data = []
        for scenario_id, metrics in self.completed_negotiations.items():
            summary_data.append({
                'scenario_id': scenario_id,
                'turns': metrics.turns_taken,
                'duration': metrics.compute_duration(),
                'final_price': metrics.final_price,
                'buyer_utility': metrics.buyer_utility,
                'seller_utility': metrics.seller_utility,
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            os.path.join(self.results_dir, f"{self.state.experiment_name}_summary.csv"),
            index=False
        )

        # complete results package を保存する
        results = {
            'experiment_name': self.state.experiment_name,
            'config': asdict(self.state.config),
            'summary': {
                'total_scenarios': self.state.scenarios_total,
                'completed': self.state.scenarios_completed,
                'failed': self.state.scenarios_failed,
                'duration': (datetime.now() - self.state.start_time).total_seconds()
            },
            'model_pair_metrics': {
                k: asdict(v) for k, v in self.model_pair_metrics.items()
            },
            'completed_negotiations': {
                k: asdict(v) for k, v in self.completed_negotiations.items()
            },
            'failed_negotiations': self.failed_negotiations
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Saved final results to {results_path}")


def test_experiment_tracker():
    """experiment tracker 機能をテストする"""
    from .config import EXPERIMENT_CONFIGS
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # tracker を作成
        tracker = ExperimentTracker(
            output_dir=temp_dir,
            experiment_name="test_experiment",
            config=EXPERIMENT_CONFIGS['baseline']
        )

        # recording completion をテスト
        metrics = NegotiationMetrics(
            start_time=datetime.now(),
            end_time=datetime.now(),
            turns_taken=5,
            final_price=100.0,
            buyer_utility=0.8,
            seller_utility=0.7,
        )

        tracker.record_completion(
            "test_scenario_1",
            metrics,
            "llama3.1",
            "llama3.1"
        )

        # チェックポイントのセーブとロードをテスト
        tracker.save_checkpoint()
        checkpoint_file = os.path.join(
            tracker.checkpoints_dir,
            os.listdir(tracker.checkpoints_dir)[0]
        )

        new_tracker = ExperimentTracker(
            output_dir=temp_dir,
            experiment_name="test_experiment",
            config=EXPERIMENT_CONFIGS['baseline']
        )
        new_tracker.load_checkpoint(checkpoint_file)
        assert new_tracker.state.scenarios_completed == 1

        # 最終結果をテスト
        tracker.save_final_results()
        results_file = os.path.join(
            tracker.results_dir,
            f"{tracker.state.experiment_name}_results.json"
        )
        assert os.path.exists(results_file)

        print("✓ All experiment tracker tests passed")
        return tracker

if __name__ == "__main__":
    tracker = test_experiment_tracker()