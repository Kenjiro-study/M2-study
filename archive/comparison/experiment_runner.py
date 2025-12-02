# experiment_runner.py
import logging, asyncio
from typing import Dict, List, Optional
from pathlib import Path

from .config import EXPERIMENT_CONFIGS
from .dspy_manager import DSPyManager
from .scenario_manager import ScenarioManager
from .negotiation_runner import NegotiationRunner, NegotiationConfig
from .experiment_state import ExperimentTracker
from .metrics_collector import MetricsCollector
from .utils.data_loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ExperimentRunner:
    """
    AgreeMate baseline system のための主要な実験 orchestrator
    交渉におけるエージェントの自律性を維持しながら、実験のセットアップ, 実行, 
    結果収集を処理する
    """

    def __init__(
        self,
        config_name: str,
        output_dir: str,
        experiment_name: Optional[str] = None
    ):
        """
        experiment runner の初期化

        Args:
            config_name: EXPERIMENT_CONFIGS の config 名
            output_dir: 実験の出力のためのディレクトリ
            experiment_name: この実行の任意の一意の名前
        """
        if config_name not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}")
        self.config = EXPERIMENT_CONFIGS[config_name]

        # 出力用ディレクトリのセットアップ
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # components の初期化
        self.data_loader = DataLoader()
        self.scenario_manager = ScenarioManager(self.data_loader)
        self.dspy_manager = DSPyManager()
        self.negotiation_runner = NegotiationRunner(self.dspy_manager)
        self.metrics = MetricsCollector()

        # 実験状態 tracker の初期化
        self.tracker = ExperimentTracker(
            output_dir=str(self.output_dir),
            experiment_name=experiment_name or config_name,
            config=self.config
        )

        self.model_combinations = self._generate_model_combinations(config_name)

    def _generate_model_combinations(self, config_name) -> List[Dict]:
        """テストする全てのモデルと戦略の組み合わせを生成する"""
        combinations = []
        # 対人実験のmodel_combinationを追加 2025/8/4
        if config_name == "human_negotiation":
            for buyer_model in self.config.models:
                for buyer_strategy in self.config.strategies:
                    combinations.append({
                                'buyer_model': buyer_model,
                                'seller_model': "human",
                                'buyer_strategy': buyer_strategy,
                                'seller_strategy': "free"
                            })
            for seller_model in self.config.models:
                for seller_strategy in self.config.strategies:
                    combinations.append({
                                'buyer_model': "human",
                                'seller_model': seller_model,
                                'buyer_strategy': "free",
                                'seller_strategy': seller_strategy
                            })
        else:
            for buyer_model in self.config.models:
                for seller_model in self.config.models:
                    for buyer_strategy in self.config.strategies:
                        for seller_strategy in self.config.strategies:
                            combinations.append({
                                'buyer_model': buyer_model,
                                'seller_model': seller_model,
                                'buyer_strategy': buyer_strategy,
                                'seller_strategy': seller_strategy
                            })
        return combinations

    async def _run_single_combination(
        self,
        combination: Dict,
        scenarios: List[str]
    ):
        """一つの model/strategy の組み合わせについて交渉を実行する"""
        configs = []
        for scenario_id in scenarios:
            scenario = self.scenario_manager.get_scenario(scenario_id)
            configs.append(
                NegotiationConfig(
                    scenario=scenario,
                    buyer_model=combination['buyer_model'],
                    seller_model=combination['seller_model'],
                    buyer_strategy=combination['buyer_strategy'],
                    seller_strategy=combination['seller_strategy'],
                    max_turns=self.config.max_turns,
                    turn_timeout=self.config.turn_timeout
                )
            )

        # 交渉を実行
        results = await self.negotiation_runner.run_batch(configs)

        # 結果を処理する
        for scenario_id, metrics in results.items():
            if metrics.final_price is not None: # 交渉が成功した場合
                self.tracker.record_completion(
                    scenario_id,
                    metrics,
                    combination
                )

                # 交渉の分析
                self.metrics.analyze_negotiation(
                    metrics=metrics,
                    buyer_model=combination['buyer_model'],
                    seller_model=combination['seller_model'],
                    buyer_strategy=combination['buyer_strategy'],
                    seller_strategy=combination['seller_strategy'],
                    scenario_id=scenario_id,
                    initial_price=configs[0].scenario.list_price,
                    target_prices={
                        'buyer': configs[0].scenario.buyer_target,
                        'seller': configs[0].scenario.seller_target
                    }
                )
            else: # 交渉が失敗した場合
                self.tracker.record_failure(
                    scenario_id,
                    Exception("No agreement reached"),
                    combination
                )
        
        pair_key = f"{combination['buyer_model']}:{combination['buyer_strategy']}_{combination['seller_model']}:{combination['seller_strategy']}"
        if pair_key in self.tracker.model_pair_metrics:
            pair_metrics = self.tracker.model_pair_metrics[pair_key]
            pair_metrics.deal_rate += (pair_metrics.num_negotiations - pair_metrics.deal_rate) / len(results)

    async def run(self):
        """完全な実験を実行する"""
        logger.info(f"Starting experiment with config: {self.config}")

        # シナリオを取得する
        scenarios = self.scenario_manager.create_evaluation_batch(
            split='test',
            size=self.config.num_scenarios,
            balanced_categories=True
        )
        scenario_ids = [s.scenario_id for s in scenarios]

        # 全ての組み合わせを実行する
        for combination in self.model_combinations:
            logger.info(f"Running combination: {combination}")
            await self._run_single_combination(combination, scenario_ids)

            # 各組み合わせ後のチェックポイント
            self.tracker.save_checkpoint()

        # 最終的な結果をセーブする
        self.tracker.save_final_results()

        # 分析結果をエクスポート
        analysis = self.metrics.export_analysis()
        analysis_path = self.output_dir / 'analysis.json'
        with open(analysis_path, 'w') as f:
            import json
            json.dump(analysis, f, indent=2)

        return {
            'config': self.config,
            'results': self.tracker.state,
            'analysis': analysis
        }


async def main():
    """コマンドラインから実験を実行"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        config_name=args.config,
        output_dir=args.output,
        experiment_name=args.name
    )
    results = await runner.run()
    logger.info(f"Experiment complete: {results}")

if __name__ == "__main__":
    asyncio.run(main())