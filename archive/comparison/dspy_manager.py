# dspy_manager.py
import os, logging, dspy
from typing import Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .config import MODEL_CONFIGS, ModelConfig
from .strategies import STRATEGIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class DSPyLMConfig:
    """DSPy 言語モデルの拡張構成"""
    base_config: ModelConfig
    strategy_name: Optional[str] = None
    agent_name: Optional[str] = None
    role: Optional[str] = None # 'buyer' or 'seller'

    def get_context_config(self) -> Dict:
        """戦略と役割固有の構成を取得"""
        config = {
            'temperature': self.base_config.temperature,
            'max_tokens': self.base_config.max_tokens
        }

        if self.strategy_name:
            strategy = STRATEGIES[self.strategy_name]
            # 戦略に基づいて温度を調整する
            if strategy['risk_tolerance'] == 'high':
                config['temperature'] *= 1.2
            elif strategy['risk_tolerance'] == 'low':
                config['temperature'] *= 0.8

            # コミュニケーションスタイルに基づいて最大トークンを調整する
            if strategy['communication_style'].startswith('Clear'):
                config['max_tokens'] = min(config['max_tokens'], 256)

        return config


class DSPyManager:
    """
    AgreeMate 交渉システム向けに, 戦略を考慮した構成を備えた DSPy 言語モデルを管理する
    初期化, キャッシング, 並列実行コンテキスト, 及び戦略固有の構成を処理する
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """オプションのキャッシュディレクトリを使用して DSPy マネージャーを初期化する"""
        self.model_configs = MODEL_CONFIGS
        self.lm_cache: Dict[str, dspy.LM] = {}
        self.context_configs: Dict[str, DSPyLMConfig] = {}

        # キャッシュディレクトリのセットアップ
        if cache_dir:
            os.environ['DSPY_CACHE_DIR'] = cache_dir

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _create_lm(self, model_key: str, config: DSPyLMConfig) -> dspy.LM:
        """configuration を使用して新しい DSPy 言語モデルインスタンスを作成する"""
        context_config = config.get_context_config()

        try:
            return dspy.LM(
                model=config.base_config.name,
                temperature=context_config['temperature'],
                max_tokens=context_config['max_tokens'],
                provider="ollama",
                cache=True # 効率化のために常にキャッシュする
            )
        except Exception as e:
            logger.error(f"Failed to create LM for {model_key}: {str(e)}")
            raise
        
    def get_lm(
        self,
        model_key: str,
        strategy_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        role: Optional[str] = None
    ) -> dspy.LM:
        """
        特定の構成を持つ DSPy 言語モデルを取得, または作成する

        Args:
            model_key: MODEL_CONFIGS のキー
            strategy_name: 特定の config のための戦略名 (オプション)
            role: 特定の config のための役割(buyer/seller) (オプション)

        Returns:
            構成済みの DSPy 言語モデルインスタンス
        """
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        # context-specific key の作成
        context_key = f"{model_key}_{strategy_name}_{agent_name}_{role}"

        # まずキャッシュをチェックする
        if context_key in self.lm_cache:
            return self.lm_cache[context_key]

        # 新しい configuration を作成する
        config = DSPyLMConfig(
            base_config=self.model_configs[model_key],
            strategy_name=strategy_name,
            agent_name=agent_name,
            role=role
        )
        self.context_configs[context_key] = config

        # 新しい言語モデル(LM)を作成してキャッシュする
        lm = self._create_lm(model_key, config)
        self.lm_cache[context_key] = lm

        return lm
    
    def get_extractor_lm(self):
        lm = dspy.LM(
            model="ollama/llama3.3:70b",
            #model="ollama/llama3.1",
            provider="ollama",
            cache=True,
        )
        return lm

    async def run_parallel(self, tasks: list) -> list:
        """複数の LM タスクを並列に実行する"""
        futures = []
        for task in tasks:
            future = self.executor.submit(task)
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                results.append(None)

        return results

    def configure_negotiation(
        self,
        buyer_model: str,
        seller_model: str,
        buyer_strategy: str,
        seller_strategy: str
    ) -> tuple:
        """
        交渉ペア用に DSPy LMs を構成する

        Returns:
            (buyer_lm, seller_lm) のタプル
        """
        buyer_lm = self.get_lm(
            buyer_model,
            strategy_name=buyer_strategy,
            role='buyer'
        )

        seller_lm = self.get_lm(
            seller_model,
            strategy_name=seller_strategy,
            role='seller'
        )

        return buyer_lm, seller_lm

    def clear_cache(self):
        """キャッシュされた LMs を全てクリアする"""
        self.lm_cache.clear()
        self.context_configs.clear()


def test_dspy_manager():
    """Test DSPy manager 機能をテストする"""
    manager = DSPyManager()

    # 基本的な LM 生成のテスト
    lm = manager.get_lm("llama-3.1-8b")
    assert lm is not None

    # strategy-specific configuration のテスト
    lm_utility = manager.get_lm(
        "llama-3.1-8b",
        strategy_name="utility",
        role="buyer"
    )
    assert lm_utility is not None

    # negotiation pair configuration のテスト
    buyer_lm, seller_lm = manager.configure_negotiation(
        "llama-3.1-8b",
        "llama-3.1-8b",
        "length",
        "fair"
    )
    assert buyer_lm is not None
    assert seller_lm is not None

    print("✓ All DSPy manager tests passed")
    return manager

if __name__ == "__main__":
    manager = test_dspy_manager()