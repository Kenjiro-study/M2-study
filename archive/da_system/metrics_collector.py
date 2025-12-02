# metrics_collector.py
import logging
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .negotiation_runner import NegotiationMetrics
from .strategies import STRATEGIES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class StrategyMetrics:
    """戦略分析のための Metrics"""
    strategy_name: str
    success_rate: float = 0.0
    avg_turns: float = 0.0
    avg_utility: float = 0.0
    language_metrics: dict[str, float] = field(default_factory=dict)

    count: int = field(init=False, default=0)

    def update(self, success: bool, turns: int, utility: float):
        """新しい交渉結果で metrics を更新"""
        self.count += 1

        # running averages を更新する
        self.success_rate = ((self.count-1) * self.success_rate + float(success)) / self.count
        self.avg_turns = ((self.count-1) * self.avg_turns + turns) / self.count
        self.avg_utility = ((self.count-1) * self.avg_utility + utility) / self.count


@dataclass
class NegotiationAnalysis:
    """交渉セッションの完全な分析"""
    # core identifiers
    scenario_id: str
    buyer_model: str
    seller_model: str

    # basic metrics
    duration: float
    turns_taken: int
    final_price: Optional[float]

    # strategy analysis
    buyer_strategy: str
    seller_strategy: str

    # price trajectory
    initial_price: float
    target_prices: dict[str, float] # buyer/seller targets
    price_history: list[float]

    # interaction analysis
    message_lengths: list[int]
    response_times: list[float]

    def compute_metrics(self) -> dict[str, float]:
        """派生 metrics を計算する"""
        metrics = {
            'success': self.final_price is not None,
            'efficiency': self._compute_efficiency(),
            'fairness': self._compute_fairness(),
            'avg_response_time': np.mean(self.response_times),
            'price_convergence': self._compute_convergence()
        }
        return metrics

    def _compute_efficiency(self) -> float:
        """交渉の efficiency スコア (0-1) を計算"""
        if not self.final_price:
            return 0.0

        # 交渉ターン, 時間, 価格変動を考慮する
        max_expected_turns = 20 # from config
        turn_score = 1 - (self.turns_taken / max_expected_turns) # 1 - (かかったターン / 最大ターン)でかかったターン数が少ないほど効率的(1に近い)

        time_per_turn = self.duration / self.turns_taken # 1ターンあたりにかけた時間が長いほど効率的
        time_score = np.exp(-time_per_turn / 30) # 30 sec baseline
        
        # 1. price_history から None を除外した、有効な価格のリストを作成
        valid_prices = [item['price'] for item in self.price_history if item['price'] is not None]

        # 2. 有効な価格が2つ未満の場合、変動を計算できないため directness は最高点 (1.0) とする
        if len(valid_prices) < 2:
            directness = 1.0
        else:
            # 3. ゼロ除算エラーを防ぐ
            start_price = valid_prices[0]
            end_price = valid_prices[-1]
            net_change = abs(end_price - start_price) # 最初と最後の価格の差
            total_movement = np.abs(np.diff(valid_prices)).sum() # 全ての価格変動の差分の合計値

            if net_change == 0:
                # 開始価格と最終価格が同じ場合
                # 全く変動がなければ (例: [100, 100])、完全に直接的 (1.0)
                # 変動があれば (例: [100, 110, 100])、寄り道したとみなし最低点 (0.0)
                directness = 1.0 if total_movement == 0 else 0.0
            else:
                # 元の計算式を実行
                directness = 1 - (total_movement / net_change)

        # 4. スコアがマイナスになる場合があるため、0未満にならないように調整
        directness = max(0.0, directness)

        return np.mean([turn_score, time_score, directness]) # ターンスコア, 時間スコア, 価格変動スコアの平均を取って効率性スコアとする

    def _compute_fairness(self) -> float:
        """交渉の fairness スコア (0-1) を計算"""
        if not self.final_price:
            return 0.0

        # midpoint からの距離
        fair_price = (self.target_prices['buyer'] + self.target_prices['seller']) / 2 # 公平な価格は二人の目標価格の中央値
        price_fairness = 1 - (abs(self.final_price - fair_price) / abs(self.target_prices['buyer'] - self.target_prices['seller'])) # 1 - (最終価格と公平な価格の差の絶対値 / 買い手と売り手の目標価格の差の絶対値)
        
        # 1. デフォルト値を設定 (計算不能な場合に使用)
        concession_balance = 0.5  # どちらとも言えない公平性として0.5を仮置き

        # 2. 履歴が2件以上あり、かつ1番目と2番目の価格がNoneでないかチェック
        if len(self.price_history) >= 2:
            # .get("price") を使うと、キーがなくてもエラーにならず安全
            buyer_initial_price = self.price_history[0].get("price")
            seller_initial_price = self.price_history[1].get("price")

            if buyer_initial_price is not None and seller_initial_price is not None:
                # 3. Noneでないことを確認してから計算を実行
                buyer_movement = abs(self.final_price - buyer_initial_price)
                seller_movement = abs(self.final_price - seller_initial_price)
            
                total_movement = buyer_movement + seller_movement

                # 4. ゼロ除算を防止
                if total_movement == 0:
                    # 双方の譲歩が0なら、バランスは完璧
                    concession_balance = 1.0
                else:
                    concession_balance = 1 - abs(buyer_movement - seller_movement) / total_movement

        return np.mean([price_fairness, concession_balance]) # 価格の変動スコアと価格の公平性スコアの平均が公平性スコア

    def _compute_convergence(self) -> float:
        """C価格収束スコア (0-1) を計算"""
        # 1. price_history から None を除外した、有効な価格のリストを作成
        valid_prices = [item['price'] for item in self.price_history if item['price'] is not None]

        # 2. 有効な価格が2つ未満の場合、収束を計算できないため 0.0 を返す
        #    (開始価格と終了価格の最低2点が必要)
        if len(valid_prices) < 2:
            return 0.0

        # 3. Noneを除外した valid_prices を使って計算する
        price_diffs = np.diff(valid_prices)
    
        # 開始価格と最終価格の差（最短距離）
        ideal_path = abs(valid_prices[-1] - valid_prices[0])
    
        # 価格が実際に動いた距離の合計
        actual_path = np.abs(price_diffs).sum()

        # 4. 計算ロジックを改善
        if actual_path == 0:
            # 全く価格が動かなかった場合、完全に収束しているとみなし 1.0 を返す
            # (ideal_path も 0 になるため、この分岐がないと 0.0 になってしまう)
            return 1.0
        else:
            # 「最短距離」を「実際に動いた距離」で割る
            return ideal_path / actual_path


class MetricsCollector:
    """
    交渉実験から metrics を収集・分析する
    リアルタイム tracking と 実験後の分析の両方を提供する
    """

    def __init__(self):
        """metrics collector の初期化"""
        self.strategy_metrics: dict[str, StrategyMetrics] = {
            name: StrategyMetrics(strategy_name=name)
            for name in STRATEGIES.keys()
        }

        self.model_pairs: dict[str, list[NegotiationAnalysis]] = {}
        self.negotiations: dict[str, NegotiationAnalysis] = {}

    def analyze_negotiation(
        self,
        metrics: NegotiationMetrics,
        buyer_model: str,
        seller_model: str,
        buyer_strategy: str,
        seller_strategy: str,
        buyer_agent: str,
        seller_agent: str,
        scenario_id: str,
        initial_price: float,
        target_prices: dict[str, float]
    ) -> NegotiationAnalysis:
        """完了した交渉対話を分析する"""

        analysis = NegotiationAnalysis(
            scenario_id=scenario_id,
            buyer_model=buyer_model,
            seller_model=seller_model,
            duration=metrics.compute_duration(),
            turns_taken=metrics.turns_taken,
            final_price=metrics.final_price,
            buyer_strategy=buyer_strategy,
            seller_strategy=seller_strategy,
            initial_price=initial_price,
            target_prices=target_prices,
            price_history=[p for p in metrics.messages if 'price' in p],
            message_lengths=[len(m['content']) for m in metrics.messages],
            response_times=[m.get('response_time', 0) for m in metrics.messages]
        )

        # strategy metrics の更新
        computed = analysis.compute_metrics()
        self.strategy_metrics[buyer_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.buyer_utility or 0.0,
        )
        self.strategy_metrics[seller_strategy].update(
            success=computed['success'],
            turns=metrics.turns_taken,
            utility=metrics.seller_utility or 0.0,
        )

        # 分析結果を保存
        self.negotiations[scenario_id] = analysis
        pair_key = f"{buyer_model}:{buyer_strategy}:{buyer_agent}_{seller_model}:{seller_strategy}:{seller_agent}"
        if pair_key not in self.model_pairs:
            self.model_pairs[pair_key] = []
        self.model_pairs[pair_key].append(analysis)

        return analysis

    def get_strategy_summary(self) -> pd.DataFrame:
        """各戦略の summary statistics を取得する"""
        records = []
        for strategy_name, metrics in self.strategy_metrics.items():
            records.append({
                'strategy': strategy_name,
                'success_rate': metrics.success_rate,
                'avg_turns': metrics.avg_turns,
                'avg_utility': metrics.avg_utility,
            })
        return pd.DataFrame.from_records(records)

    def get_model_pair_summary(self) -> pd.DataFrame:
        """各モデルペアの組み合わせの summary statistics を取得する"""
        records = []
        for pair_key, analyses in self.model_pairs.items():
            buyer_model, seller_model = pair_key.split('_')

            # aggregate metrics の計算
            success_rate = np.mean([
                bool(a.final_price) for a in analyses
            ])
            avg_efficiency = np.mean([
                a.compute_metrics()['efficiency'] for a in analyses
            ])
            avg_fairness = np.mean([
                a.compute_metrics()['fairness'] for a in analyses
            ])

            records.append({
                'buyer_model': buyer_model,
                'seller_model': seller_model,
                'num_negotiations': len(analyses),
                'success_rate': success_rate,
                'avg_efficiency': avg_efficiency,
                'avg_fairness': avg_fairness
            })
        return pd.DataFrame.from_records(records)

    def export_analysis(self) -> dict:
        """完全な分析結果をエクスポートする"""
        return {
            'strategy_summary': self.get_strategy_summary().to_dict('records'),
            'model_summary': self.get_model_pair_summary().to_dict('records'),
            'negotiations': {
                sid: analysis.compute_metrics()
                for sid, analysis in self.negotiations.items()
            }
        }


def test_metrics_collector():
    """metrics collector 機能をテストする"""
    from datetime import datetime, timedelta

    collector = MetricsCollector()

    # test metrics を作成
    metrics = NegotiationMetrics(
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        turns_taken=10,
        final_price=150.0,
        buyer_utility=0.8,
        seller_utility=0.7,
        messages=[
            {'role': 'buyer', 'content': 'Offer: $100', 'price': 100},
            {'role': 'seller', 'content': 'Counter: $200', 'price': 200},
            {'role': 'buyer', 'content': 'Accept: $150', 'price': 150}
        ]
    )

    # 分析をテストする
    analysis = collector.analyze_negotiation(
        metrics=metrics,
        buyer_model='llama-3.1-8b',
        seller_model='llama-3.1-8b',
        buyer_strategy='length',
        seller_strategy='fair',
        buyer_agent='damf',
        seller_agent='damf',
        scenario_id='test_1',
        initial_price=200.0,
        target_prices={'buyer': 100.0, 'seller': 180.0}
    )
    assert analysis.turns_taken == 10
    assert analysis.final_price == 150.0

    # サマリーをテストする
    strategy_summary = collector.get_strategy_summary()
    assert len(strategy_summary) == len(STRATEGIES)
    print("strategy_summary: ", strategy_summary)

    model_summary = collector.get_model_pair_summary()
    assert len(model_summary) > 0
    print("model_summary: ", model_summary)

    print("✓ All metrics collector tests passed")
    return collector

if __name__ == "__main__":
    collector = test_metrics_collector()