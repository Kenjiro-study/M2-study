# negotiation_runner.py
import logging, asyncio
from typing import Optional
from dataclasses import dataclass, field # 2025/7/18 field追加
from datetime import datetime

from .agents.buyer import BuyerAgent
from .agents.seller import SellerAgent
from .agents.search_buyer import SearchBaseBuyerAgent
from .agents.search_seller import SearchBaseSellerAgent
from .agents.simple_llm_buyer import SimpleLLMBuyerAgent
from .agents.simple_llm_seller import SimpleLLMSellerAgent
from .agents.all_one_buyer import AllinOneLLMBuyerAgent
from .agents.all_one_seller import AllinOneLLMSellerAgent
from .agents.human import HumanAgent
from .agents.extractor import PriceExtractor
from .scenario_manager import NegotiationScenario
from .dspy_manager import DSPyManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class NegotiationConfig:
    """単一の交渉の Configuration"""
    scenario: NegotiationScenario
    buyer_model: str
    seller_model: str
    buyer_strategy: str
    seller_strategy: str
    buyer_agent: str
    seller_agent: str
    max_turns: int
    turn_timeout: float

@dataclass
class NegotiationMetrics:
    """交渉中に収集された Metrics"""
    start_time: datetime
    end_time: Optional[datetime] = None
    turns_taken: int = 0
    buyer_hcv: bool = False
    seller_hcv: bool = False
    buyer_target_price: Optional[float] = None
    seller_target_price: Optional[float] = None
    buyer_max_price: Optional[float] = None
    seller_min_price: Optional[float] = None
    final_price: Optional[float] = None
    buyer_utility: Optional[float] = None
    seller_utility: Optional[float] = None
    fairness: Optional[float] = None
    messages: list[dict] = field(default_factory=list) # 2025/7/18 Noneからfieldに変更

    def compute_duration(self) -> float:
        """交渉時間を秒単位で計算する"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class NegotiationRunner:
    """
    AgreeMate システムの個々の交渉セッションの実行を管理する
    agent の初期化, 交渉ターンの管理, 及び metrics の収集を処理する
    """

    def __init__(
        self,
        dspy_manager: DSPyManager,
        max_concurrent: int = 4
    ):
        """
        negotiation runner を初期化する

        Args:
            dspy_manager: DSPy LM マネージャーインスタンス
            max_concurrent: 同時交渉の最大数
        """
        self.dspy_manager = dspy_manager
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_negotiations: dict[str, NegotiationMetrics] = {}

    def _show_scenario(self,scenario):
            print("\n--------Let's Negotiate!--------\n" \
            f"Title: {scenario.title}\n" \
            f"Category: {scenario.category}\n" \
            f"List Price: {scenario.list_price}\n" \
            f"description: {scenario.description}\n")
     
    # 交渉のために買い手エージェントと売り手エージェントを初期化する
    def _initialize_agents(self,config: NegotiationConfig):
        if config.buyer_model == "human":
            seller_lm = self.dspy_manager.get_lm(
                config.seller_model,
                strategy_name=config.seller_strategy,
                agent_name=config.seller_agent,
                role='seller'
            )
            buyer = HumanAgent(
                strategy_name=config.buyer_strategy,
                target_price=config.scenario.buyer_target,
                list_price=config.scenario.list_price,
                category=config.scenario.category,
                is_buyer=True,
                item_info={
                    "item_name": (config.scenario).title,
                    "category": (config.scenario).category,
                    "list_price": (config.scenario).list_price,
                    "description": (config.scenario).description
                }
            )
            seller = SellerAgent(
                strategy_name=config.seller_strategy,
                target_price=config.scenario.seller_target,
                list_price=config.scenario.list_price,
                category=config.scenario.category,
                item_info={
                    "item_name": (config.scenario).title,
                    "category": (config.scenario).category,
                    "list_price": (config.scenario).list_price,
                    "description": (config.scenario).description
                },
                lm=seller_lm
            )
            #buyer.max_price = buyer.max_price_select()
            print("buyer's target_price: ", buyer.target_price) ############
            print("buyer's max_price: ", buyer.max_price) ############
            #seller.min_price = seller.min_price_select()
            print("seller's target_price: ", seller.target_price) ############
            print("seller's min_price: ", seller.min_price) #########

        elif config.seller_model == "human":
            buyer_lm = self.dspy_manager.get_lm(
                config.buyer_model,
                strategy_name=config.buyer_strategy,
                agent_name=config.buyer_agent,
                role='buyer'
            )
            buyer = BuyerAgent(
                strategy_name=config.buyer_strategy,
                target_price=config.scenario.buyer_target,
                list_price=config.scenario.list_price,
                category=config.scenario.category,
                item_info={
                    "item_name": (config.scenario).title,
                    "category": (config.scenario).category,
                    "list_price": (config.scenario).list_price,
                    "description": (config.scenario).description
                },
                lm=buyer_lm
            )
            seller = HumanAgent(
                strategy_name=config.seller_strategy,
                target_price=config.scenario.seller_target,
                list_price=config.scenario.list_price,
                category=config.scenario.category,
                is_buyer=False,
                item_info={
                    "item_name": (config.scenario).title,
                    "category": (config.scenario).category,
                    "list_price": (config.scenario).list_price,
                    "description": (config.scenario).description
                }
            )
            #print("buyer's target_price: ", buyer.target_price) ############
            #print("buyer's max_price: ", buyer.max_price) ############
            #print("seller's target_price: ", seller.target_price) ############
            #print("seller's min_price: ", seller.min_price) #########

        else:
            # 戦略固有の構成をもつ DSPy LMs を取得する
            buyer_lm, seller_lm = self.dspy_manager.configure_negotiation(
                config.buyer_model,
                config.seller_model,
                config.buyer_strategy,
                config.seller_strategy,
                config.buyer_agent,
                config.seller_agent
            )
            if config.seller_agent == "damf":
                seller = SellerAgent(
                    strategy_name=config.seller_strategy,
                    target_price=config.scenario.seller_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=seller_lm
                )
            elif config.seller_agent == "search":
                seller = SearchBaseSellerAgent(
                    strategy_name=config.seller_strategy,
                    target_price=config.scenario.seller_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=seller_lm
                )
            elif config.seller_agent == "simple":
                seller = SimpleLLMSellerAgent(
                    target_price=config.scenario.seller_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=seller_lm
                )
            elif config.seller_agent == "all":
                seller = AllinOneLLMSellerAgent(
                    target_price=config.scenario.seller_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=seller_lm
                )
            else:
                print("config.seller_agent:", config.seller_agent)
                raise ValueError("Invalid agent name.")
            
            if config.buyer_agent == "damf":
                buyer = BuyerAgent(
                    strategy_name=config.buyer_strategy,
                    target_price=config.scenario.buyer_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=buyer_lm
                )
            elif config.buyer_agent == "search":
                buyer = SearchBaseBuyerAgent(
                    strategy_name=config.buyer_strategy,
                    target_price=config.scenario.buyer_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=buyer_lm
                )
            elif config.buyer_agent == "simple":
                buyer = SimpleLLMBuyerAgent(
                    target_price=config.scenario.buyer_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=buyer_lm
                )
            elif config.buyer_agent == "all":
                buyer = AllinOneLLMBuyerAgent(
                    target_price=config.scenario.buyer_target,
                    list_price=config.scenario.list_price,
                    category=config.scenario.category,
                    item_info={
                        "item_name": (config.scenario).title,
                        "category": (config.scenario).category,
                        "list_price": (config.scenario).list_price,
                        "description": (config.scenario).description
                    },
                    lm=buyer_lm
                )
            else:
                print("config.seller_agent:", config.seller_agent)
                raise ValueError("Invalid agent name.")
            
            #print("buyer's target_price: ", buyer.target_price) ############
            #print("buyer's max_price: ", buyer.max_price) ###########
            #print("seller's target_price: ", seller.target_price) ############
            #print("seller's min_price: ", seller.min_price) #########
        return buyer, seller
    
    # 交渉のために価格抽出器を初期化する
    def _initialize_extractor(self):
        extractor_lm = self.dspy_manager.get_extractor_lm()
        extractor = PriceExtractor(
            lm=extractor_lm
        )
        return extractor


    def _validate_price_movement(
        self,
        agent_role: str,
        new_price: float,
        metrics: NegotiationMetrics
    ) -> bool:
        """価格の変動が交渉ルールに従っているかどうかを検証する"""
        if not metrics.messages:
            return True # 最初のオファーは常に有効

        last_price = next(
            (m['price'] for m in reversed(metrics.messages) 
            if m['price'] is not None),
            None
        )

        if last_price is None:
            return True

        # 価格は交渉内でお互いに譲歩していく必要があるので, 買い手はより高い価格を提示すべきであり, 売り手はより低い価格を提示すべきである
        if agent_role == 'buyer':
            return new_price >= last_price
        else:
            return new_price <= last_price

    async def _run_negotiation_turn(
        self,
        buyer, # BuyerAgent
        seller, # SellerAgent
        extractor: PriceExtractor,
        metrics: NegotiationMetrics,
        timeout: float
    ) -> bool:
        """
        交渉を1ターン実行する

        Returns:
            bool: 交渉を継続する場合は True
        """
        try:
            # buyerとsellerを交互に行う(buyerから交渉開始)
            current_agent = buyer if metrics.turns_taken % 2 == 0 else seller

            # タイムアウトでターンを実行する
            async with asyncio.timeout(timeout):
                partner_data = metrics.messages[-1] if metrics.messages else None
                response = current_agent.step(partner_data, extractor) # 2025/7/18 await current_agent.step()のawait削除

                # message の構造を検証する
                if not isinstance(response, dict) or 'role' not in response:
                    raise ValueError("Invalid message format")

                # 必須となる fields を確認する
                response.setdefault('price', None)
                response.setdefault('intent', 'counter')

            metrics.turns_taken += 1
            metrics.messages.append(response)

            # handle completion
            if response['intent'] in ['accept', 'reject']:
                metrics.end_time = datetime.now()
                metrics.final_price = (
                    response['price'] if response['intent'] == 'accept' 
                    else None
                )
                print("final_price: ", metrics.final_price)
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Turn timeout after {timeout}s")
            metrics.end_time = datetime.now()
            return False
        except Exception as e:
            logger.error(f"Turn error: {str(e)}")
            metrics.end_time = datetime.now()
            return False

    async def run_negotiation(
        self,
        config: NegotiationConfig
    ) -> NegotiationMetrics:
        """
        完全な交渉セッションを実行する

        Args:
            config: Negotiation configuration

        Returns:
            Completed negotiation metrics
        """
        async with self.semaphore:
            # metrics のトラッキングを初期化する
            metrics = NegotiationMetrics(
                start_time=datetime.now(),
            )

            try:
                # エージェントを初期化
                buyer, seller = self._initialize_agents(config)
                metrics.buyer_target_price = buyer.target_price
                metrics.seller_target_price = seller.target_price
                metrics.buyer_max_price = buyer.max_price
                metrics.seller_min_price = seller.min_price
                # 価格抽出器を初期化 2025/9/17追加
                price_extractor = self._initialize_extractor()
                
                # active な交渉をトラッキングする
                self.active_negotiations[config.scenario.scenario_id] = metrics

                # scenarioの内容を表示する
                self._show_scenario(config.scenario)

                # 交渉ターンを実行する
                continue_negotiation = True
                
                while (continue_negotiation and metrics.turns_taken < config.max_turns):
                    continue_negotiation = await self._run_negotiation_turn(buyer, seller, price_extractor, metrics, config.turn_timeout)

                # ▼▼▼▼▼ 2025/7/18 このブロックをここに追加 ▼▼▼▼▼
                # 最大ターン数に達して交渉が終了した場合の処理
                if metrics.end_time is None:
                    logger.info(f"Negotiation reached max turns ({config.max_turns}) without an agreement.")
                    metrics.end_time = datetime.now()
                # ▲▲▲▲▲ ここまで ▲▲▲▲▲

                # 最終的な metrics を計算する
                self._compute_final_metrics(metrics, buyer, seller)

                return metrics

            except Exception as e:
                logger.error(f"Negotiation failed: {str(e)}")
                metrics.end_time = datetime.now()
                return metrics
            finally: # active な交渉から除外する
                self.active_negotiations.pop(config.scenario.scenario_id, None)
    
    def _compute_fairness(
            self, 
            final_price: float,
            buyer_target_price: float,
            seller_target_price: float
    ):
        median_diff = final_price - ((seller_target_price + buyer_target_price) / 2.0)
        abs_median_diff = abs(median_diff)
        target_diff = seller_target_price - buyer_target_price
        fairness = 1.0 - (2.0 * abs_median_diff / target_diff)

        if fairness >= 1.0:
            fairness = 1.0
        elif fairness <= 0.0:
            fairness = 0.0

        return fairness


    def _compute_final_metrics(
        self,
        metrics: NegotiationMetrics,
        buyer,
        seller
    ):
        """最終的な交渉 metrics の計算"""
        if metrics.final_price:
            # utilities の計算
            metrics.buyer_utility = buyer.compute_utility(metrics.final_price)
            metrics.seller_utility = seller.compute_utility(metrics.final_price)
            metrics.fairness = self._compute_fairness(metrics.final_price, buyer.target_price, seller.target_price)
            if metrics.final_price > buyer.max_price:
                metrics.buyer_hcv = True
            elif metrics.final_price < seller.min_price:
                metrics.seller_hcv = True

    async def run_batch(
        self,
        configs: list[NegotiationConfig]
    ) -> dict[str, NegotiationMetrics]:
        """バッチの交渉を並列して実行する"""
        tasks = []
        for config in configs:
            task = asyncio.create_task(self.run_negotiation(config))
            tasks.append((config.scenario.scenario_id, task))

        results = {}
        for scenario_id, task in tasks:
            try:
                metrics = await task
                results[scenario_id] = metrics
            except Exception as e:
                logger.error(f"Batch task failed: {str(e)}")

        return results


def test_negotiation_runner():
    """negotiation runner 機能をテストする"""
    from .scenario_manager import ScenarioManager
    from .utils.data_loader import DataLoader
    import logging
    logging.getLogger('dspy.adapters.json_adapter').setLevel(logging.ERROR) # JSONになるWARNINGを消す処理


    # components の初期化
    data_loader = DataLoader()
    scenario_manager = ScenarioManager(data_loader)
    dspy_manager = DSPyManager()
    runner = NegotiationRunner(dspy_manager)

    # テストシナリオの作成
    scenarios = scenario_manager.create_evaluation_batch(
        split='test',
        size=1
    )

    # テスト用の configuration の作成
    config = NegotiationConfig(
        scenario=scenarios[0],
        buyer_model="llama3.1",
        seller_model="llama3.1",
        buyer_strategy="length",
        seller_strategy="fair",
        buyer_agent="damf",
        seller_agent="damf",
        max_turns=5,
        turn_timeout=30.0
    )
    # テスト交渉を実行
    async def run_test():
        metrics = await runner.run_negotiation(config)
        assert metrics.turns_taken > 0
        assert metrics.end_time is not None
        print("✓ Negotiation completed successfully")
        return metrics
    
    import asyncio
    metrics = asyncio.run(run_test())

    print("✓ All negotiation runner tests passed")
    return runner, metrics

if __name__ == "__main__":
    runner, metrics = test_negotiation_runner()