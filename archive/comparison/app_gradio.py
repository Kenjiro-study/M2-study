import gradio as gr
import random
import datetime
import json
import time
import uuid
import os
import dspy
import torch
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import pandas as pd # ScenarioManager で使う

# -----------------------------------------------------------------
# 1. 必要なクラスのインポート
# -----------------------------------------------------------------
# (パスが通っていることを確認してください)
try:
    from .agents.human import HumanAgent
    from .agents.buyer import BuyerAgent
    from .agents.seller import SellerAgent
    from .agents.extractor import PriceExtractor
    from .agents.search_buyer import SearchBaseBuyerAgent
    from .agents.search_seller import SearchBaseSellerAgent
    from .agents.simple_llm_buyer import SimpleLLMBuyerAgent
    from .agents.simple_llm_seller import SimpleLLMSellerAgent
    from .agents.all_one_buyer import AllinOneLLMBuyerAgent
    from .agents.all_one_seller import AllinOneLLMSellerAgent
    from .agents.base_agent import NegotiationPhase
    from .dspy_manager import DSPyManager, DSPyLMConfig
    from .config import MODEL_CONFIGS, ModelConfig
    from .strategies import STRATEGIES, CATEGORY_CONTEXT, BUYER_INTENT_CONTEXT, SELLER_INTENT_CONTEXT, BUYER_LANGUAGE_SKILLS, SELLER_LANGUAGE_SKILLS
    from .scenario_manager import ScenarioManager, NegotiationScenario
    from .utils.data_loader import DataLoader

except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なファイル (human.py, ... scenario_manager.py, data_loader.py) が正しく配置されているか確認してください。")
    raise

# -----------------------------------------------------------------
# 1. NegotiationConfig の定義
# -----------------------------------------------------------------
@dataclass
class NegotiationConfig:
    scenario: NegotiationScenario
    max_turns: int = 20
# -----------------------------------------------------------------
# 2. Gradio対応 HumanAgent (GradioHumanAgent)
# -----------------------------------------------------------------
class GradioHumanAgent(HumanAgent):
    _parser = None
    _tokenizer = None
    _device = None

    @classmethod
    def _get_parser_components(cls):
        if cls._parser is None or cls._tokenizer is None:
            print("--- Loading HumanAgent Parser (初回のみ) ---")
            cls._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = "archive/comparison/agents/parser/model/roberta_fold_1/checkpoint-82304"
            cls._parser = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
            cls._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            cls._parser = cls._parser.to(cls._device)
            print("--- HumanAgent Parser Loaded ---")
        return cls._parser, cls._tokenizer, cls._device

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser, self.tokenizer, self.device = self._get_parser_components()

    def step(self, partner_data, extractor):
        raise NotImplementedError("HumanAgent.step() はGradio環境では呼び出せません。")

# -----------------------------------------------------------------
# 3. グローバルインスタンスの初期化
# -----------------------------------------------------------------
print("--- DSPyManager, ScenarioManager, Extractor を初期化します ---")

# (1) DSPyManager
dspy_manager = DSPyManager(cache_dir="./.dspy_cache")

# (2) Extractor
try:
    extractor_lm = dspy_manager.get_extractor_lm()
    price_extractor_instance = PriceExtractor(lm=extractor_lm)
    print("--- PriceExtractor 初期化完了 ---")
except Exception as e:
    print(f"PriceExtractorの初期化に失敗: {e}")
    price_extractor_instance = None

# (3) ScenarioManager
try:
    data_loader = DataLoader()
    scenario_manager = ScenarioManager(data_loader=data_loader)
    print(f"--- Real ScenarioManager 初期化完了 ---")
    print(f"--- (Loaded {len(scenario_manager.test_df)} test scenarios) ---")
except Exception as e:
    print(f"ScenarioManagerの初期化に失敗: {e}")
    print("DataLoaderのパス (data_dir=) が正しいか確認してください。")
    # (致命的なエラーなので、ここで停止するのが望ましい)
    raise

# (4) 8パターンのエージェント組み合わせ (本番用)
ALL_AGENT_COMBINATIONS = [
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "damf", "role": "seller", "strategy": "fair"},
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "search", "role": "seller", "strategy": "fair"},
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "simple", "role": "seller", "strategy": "free"},
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "all", "role": "seller", "strategy": "free"},
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "damf", "role": "buyer", "strategy": "fair"},
    #{"model_key": "openai/gpt-oss-20b", "agent_name": "search", "role": "buyer", "strategy": "fair"},
    {"model_key": "openai/gpt-oss-20b", "agent_name": "simple", "role": "buyer", "strategy": "free"},
    {"model_key": "openai/gpt-oss-20b", "agent_name": "all", "role": "buyer", "strategy": "free"},
]
# 実験回数はこのリストの長さになる
NUM_SESSIONS = len(ALL_AGENT_COMBINATIONS)

# -----------------------------------------------------------------
# 4. イベントハンドラ関数群 (ロジック)
# -----------------------------------------------------------------

def initialize_session(session_task: tuple):
    """
    指定された組み合わせで HumanAgent と AI Agent を初期化するヘルパー関数
    Args:
        session_task: (agent_combo_dict, scenario_obj) のタプル
    """
    ai_agent_combo, scenario_obj = session_task
    
    # (1) 役割を決定
    human_is_buyer = True if ai_agent_combo["role"] == "seller" else False
    human_role = "buyer" if human_is_buyer else "seller"
    
    # (2) ★ シナリオオブジェクトから item_info 辞書を作成
    item_info_dict = {
        "item_name": scenario_obj.title,
        "category": scenario_obj.category,
        "list_price": scenario_obj.list_price,
        "description": scenario_obj.description
    }
    
    # (3) HumanAgent を初期化
    human_agent = GradioHumanAgent(
        strategy_name=ai_agent_combo["strategy"],
        target_price=scenario_obj.buyer_target if human_is_buyer else scenario_obj.seller_target,
        list_price=scenario_obj.list_price,
        category=scenario_obj.category,
        is_buyer=human_is_buyer,
        item_info=item_info_dict
    )
    
    # (4) ★ AI Agent のための「戦略的LM」を DSPyManager から取得 ★
    print(f"--- DSPyManager: '{ai_agent_combo['agent_name']}' (Strategy: {ai_agent_combo['strategy']}) 用のLMを取得 ---")
    ai_lm = dspy_manager.get_lm(
        model_key=ai_agent_combo["model_key"],
        strategy_name=ai_agent_combo["strategy"],
        agent_name=ai_agent_combo["agent_name"],
        role=ai_agent_combo["role"]
    )
    
    # (5) AI Agent (Buyer/Seller) を初期化
    if human_is_buyer: # AIはSeller
        if ai_agent_combo["agent_name"] == "damf":
            ai_agent = SellerAgent(
                strategy_name=ai_agent_combo["strategy"],
                target_price=scenario_obj.seller_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "search":
            ai_agent = SearchBaseSellerAgent(
                strategy_name=ai_agent_combo["strategy"],
                target_price=scenario_obj.seller_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "simple":
            ai_agent = SimpleLLMSellerAgent(
                target_price=scenario_obj.seller_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "all":
            ai_agent = AllinOneLLMSellerAgent(
                target_price=scenario_obj.seller_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        else:
            raise ValueError(f"{ai_agent_combo['agent_name']} is invalid agent name.")
    else: # AIはBuyer
        if ai_agent_combo["agent_name"] == "damf":
            ai_agent = BuyerAgent(
                strategy_name=ai_agent_combo["strategy"],
                target_price=scenario_obj.buyer_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "search":
            ai_agent = SearchBaseBuyerAgent(
                strategy_name=ai_agent_combo["strategy"],
                target_price=scenario_obj.buyer_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "simple":
            ai_agent = SimpleLLMBuyerAgent(
                target_price=scenario_obj.buyer_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        elif ai_agent_combo["agent_name"] == "all":
            ai_agent = AllinOneLLMBuyerAgent(
                target_price=scenario_obj.buyer_target,
                list_price=scenario_obj.list_price,
                category=scenario_obj.category,
                item_info=item_info_dict,
                lm=ai_lm
            )
        else:
            raise ValueError(f"{ai_agent_combo['agent_name']} is invalid agent name.")
        
    # (6) NegotiationConfig を初期化 (ログ記録用)
    config = NegotiationConfig(
        scenario=scenario_obj, # ★ 本物のシナリオオブジェクト
        max_turns=20
    )

    # (7) UI表示用のテキストを準備
    if human_is_buyer:
        scenario_text = f"""
            あなたは **{human_role}** です。この商品の初期販売価格は **${scenario_obj.list_price}** です。
            売り手と交渉を行い, できるだけ目標価格に近い価格で買うことを目指してください。

            ### あなたの交渉情報
            * **役割:** {human_role}
            * **目標価格:** ${human_agent.target_price}


            **商品の詳細:**
            * **商品:** {scenario_obj.title}
            * **カテゴリー:** {scenario_obj.category}
            * **初期販売価格:** ${scenario_obj.list_price}
            * **商品説明:** {scenario_obj.description}

        """
    
    else:
        scenario_text = f"""
            あなたは **{human_role}** です。この商品の初期販売価格は **${scenario_obj.list_price}** です。
            売り手と交渉を行い, できるだけ目標価格(初期販売価格)に近い価格で売ることを目指してください。

            **あなたの交渉情報:**
            * **役割:** {human_role}
            * **目標価格:** ${human_agent.target_price}

            **商品の詳細:**
            * **商品:** {scenario_obj.title}
            * **カテゴリー:** {scenario_obj.category}
            * **初期販売価格:** ${scenario_obj.list_price}
            * **商品説明:** {scenario_obj.description}

        """
    operate_text = f"""
        * 交渉は画面右側の交渉チャットボックスで行います。シナリオに記載されている役割(buyerかseller)になりきって, パートナーと交渉を行ってください。
        * 対話ログ内にパートナーからのメッセージが届いたら, 「メッセージ」と書かれたチャットボックスに文を入力し, Entetキー, または「送信」ボタンを押してメッセージを送信してください。
        * 交渉を進めていき, 自分が納得できる価格が提案された場合は **交渉合意ボタン** , これ以上交渉を続けても仕方がないと感じた場合は **交渉非合意ボタン** を押して交渉を終了してください。

    """

    caution_text = f"""
        * あなたが **buyer** の場合は, あなたからメッセージを送って交渉を開始してください。
        * 交渉は **交互に** 行います。自分が一度メッセージを送った後は相手からメッセージが送られてくるまでボタンを押したりやメッセージを送信したりしないでください。
        * タイプミスや文法間違いには十分気をつけて, **<span style="color: red; ">英語</span>** でチャットを行ってください
        * **価格交渉以外の会話も遠慮なくしてください**。例えば商品の状態を聞いたり, 送料無料を提案したり, 追加特典をつけたりすることです。またsellerの場合で, 商品説明の欄の情報が少ない場合は自由に設定を考えて商品の状態や付属品等について説明して構いません。ただし, 商品説明の内容と矛盾しないよう心がけてください。
        * 実験中に問題が発生した場合はslackの **<span style="color: red; ">times-morimoto</span>** までご連絡ください

    """
    
    # (8) 最初のAIの発話
    initial_chat_history = []
    if not human_is_buyer: # 被験者がSellerの場合、AI (Buyer) が先に話す
        print("AI (Buyer) が最初の発話を行います。")
        ai_response_dict = ai_agent.step(partner_data=None, extractor=price_extractor_instance)
        # AIの状態更新 (base_agent.py の update_state() を内部で呼んでいるはず)
        # 人間の状態更新 (相手の情報をセット)
        human_agent.partner_data = ai_response_dict
        human_agent.conversation_history.append(ai_response_dict) # ★ 人間の履歴にもAIの初回発話を追加
        human_agent.pertner_intent_history.append(ai_response_dict['intent'])
        if ai_response_dict['price'] is not None:
            human_agent.partner_price_history.append(ai_response_dict['price'])
            human_agent.all_price_history.append(ai_response_dict['price'])
            
        initial_chat_history = [{"role": "assistant", "content": ai_response_dict["content"]}]

    return human_agent, ai_agent, config, scenario_text, operate_text, caution_text, initial_chat_history

def handle_start_experiment(request: gr.Request):
    """
    [流れ1] 実験開始ボタンが押された時の処理
    """
    print("--- イベント: handle_start_experiment ---")
    if request and request.username:
        user_name = request.username
    else:
        user_name = "guest" # 認証なし、またはローカルテストの場合
    
    # ファイル名が被らないように「ユーザー名_日時」をIDとする
    # 例: user1_20251118-103000
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"{user_name}_{timestamp}"

    print(f"セッションID (被験者ID) を発行: {session_id}")
    
    # ★★★ シナリオとエージェントをzipする ★★★
    try:
        # (1) 8種類のエージェント定義をシャッフル
        shuffled_agents = random.sample(ALL_AGENT_COMBINATIONS, NUM_SESSIONS)
        
        # (2) 8個のシナリオをScenarioManagerから取得 (testスプリットからバランス良く)
        scenario_batch = scenario_manager.create_evaluation_batch(
            split='test',
            size=NUM_SESSIONS,
            balanced_categories=True
        )
        scenario_batch[1] = scenario_batch[0] ########
        print("scenario_batch: ", scenario_batch) ########

        # (3) (エージェント, シナリオ) のタプルリストを作成
        session_tasks = list(zip(shuffled_agents, scenario_batch))
        
        if not session_tasks:
            raise ValueError("エージェントまたはシナリオのリストが空です。")

    except Exception as e:
        print(f"セッションの準備に失敗: {e}")
        # (エラーUI)
        return (session_id, None, [], None, None, None, None, 
                gr.update(), gr.update(visible=False), 
                f"エラー: {e}", "エラー", [])

    # 最初のタスクを取り出し、残りをキューに入れる
    first_task = session_tasks[0]
    remaining_queue = session_tasks[1:] # ★ タプル (agent, scenario) のリスト
    
    try:
        human_agent, ai_agent, config, scenario_text, operate_text, caution_text, initial_chat_history = initialize_session(
            session_task=first_task
        )
    except Exception as e:
        print(f"エージェント初期化中にエラーが発生しました: {e}")
        # (エラーUI)
        return (session_id, None, [], None, None, None, None, 
                gr.update(), gr.update(visible=False), 
                f"エラー: {e}", f"エラー: {e}", [])

    # 戻り値で、UIの表示/非表示と、Stateの更新を行う
    return (
        session_id,                # [Output] state_session_id
        remaining_queue,           # [Output] state_task_queue (★ `agent_queue` -> `task_queue`)
        [],                        # [Output] state_all_results (リセット)
        human_agent,               # [Output] state_human_agent
        ai_agent,                  # [Output] state_ai_agent
        config,                    # [Output] state_config
        price_extractor_instance,  # [Output] state_extractor
        gr.update(visible=False),  # [Output] start_group
        gr.update(visible=True),   # [Output] negotiation_group
        scenario_text,             # [Output] scenario_display
        operate_text,              # [Output] operate_display
        caution_text,              # [Output] caution_display
        initial_chat_history       # [Output] chatbot_display
    )

def handle_chat_message(
    human_message_text,       # [Input] chat_input
    chat_history,             # [Input] chatbot_display
    human_agent, ai_agent,    # [Input]
    config, extractor
):
    """
    [流れ2] 被験者がメッセージを送信した時の処理
    """
    print("--- イベント: handle_chat_message ---")
    
    # gradioの状態管理で稀にNoneとなってしまった場合のリカバリ処理
    if extractor is None:
        print("Warning: extractor arg is None. Using global price_extractor_instance.")
        extractor = price_extractor_instance
    
    # -------------------------------------------------
    # 1. 被験者 (Human) のターン (状態更新)
    # -------------------------------------------------
    try:
        # human.py の update_state() を呼び出す
        human_message_dict = human_agent.update_state(
            message={"role": human_agent.role, "content": human_message_text},
            extractor=extractor
        )
        print(f"Human ({human_agent.role}) state updated. Intent: {human_message_dict['intent']}, Price: {human_message_dict['price']}")
        
        chat_history.append({"role": "user", "content": human_message_text})
        #ai_agent.partner_data = human_message_dict
        
    except Exception as e:
        print(f"HumanAgent.update_state() でエラー: {e}")
        human_message_dict = {"role": human_agent.role, "content": human_message_text, "intent": "unknown", "price": None}
        chat_history.append({"role": "user", "content": human_message_text})
        chat_history.append({"role": "assistant", "content": f"(システムエラー: {e})"})
        ai_agent.partner_data = human_message_dict
    
    # AIの応答を待つ間、ボタンを無効化
    yield (
        chat_history, "", human_agent, ai_agent, 
        gr.update(), gr.update(), 
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
        None
    )

    # -------------------------------------------------
    # 2. エージェント (dspy) のターン (step実行)
    # -------------------------------------------------
    try:
        ai_response_dict = ai_agent.step(
            partner_data=human_message_dict,
            extractor=extractor
        )
        print(f"AI ({ai_agent.role}) step executed. Intent: {ai_response_dict['intent']}, Price: {ai_response_dict['price']}")
        
    except Exception as e:
        print(f"AI Agent.step() でエラー: {e}")
        ai_response_dict = {"role": ai_agent.role, "content": f"エラーが発生しました: {e}", "price": None, "intent": "reject"}
    
    ai_intent_str = copy.copy(ai_response_dict['intent'])
    chat_history.append({"role": "assistant", "content": ai_response_dict["content"]})

    # -------------------------------------------------
    # 3. ★★★ AIの応答を HumanAgent が解析するターン ★★★
    # (Human.py の step() の `input()` より前の部分のロジックをここに移植)
    # -------------------------------------------------
    try:
        print(f"Human ({human_agent.role}) が AI の応答を解析中...")
        # (1) パートナー(AI)のデータを HumanAgent にセット
        human_agent.partner_data = ai_response_dict

        # (2) AIの応答を `parse_dialogue` で解析
        ai_intent = human_agent.parse_dialogue(ai_response_dict['content'])
        ai_price = None
        human_agent.partner_data['intent'] = ai_intent # intent を上書き
        
        # (3) AIの価格を `extractor` で抽出・インテント修正 (Human.py の step() と同じロジック)
        if ai_intent in ["init-price", "counter-price", "insist"]:
            with dspy.context(lm=extractor.lm):
                price_prediction = extractor.compiled_extractor(
                    message_content=ai_response_dict['content']
                )
            ai_price = price_prediction["extracted_price"]
            
            # Human.py と同じインテント修正ロジック
            if ai_price == None:
                human_agent.partner_data['intent'] = "unknown"
            elif (not human_agent.price_history) and (not human_agent.partner_price_history) and (ai_intent in ["counter-price", "insist"]):
                human_agent.partner_data['intent'] = "init-price"
            elif (human_agent.partner_price_history) and (human_agent.partner_price_history[-1] == ai_price) and (ai_intent in ["init-price", "counter-price"]):
                human_agent.partner_data['intent'] = "insist"
            elif (human_agent.partner_price_history or human_agent.price_history) and (ai_intent == "init-price"):
                human_agent.partner_data['intent'] = "counter-price"

        human_agent.partner_data['price'] = ai_price
        
        # (4) HumanAgent の履歴を更新
        human_agent.conversation_history.append(human_agent.partner_data)
        human_agent.pertner_intent_history.append(human_agent.partner_data['intent'])
        if ai_price is not None:
            human_agent.partner_price_history.append(ai_price)
            human_agent.all_price_history.append(ai_price)
            
        print(f"Human が解析した AI の意図: {human_agent.partner_data['intent']}(price={ai_price})")

    except Exception as e:
        print(f"HumanAgent が AI の応答解析中にエラー: {e}")
        # (エラーが発生しても、最低限の履歴は残す)
        if human_agent.partner_data: # ai_response_dict がセットされているはず
             human_agent.conversation_history.append(human_agent.partner_data)
    
    # -------------------------------------------------
    # 4. 交渉終了判定 (仕様変更対応)
    # -------------------------------------------------
    print("ai_intent_str: ", ai_intent_str)
    print("ai_intent_str == 'accept': ", ai_intent_str == 'accept')
    
    # 【パターンA】 AIが「合意(accept)」した場合
    if ai_intent_str == 'accept':
        price_info = f" ${ai_agent.all_price_history[-1]}" if ai_agent.all_price_history else ""
        system_msg = f"**相手が{price_info} で交渉を合意しました。**\n**これ以上メッセージは送らず、下の「交渉合意」ボタンを押してください。**"
        chat_history.append({"role": "assistant", "content": system_msg})
        
        # 画面を更新して、ボタンを有効化 (ユーザーの入力を待つ)
        yield (
            chat_history, "", human_agent, ai_agent, 
            gr.update(), gr.update(), 
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
            None
        )
        return # ここで処理終了

    # 【パターンB】 AIが「拒否(reject)」した場合
    elif ai_intent_str == 'reject':
        system_msg = "**相手が交渉終了を選択しました。**\n**これ以上メッセージは送らず、下の「交渉非合意」ボタンを押して交渉を終了してください。**"
        chat_history.append({"role": "assistant", "content": system_msg})
        
        # 画面を更新して、ボタンを有効化 (ユーザーの入力を待つ)
        yield (
            chat_history, "", human_agent, ai_agent, 
            gr.update(), gr.update(), 
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
            None
        )
        return # ここで処理終了
    
    # 【パターンC】 最大ターン数に達した場合
    if human_agent.num_turns >= config.max_turns:
        print("最大ターン数に達しました。")
        system_msg = "**最大ターン数に達しました。交渉を終了します。**\n(5秒後に評価画面へ移動します...)"
        chat_history.append({"role": "assistant", "content": system_msg})
        
        # まずメッセージを表示 (ボタンは無効のまま)
        yield (
            chat_history, "", human_agent, ai_agent, 
            gr.update(), gr.update(), 
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
            None
        )
        
        # 5秒待機
        time.sleep(5)
        
        # 自動で評価画面へ遷移
        yield (
            chat_history, "", human_agent, ai_agent,
            gr.update(visible=False), gr.update(visible=True), # negotiation -> evaluation
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
            chat_history
        )
        return # ここで処理終了
    
    # 【パターンD】 交渉継続
    # チャット履歴を更新して、ボタンを再度有効化
    yield (
        chat_history, "", human_agent, ai_agent,
        gr.update(), gr.update(), 
        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
        None
    )

def handle_finish_negotiation(
    decision_type,        # [Input] "accept" or "reject"
    chat_history,         # [Input] chatbot_display
    human_agent, ai_agent, # [Input] states
    config, extractor
):
    """
    [流れ2亜種] 被験者が「合意」「非合意」ボタンを押した時の処理
    """
    print(f"--- イベント: handle_finish_negotiation (type: {decision_type}) ---")

    # ★★★ 追加: extractor が None の場合のリカバリ処理 ★★★
    if extractor is None:
        print("Warning: extractor arg is None. Using global price_extractor_instance.")
        extractor = price_extractor_instance

    # 1. 人間の最終意思決定を HumanAgent の状態に反映
    human_message_text = "【交渉合意】" if decision_type == "accept" else "【交渉非合意】"
    human_message_dict = human_agent.update_state(
        # "accept" or "reject" は human.py の update_state で特別扱いされる
        message={"role": human_agent.role, "content": decision_type}, 
        extractor=extractor
    )
    
    ai_agent.partner_data = human_message_dict
    chat_history.append({"role": "user", "content": human_message_text})

    return (
        chat_history,
        human_agent, ai_agent, 
        gr.update(visible=False), # negotiation_group
        gr.update(visible=True),  # evaluation_group
        chat_history
    )


def handle_submit_evaluation(
    hl1_score, hl2_score, hl3_score, hl4_score, hl5_score,  # [Input] human_likeness_slider
    human_agent, ai_agent, # [Input]
    all_results,           # [Input] state_all_results
    task_queue,            # [Input] state_task_queue (★ `agent_queue` -> `task_queue`)
    config,                # [Input] state_config
    session_id             # [Input] state_session_id
):
    """
    [流れ3] 評価送信ボタンが押された時の処理
    """
    print("--- イベント: handle_submit_evaluation ---")
    results_dir = "archive/comparison/results"
    # -------------------------------------------------
    # 1. Agent内部状態からMetricsを事後生成して保存する
    # -------------------------------------------------
    try:
        final_history = human_agent.conversation_history
        final_price = None
        if final_history and final_history[-1]["intent"] == "accept":
            final_price = final_history[-1]["price"]
        
        buyer = human_agent if human_agent.is_buyer else ai_agent
        seller = ai_agent if human_agent.is_buyer else human_agent
        
        buyer_utility = buyer.compute_utility(final_price) if final_price else None
        seller_utility = seller.compute_utility(final_price) if final_price else None
        
        # Fairness計算
        if final_price:
            median_diff = final_price - ((seller.target_price + buyer.target_price) / 2.0)
            abs_median_diff = abs(median_diff)
            target_diff = seller.target_price - buyer.target_price
            fairness = 1.0 - (2.0 * abs_median_diff / target_diff)

            if fairness >= 1.0:
                fairness = 1.0
            elif fairness <= 0.0:
                fairness = 0.0
        else:
            fairness = None

        metrics_dict = {
            "start_time": None, # (別途記録必要)
            "end_time": datetime.datetime.now().isoformat(),
            "messages": final_history,
            "final_price": final_price,
            "buyer_utility": buyer_utility,
            "seller_utility": seller_utility,
            "fairness": fairness,
            "turns_taken": human_agent.num_turns,
        }

    except Exception as e:
        print(f"Metricsの事後生成でエラー: {e}")
        metrics_dict = f"METRICS_ERROR: {e}"

    evaluation_scores = {
        "human_likeness_1": hl1_score,
        "human_likeness_2": hl2_score,
        "human_likeness_3": hl3_score,
        "human_likeness_4": hl4_score,
        "human_likeness_5": hl5_score,
    }
    # -------------------------------------------------
    # 2. 今回の結果を保存する
    # -------------------------------------------------
    if human_agent.is_buyer:
        current_result = {
            "session_id": session_id,
            "config": {
                "ai_agent_name": ai_agent.__class__.__name__ if ai_agent else "N/A",
                "ai_agent_role": ai_agent.role if ai_agent else "N/A",
                "ai_agent_target": ai_agent.target_price if ai_agent else "N/A",
                "ai_agent_min_price": ai_agent.min_price if ai_agent else "N/A",
                "human_role": human_agent.role if human_agent else "N/A",
                "human_target": human_agent.target_price if human_agent else "N/A",
                "scenario_id": config.scenario.scenario_id if config else "N/A",
            },
            "metrics": metrics_dict,
            "evaluation": evaluation_scores,
        }
    else:
        current_result = {
            "session_id": session_id,
            "config": {
                "ai_agent_name": ai_agent.__class__.__name__ if ai_agent else "N/A",
                "ai_agent_role": ai_agent.role if ai_agent else "N/A",
                "ai_agent_target": ai_agent.target_price if ai_agent else "N/A",
                "ai_agent_max_price": ai_agent.max_price if ai_agent else "N/A",
                "human_role": human_agent.role if human_agent else "N/A",
                "human_target": human_agent.target_price if human_agent else "N/A",
                "scenario_id": config.scenario.scenario_id if config else "N/A",
            },
            "metrics": metrics_dict,
            "evaluation": evaluation_scores,
        }
    all_results.append(current_result)
    
    log_filename = os.path.join(results_dir, f"app_gradio_results_{session_id}.jsonl")
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(current_result, ensure_ascii=False) + "\n")
        print(f"結果を {log_filename} に追記しました。")
    except Exception as e:
        print(f"ログ保存エラー: {e}")

    # -------------------------------------------------
    # 3. 次の実験 or 終了 を判断する
    # -------------------------------------------------
    if not task_queue: # ★ `task_queue` を見る
        # --- 9. 全てのパターンが終了 ---
        print("全セッション終了。")
        final_filename = os.path.join(results_dir, f"app_gradio_results_FINAL_{session_id}.json")
        try:
            with open(final_filename, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"最終結果ファイル {final_filename} を保存しました。")
        except Exception as e:
            print(f"最終ログ保存エラー: {e}")
            
        return (
            all_results, task_queue, 
            None, None, None, # 3 states
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), # 3 groups
            None, None, None, None, # 4 text displays (scenario, operate, caution, chatbot)
            None # ★追加: evaluation_chatbot_display
        )
    
    else:
        # --- 8. 次の交渉セッションを開始 ---
        print("次のセッションへ...")
        next_task = task_queue[0]
        remaining_queue = task_queue[1:]

        try:
            human_agent, ai_agent, config, scenario_text, operate_text, caution_text, initial_chat_history = initialize_session(
                session_task=next_task
            )
        except Exception as e:
            print(f"次のセッションの初期化エラー: {e}")
            human_agent, ai_agent, config = None, None, None
            scenario_text, operate_text, caution_text, initial_chat_history = f"エラー: {e}", f"エラー: {e}", []

        return (
            all_results,             # [Output] state_all_results (更新)
            remaining_queue,         # [Output] state_task_queue (更新)
            human_agent,             # [Output] state_human_agent
            ai_agent,                # [Output] state_ai_agent
            config,                  # [Output] state_config
            gr.update(visible=False),  # [Output] evaluation_group
            gr.update(visible=True),   # [Output] negotiation_group
            gr.update(visible=False),  # [Output] end_group
            scenario_text,           # [Output] scenario_display
            operate_text,            # [Output] operate_display
            caution_text,            # [Output] caution_display
            initial_chat_history,     # [Output] chatbot_display (リセット)
            []                       # ★追加: evaluation_chatbot_display (リセット)
        )

# -----------------------------------------------------------------
# 5. UIの定義 (gr.Blocks)
# -----------------------------------------------------------------

theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Noto Sans JP"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"]
)

with gr.Blocks(theme=theme) as demo:
    # -----------------------------------------------------------------
    # 状態管理 (UIに見えないデータ)
    # -----------------------------------------------------------------
    state_session_id = gr.State(None)
    state_task_queue = gr.State([])
    state_all_results = gr.State([])
    state_human_agent = gr.State(None)
    state_ai_agent = gr.State(None)
    state_config = gr.State(None)
    state_extractor = gr.State(None)

    # -----------------------------------------------------------------
    # UIの画面定義
    # -----------------------------------------------------------------
    # --- ① スタート画面 ---
    with gr.Group(visible=True) as start_group:
        gr.Markdown("# 交渉実験へようこそ")
        gr.Markdown(
            f"""
            本実験にご協力いただきありがとうございます。
            この実験では、AIエージェントと価格交渉を行っていただきます。
            
            全部で **{NUM_SESSIONS}** 回、異なるエージェントまたは異なる役割（買い手・売り手）で交渉を行います。
            
            準備ができたら下のボタンを押して開始してください。
            """
        )
        start_button = gr.Button("実験開始", variant="primary", scale=1)

    # --- ② 交渉画面 ---
    with gr.Group(visible=False) as negotiation_group:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 交渉シナリオ")
                scenario_display = gr.Markdown("ここにシナリオが表示されます")
                with gr.Accordion("交渉の進め方（クリックして開く）", open=False):
                    operate_display = gr.Markdown("ここに注意書きが表示されます")
                with gr.Accordion("交渉の注意書き【重要】（クリックして開く）", open=False):
                    caution_display = gr.Markdown("ここに注意書きが表示されます")
                #gr.Markdown("### 注意書き")
                #caution_display = gr.Markdown("ここに注意書きが表示されます")
            with gr.Column(scale=7):
                gr.Markdown("### 交渉チャット")
                # ★★★ 修正 (Warning対応) ★★★
                chatbot_display = gr.Chatbot(
                    label="対話ログ", 
                    height=400, 
                    type='messages' # (タプル形式ではなく辞書形式 {role: 'user', content: '...'} を使う)
                )
                chat_input = gr.Textbox(label="メッセージ", placeholder="メッセージを入力してEnterキーを押してください...")
                
                with gr.Row():
                    send_button = gr.Button("送信", variant="primary")
                    agree_button = gr.Button("交渉合意", variant="stop")
                    reject_button = gr.Button("交渉非合意", variant="stop")

    # --- ③ 評価画面 ---
    with gr.Group(visible=False) as evaluation_group:
        gr.Markdown("## 評価")
        gr.Markdown("交渉は終了しました。")

        # ★★★ 追加: 対話ログ確認用のアコーディオン ★★★
        with gr.Accordion("▼ 対話ログを確認する（クリックして開閉）", open=False):
            evaluation_chatbot_display = gr.Chatbot(
                label="今回の対話ログ", 
                type='messages', 
                height=300
            )

        gr.Markdown("今回の交渉相手の「人間らしさ」について, 以下の項目を5段階で評価してください。")
        hl1_slider = gr.Slider(minimum=1, maximum=5, step=1, label="評価項目1：「非機械性」：エージェントの反応に同じ文の反復など, 機械的・事務的なプログラムらしさはなく, 人間らしさがありましたか？(1: 機械的 〜 5: 人間らしい)", value=3)
        hl2_slider = gr.Slider(minimum=1, maximum=5, step=1, label="評価項目2：「説得の論理性」：価格に対するエージェントの説明や理由は、人間が話す内容として納得できるものでしたか？(1: 非論理的 〜 5: 論理的)", value=3)
        hl3_slider = gr.Slider(minimum=1, maximum=5, step=1, label="評価項目3：「話の簡潔性」：エージェントの反応は論点が1,2点にまとまって簡潔なものでしたか？(1: 冗長 〜 5: 簡潔))", value=3)
        hl4_slider = gr.Slider(minimum=1, maximum=5, step=1, label="評価項目4：「自然さ」：意味が理解できない応答や, 不当な価格提案はなく, 交渉者としてエージェントの反応は自然なものでしたか？(1: 不自然 〜 5: 自然))", value=3)
        hl5_slider = gr.Slider(minimum=1, maximum=5, step=1, label="評価項目5：「交渉力」：エージェントは直接的な価格交渉のみならず, 商品の状態への質問など対話を深めて交渉を有利に進めようとしていましたか？ (1: 交渉力がない 〜 5: 交渉力がある))", value=3)
        evaluation_submit_button = gr.Button("評価を送信して次へ", variant="primary")

    # --- ④ 終了画面 ---
    with gr.Group(visible=False) as end_group:
        gr.Markdown("## 実験終了")
        gr.Markdown("全ての交渉が終了しました。ご協力ありがとうございました。")
        gr.Markdown("このウィンドウを閉じて実験を終了してください。終了後, **<span style='color: red; '>slackのtime-morimotoに終了スタンプ</span>** をお願いいたします。")

    # -----------------------------------------------------------------
    # 6. イベント紐付け
    # -----------------------------------------------------------------

    # [流れ1] スタートボタン
    start_button.click(
        fn=handle_start_experiment,
        inputs=[],
        outputs=[
            state_session_id,
            state_task_queue,
            state_all_results,
            state_human_agent,
            state_ai_agent,
            state_config,
            state_extractor,
            start_group,
            negotiation_group,
            scenario_display,
            operate_display,
            caution_display,
            chatbot_display
        ]
    )

    # [流れ2] チャット送信 (Enterキー)
    chat_input.submit(
        fn=handle_chat_message,
        inputs=[
            chat_input, chatbot_display,
            state_human_agent, state_ai_agent,
            state_config, state_extractor
        ],
        outputs=[
            chatbot_display, chat_input,
            state_human_agent, state_ai_agent,
            negotiation_group, evaluation_group,
            send_button, agree_button, reject_button,
            evaluation_chatbot_display
        ]
    )
    # [流れ2] チャット送信 (送信ボタン)
    send_button.click(
        fn=handle_chat_message,
        inputs=[
            chat_input, chatbot_display,
            state_human_agent, state_ai_agent,
            state_config, state_extractor
        ],
        outputs=[
            chatbot_display, chat_input,
            state_human_agent, state_ai_agent,
            negotiation_group, evaluation_group,
            send_button, agree_button, reject_button,
            evaluation_chatbot_display
        ]
    )
    
    # [流れ2 亜種] 合意ボタン
    agree_button.click(
        fn=lambda *args: handle_finish_negotiation("accept", *args),
        inputs=[
            chatbot_display,
            state_human_agent, state_ai_agent,
            state_config, state_extractor
        ],
        outputs=[
            chatbot_display,
            state_human_agent, state_ai_agent,
            negotiation_group, evaluation_group,
            evaluation_chatbot_display
        ]
    )
    # [流れ2 亜種] 非合意ボタン
    reject_button.click(
        fn=lambda *args: handle_finish_negotiation("reject", *args),
        inputs=[
            chatbot_display,
            state_human_agent, state_ai_agent,
            state_config, state_extractor
        ],
        outputs=[
            chatbot_display,
            state_human_agent, state_ai_agent,
            negotiation_group, evaluation_group,
            evaluation_chatbot_display
        ]
    )

    # [流れ3] 評価送信ボタン
    evaluation_submit_button.click(
        fn=handle_submit_evaluation,
        inputs=[
            hl1_slider, hl2_slider, hl3_slider, hl4_slider, hl5_slider,
            state_human_agent, state_ai_agent,
            state_all_results,
            state_task_queue,
            state_config,
            state_session_id
        ],
        outputs=[
            # (returnの順番と数を合わせる)
            state_all_results, state_task_queue,
            state_human_agent, state_ai_agent,
            state_config,
            evaluation_group, negotiation_group, end_group,
            scenario_display, operate_display, caution_display, chatbot_display,
            evaluation_chatbot_display
        ]
    )

# -----------------------------------------------------------------
# 7. アプリの起動
# -----------------------------------------------------------------
if __name__ == "__main__":
    try:
        GradioHumanAgent._get_parser_components()
    except Exception as e:
        print(f"HumanAgentのTransformerモデルのロードに失敗しました: {e}")
        print("パス (checkpoint) が正しいか確認してください。")
    
    # ログイン処理
    auth_users = [
        ("morimoto", "55katfuji!!"), # (ID, Password)
        ("kitashima", "55katfuji!!"),
        ("genseki", "55katfuji!!"),
        ("honda", "55katfuji!!"),
        ("watanabe", "55katfuji!!"),
        ("fukutoku", "55katfuji!!"),
        ("kobayashi", "55katfuji!!"),
        ("mochizuki", "55katfuji!!"),
        ("sota", "55katfuji!!"),
        ("kon", "55katfuji!!"),
        ("matsumoto", "55katfuji!!"),
    ]

    print("--- 実験アプリを起動します ---")
    print("アクセス時にIDとパスワードが求められます。")
    
    demo.launch(
        server_name="0.0.0.0", # ローカルネットワーク内からもアクセス可能にする設定
        server_port=7860,
        auth=auth_users,       # ★ ここで認証を設定
        share=False            # ngrokを使うので、Gradio標準のshareはFalseにしておきます
    )