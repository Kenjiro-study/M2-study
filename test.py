import gradio as gr
import random
import datetime
import json
import time
import uuid # ★ 修正点1 (セッションID用)

# -----------------------------------------------------------------
# 1. 必要なクラスのインポート
# -----------------------------------------------------------------
# ( ... ここにあなたのプロジェクトから必要なクラスをimportする ... )
# 例：
# from agents.base_agent import BuyerAgent, SellerAgent # (仮のクラス名)
# from negotiation.config import NegotiationConfig # (仮のクラス名)
# from negotiation.metrics import NegotiationMetrics # (仮のクラス名)
# from extraction.extractor import PriceExtractor # (仮のクラス名)
# from scenarios.manager import ScenarioManager # (仮のクラス名)

# --- (仮のダミークラス) ---
# 実際のimportが動くまで、これらでテストできます
class NegotiationConfig:
    def __init__(self, scenario=None, buyer_model=None, seller_model=None, max_turns=10):
        self.scenario = scenario
        self.max_turns = max_turns
        print("NegotiationConfig initialized")

class NegotiationMetrics:
    def __init__(self, start_time=None):
        self.start_time = start_time
        self.end_time = None
        self.turns_taken = 0
        self.messages = []
        self.final_price = None
        self.buyer_utility = None
        self.seller_utility = None
        self.fairness = None
        print("NegotiationMetrics initialized")
    
    def to_dict(self):
        # ログ保存用に辞書化する
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "turns_taken": self.turns_taken,
            "final_price": self.final_price,
            "messages": self.messages,
            "buyer_utility": self.buyer_utility,
            "seller_utility": self.seller_utility,
            "fairness": self.fairness,
        }

class DummyAgent:
    def __init__(self, role, name):
        self.role = role
        self.name = name
        print(f"DummyAgent ({self.role} / {self.name}) initialized")
    
    def step(self, partner_data, extractor):
        print(f"Agent ({self.role}) received: {partner_data['content']}")
        # ( ... ここで dspy.step() が呼ばれる ... )
        time.sleep(1) # AIが考えているフリ
        
        # 相手が受け入れたら、こちらも受け入れる (ダミーロジック)
        if partner_data.get("intent") == "accept":
             return {
                "role": self.role,
                "content": "承知しました。その価格でお願いします。",
                "price": partner_data.get("price"),
                "intent": "accept"
            }
           
        return {
            "role": self.role,
            "content": f"私は{self.role} ({self.name}) です。ダミーの応答です。",
            "price": 150, # (仮)
            "intent": "counter"
        }
    
    def compute_utility(self, price):
        return 0.5 # (仮)

class PriceExtractor:
    def __init__(self):
        print("PriceExtractor initialized")
        
class DummyScenario:
     def __init__(self, id):
        self.scenario_id = id
        print("DummyScenario initialized")

# ここに「本番の8パターン」をすべて定義してください。
# このリストの長さが、そのまま実験回数になります。
ALL_AGENT_COMBINATIONS = [
    {"agent_name": "dspy_agent_A", "role": "buyer"},
    {"agent_name": "dspy_agent_B", "role": "buyer"}
    # ここに残り6個入れる
]

# (仮) シナリオマネージャーとExtractorのインスタンス化
# scenario_manager = ScenarioManager()
# price_extractor_instance = PriceExtractor()
# (ダミー)
price_extractor_instance = PriceExtractor()

# -----------------------------------------------------------------
# 2. イベントハンドラ関数群 (ロジック)
# -----------------------------------------------------------------

def handle_start_experiment():
    """
    [流れ1] 実験開始ボタンが押された時の処理
    """
    print("--- イベント: handle_start_experiment ---")
    
    # ★ 修正点1 (ログ上書き問題)
    # このセッション（被験者）固有のIDを生成
    session_id = str(uuid.uuid4())
    print(f"セッションID (被験者ID) を発行: {session_id}")
    
    # 1. これから実験する8パターンのリストを作成
    shuffled_combinations = random.sample(ALL_AGENT_COMBINATIONS, len(ALL_AGENT_COMBINATIONS))
    
    # 2. 最初の交渉を開始する準備 ( .pop() ではなく [0] と [1:] で扱う)
    if not shuffled_combinations:
        print("エラー: エージェントリストが空です")
        # (エラー表示)
        return (session_id, None, [], None, None, None, None, None, 
                gr.update(), gr.update(visible=False), 
                "エラー", "エラー", [])

    next_combo = shuffled_combinations[0]      # 先頭の要素を取得
    remaining_queue = shuffled_combinations[1:]  # 「残り」のリストを作成
    
    # --- ここであなたのコードの出番 ---
    # (仮) シナリオを取得
    # scenario = scenario_manager.get_scenario(...) 
    scenario = DummyScenario(id="scenario_001")
    
    human_role = "seller" if next_combo["role"] == "buyer" else "buyer"
    
    # (仮) NegotiationConfig を作成
    config = NegotiationConfig(
        scenario=scenario,
        max_turns=10
    )
    
    # (仮) エージェントを初期化 (エージェント側 *だけ* を初期化)
    # agent = YourAgentClass(...)
    agent = DummyAgent(role=next_combo["role"], name=next_combo["agent_name"])
    
    # (仮) Metrics を初期化
    metrics = NegotiationMetrics(start_time=datetime.datetime.now())
    
    # (仮) シナリオと注意書き
    scenario_text = f"あなたは **{human_role}** です。\n\nシナリオID: {scenario.scenario_id}\n\n[...ここにシナリオ詳細...]"
    caution_text = f"エージェント: **{next_combo['agent_name']}** ({next_combo['role']})\n\n[...ここに注意書き...]"

    # 戻り値で、UIの表示/非表示と、Stateの更新を行う
    return (
        session_id,                # [Output] state_session_id (★追加)
        remaining_queue,           # [Output] state_agent_queue (★ .pop() バグ修正)
        [],                        # [Output] state_all_results (リセット)
        agent,                     # [Output] state_current_agent
        human_role,                # [Output] state_human_role
        metrics,                   # [Output] state_metrics
        config,                    # [Output] state_config
        price_extractor_instance,  # [Output] state_extractor
        gr.update(visible=False),  # [Output] start_group
        gr.update(visible=True),   # [Output] negotiation_group
        scenario_text,             # [Output] scenario_display
        caution_text,              # [Output] caution_display
        []                         # [Output] chatbot_display (リセット)
    )

def handle_chat_message(
    human_message_text,       # [Input] chat_input
    chat_history,             # [Input] chatbot_display
    agent, metrics,           # [Input] 各 gr.State
    config, extractor, 
    human_role
):
    """
    [流れ2] 被験者がメッセージを送信した時の処理
    """
    print("--- イベント: handle_chat_message ---")
    
    # 1. 被験者 (Human) のターン
    chat_history.append((human_message_text, None))
    human_message_dict = {
        "role": human_role,
        "content": human_message_text, "price": None, "intent": "counter"
    }
    metrics.messages.append(human_message_dict)
    metrics.turns_taken += 1
    
    yield (
        chat_history, "", metrics, 
        gr.update(), gr.update(), 
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # ボタン類を無効化
    )

    # 2. エージェント (dspy) のターン
    try:
        agent_response_dict = agent.step(
            partner_data=human_message_dict,
            extractor=extractor
        )
    except Exception as e:
        print(f"エージェントエラー: {e}")
        agent_response_dict = {"role": agent.role, "content": f"エラーが発生しました: {e}", "price": None, "intent": "reject"}

    metrics.turns_taken += 1
    metrics.messages.append(agent_response_dict)
    chat_history.append((None, agent_response_dict["content"]))
    
    # 3. 交渉終了判定
    continue_negotiation = True
    if agent_response_dict['intent'] in ['accept', 'reject']:
        continue_negotiation = False
        metrics.end_time = datetime.datetime.now()
        metrics.final_price = (
            agent_response_dict['price'] if agent_response_dict['intent'] == 'accept'
            else None
        )
        
    if metrics.turns_taken >= config.max_turns:
        continue_negotiation = False
        metrics.end_time = datetime.datetime.now()
        print("最大ターン数に達しました。")
    
    if not continue_negotiation:
        # --- 交渉終了 → 評価画面へ ---
        if metrics.final_price:
            print(f"交渉成立: 最終価格 {metrics.final_price}")
        else:
            print("交渉不成立")
        # (...ここで _compute_final_metrics() 相当のロジック...)

        yield (
            chat_history, "", metrics, 
            gr.update(visible=False), # negotiation_group
            gr.update(visible=True),  # evaluation_group
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True) # ボタン類を有効化
        )
    else:
        # --- 交渉継続 ---
        yield (
            chat_history, "", metrics, 
            gr.update(), gr.update(), 
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True) # ボタン類を有効化
        )

def handle_finish_negotiation(
    decision_type,        # [Input] "accept" or "reject"
    chat_history,         # [Input] chatbot_display
    agent, metrics,       # [Input] 各 gr.State
    config, extractor, 
    human_role
):
    """
    [流れ2亜種] 被験者が「合意」「非合意」ボタンを押した時の処理
    """
    print(f"--- イベント: handle_finish_negotiation (type: {decision_type}) ---")

    # 1. 人間の最終意思決定
    human_message_text = "【交渉合意】" if decision_type == "accept" else "【交渉非合意】"
    chat_history.append((human_message_text, None))
    human_message_dict = {
        "role": human_role,
        "content": human_message_text,
        "price": metrics.messages[-1].get("price") if decision_type == "accept" and metrics.messages else None,
        "intent": decision_type
    }
    metrics.messages.append(human_message_dict)
    metrics.turns_taken += 1
    
    # 2. 終了処理
    metrics.end_time = datetime.datetime.now()
    metrics.final_price = human_message_dict["price"] if decision_type == "accept" else None
    
    if metrics.final_price:
        print(f"交渉成立 (人間): 最終価格 {metrics.final_price}")
    else:
        print("交渉不成立 (人間)")
    # (...ここで _compute_final_metrics() 相当のロジック...)

    # 3. 評価画面へ
    return (
        chat_history,
        metrics, 
        gr.update(visible=False), # negotiation_group
        gr.update(visible=True),  # evaluation_group
    )


def handle_submit_evaluation(
    human_likeness_score,  # [Input] human_likeness_slider
    metrics,               # [Input] state_metrics (計算済みのもの)
    all_results,           # [Input] state_all_results (今までの結果)
    agent_queue,           # [Input] state_agent_queue (残りのキュー)
    config,                # [Input] state_config
    session_id             # [Input] state_session_id (★追加)
):
    """
    [流れ3] 評価送信ボタンが押された時の処理
    """
    print("--- イベント: handle_submit_evaluation ---")
    
    # -------------------------------------------------
    # 1. 今回の結果を保存する
    # -------------------------------------------------
    current_result = {
        "session_id": session_id,
        "config": {
            "agent_name": (
                # agent.name や agent.role を config に含めるのが本当は望ましい
                metrics.messages[1].get("role") if len(metrics.messages) > 1 else "unknown"
            ),
            "scenario_id": config.scenario.scenario_id,
        },
        "metrics": metrics.to_dict(),
        "human_likeness": human_likeness_score,
    }
    all_results.append(current_result)
    
    # ★ 修正点1 (ログ上書き問題)
    # セッションIDを含めたファイル名で「追記」する
    log_filename = f"app_gradio_results_{session_id}.jsonl"
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(current_result, ensure_ascii=False) + "\n")
        print(f"結果を {log_filename} に追記しました。")
    except Exception as e:
        print(f"ログ保存エラー: {e}")

    # -------------------------------------------------
    # 2. 次の実験 or 終了 を判断する
    # -------------------------------------------------
    if not agent_queue:
        # --- 9. 全てのパターンが終了 ---
        print("全セッション終了。")
        
        # ★ 修正点1 (ログ上書き問題)
        # セッションIDを含めた「最終」ファイル名を「書き込み」する
        final_filename = f"app_gradio_results_FINAL_{session_id}.json"
        try:
            with open(final_filename, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"最終結果ファイル {final_filename} を保存しました。")
        except Exception as e:
            print(f"最終ログ保存エラー: {e}")
            
        # (★ .pop() バグ修正: return の数を outputs と合わせる)
        return (
            all_results, agent_queue,
            None, None, None, None, # 6 states
            gr.update(visible=False),   # evaluation_group
            gr.update(visible=False),   # negotiation_group
            gr.update(visible=True),    # end_group
            None, None, None            # 3 displays
        )
    
    else:
        # --- 8. 次の交渉セッションを開始 ---
        print("次のセッションへ...")
        
        # (★ .pop() バグ修正: [0] と [1:] で扱う)
        next_combo = agent_queue[0]
        remaining_queue = agent_queue[1:]

        # (仮) シナリオ・Config・エージェント・Metricsの初期化
        scenario = DummyScenario(id=f"scenario_{len(ALL_AGENT_COMBINATIONS) - len(remaining_queue)}")
        human_role = "seller" if next_combo["role"] == "buyer" else "buyer"
        config = NegotiationConfig(scenario=scenario, max_turns=10)
        agent = DummyAgent(role=next_combo["role"], name=next_combo["agent_name"])
        metrics = NegotiationMetrics(start_time=datetime.datetime.now())
        
        scenario_text = f"あなたは **{human_role}** です。\n\nシナリオID: {scenario.scenario_id}\n\n[...次のシナリオ...]"
        caution_text = f"エージェント: **{next_combo['agent_name']}** ({next_combo['role']})\n\n[...次の注意書き...]"

        return (
            all_results,             # [Output] state_all_results (更新)
            remaining_queue,         # [Output] state_agent_queue (★ .pop() バグ修正)
            agent,                   # [Output] state_current_agent
            human_role,              # [Output] state_human_role
            metrics,                 # [Output] state_metrics
            config,                  # [Output] state_config,
            gr.update(visible=False),  # [Output] evaluation_group
            gr.update(visible=True),   # [Output] negotiation_group
            gr.update(visible=False),  # [Output] end_group
            scenario_text,           # [Output] scenario_display
            caution_text,            # [Output] caution_display
            []                       # [Output] chatbot_display (リセット)
        )


# -----------------------------------------------------------------
# 3. UIの定義 (gr.Blocks)
# -----------------------------------------------------------------

# ★ 修正点2 (フォント問題)
# Noto Sans JP を指定したテーマを定義する
theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Noto Sans JP"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"]
)

with gr.Blocks(theme=theme) as demo: # 修正した theme を適用
    
    # -----------------------------------------------------------------
    # 状態管理 (UIに見えないデータ)
    # -----------------------------------------------------------------
    state_session_id = gr.State(None) # ★ 修正点1 (ログ上書き問題)
    state_agent_queue = gr.State([])
    state_all_results = gr.State([])
    state_current_agent = gr.State(None)
    state_human_role = gr.State(None)
    state_metrics = gr.State(None)
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
            この実験では、AIエージェントと提示された商品の価格交渉を行っていただきます。
            
            全部で **{len(ALL_AGENT_COMBINATIONS)}** 回、異なるエージェントまたは異なる役割（買い手・売り手）で交渉を行います。
            
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
                gr.Markdown("### 注意書き")
                caution_display = gr.Markdown("ここに注意書きが表示されます")
            with gr.Column(scale=7):
                gr.Markdown("### 交渉チャット")
                chatbot_display = gr.Chatbot(label="対話ログ", height=400, bubble_full_width=False)
                chat_input = gr.Textbox(label="メッセージ", placeholder="メッセージを入力してEnterキーを押してください...")
                
                with gr.Row():
                    send_button = gr.Button("送信", variant="primary")
                    agree_button = gr.Button("交渉合意", variant="stop")
                    reject_button = gr.Button("交渉非合意", variant="stop")

    # --- ③ 評価画面 ---
    with gr.Group(visible=False) as evaluation_group:
        gr.Markdown("## 評価")
        gr.Markdown("交渉は終了しました。")
        gr.Markdown("今回の交渉相手の「人間らしさ」を5段階で評価してください。")
        human_likeness_slider = gr.Slider(minimum=1, maximum=5, step=1, label="人間らしさ (1: 機械的 〜 5: 人間らしい)", value=3)
        evaluation_submit_button = gr.Button("評価を送信して次へ", variant="primary")

    # --- ④ 終了画面 ---
    with gr.Group(visible=False) as end_group:
        gr.Markdown("## 実験終了")
        gr.Markdown("全ての交渉が終了しました。ご協力ありがとうございました。")

    # -----------------------------------------------------------------
    # 4. イベント紐付け
    # -----------------------------------------------------------------

    # [流れ1] スタートボタン
    start_button.click(
        fn=handle_start_experiment,
        inputs=[],
        outputs=[
            state_session_id, # (★追加)
            state_agent_queue,
            state_all_results,
            state_current_agent,
            state_human_role,
            state_metrics,
            state_config,
            state_extractor,
            start_group,
            negotiation_group,
            scenario_display,
            caution_display,
            chatbot_display
        ]
    )

    # [流れ2] チャット送信 (Enterキー)
    chat_input.submit(
        fn=handle_chat_message,
        inputs=[
            chat_input, chatbot_display,
            state_current_agent, state_metrics, state_config, state_extractor, state_human_role
        ],
        outputs=[
            chatbot_display, chat_input, state_metrics,
            negotiation_group, evaluation_group,
            send_button, agree_button, reject_button
        ]
    )
    # [流れ2] チャット送信 (送信ボタン)
    send_button.click(
        fn=handle_chat_message,
        inputs=[
            chat_input, chatbot_display,
            state_current_agent, state_metrics, state_config, state_extractor, state_human_role
        ],
        outputs=[
            chatbot_display, chat_input, state_metrics,
            negotiation_group, evaluation_group,
            send_button, agree_button, reject_button
        ]
    )
    
    # [流れ2 亜種] 合意ボタン
    agree_button.click(
        fn=lambda *args: handle_finish_negotiation("accept", *args),
        inputs=[
            chatbot_display,
            state_current_agent, state_metrics, state_config, state_extractor, state_human_role
        ],
        outputs=[
            chatbot_display, state_metrics,
            negotiation_group, evaluation_group
        ]
    )
    # [流れ2 亜種] 非合意ボタン
    reject_button.click(
        fn=lambda *args: handle_finish_negotiation("reject", *args),
        inputs=[
            chatbot_display,
            state_current_agent, state_metrics, state_config, state_extractor, state_human_role
        ],
        outputs=[
            chatbot_display, state_metrics,
            negotiation_group, evaluation_group
        ]
    )

    # [流れ3] 評価送信ボタン
    evaluation_submit_button.click(
        fn=handle_submit_evaluation,
        inputs=[
            human_likeness_slider,
            state_metrics,
            state_all_results,
            state_agent_queue,
            state_config,
            state_session_id # (★追加)
        ],
        outputs=[
            # (returnの順番と数を合わせる)
            state_all_results, state_agent_queue,
            state_current_agent, state_human_role, state_metrics, state_config,
            evaluation_group, negotiation_group, end_group,
            scenario_display, caution_display, chatbot_display
        ]
    )

# -----------------------------------------------------------------
# 5. アプリの起動
# -----------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()