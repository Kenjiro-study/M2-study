# manager_test.py
import os, dspy, json
from typing import Dict, List, Optional
from dspy.evaluate import Evaluate

from ..strategies import STRATEGIES, CATEGORY_CONTEXT
from ..agents.buyer import BuyerAgent

class NegotiationManager(dspy.Signature):
    """As a price negotiation agent, considering the dialogue history, the partner's last utterance, roles, and your own strategy, select the single most strategic "intent" to take next.

    [THOUGHT PROCESS]
    1.  First, analyze the current dialogue history and the partner's most recent `partner_intent`.
    2.  Next, consider your own `agent_role` (e.g., Buyer or Seller) and `agent_strategy`.
    3.  Finally, strictly follow the "Intent Selection Criteria" below to select the single most appropriate intent label.
    
    [INTENT SELECTION CRITERIA (top priorityy)]
    - intro: Select at the beginning of the dialogue (e.g., when the history is empty or contains only greetings).
    - inquire: Select this when you need to ask about details such as the condition of the product.
    - inform: Select only as a direct response to the partner's `inquire` intent.
    - init-price: Select to make the *first* price proposal.
        (Condition: The `dialogue_history` and `partner_intent` must not yet contain `init-price`, `counter-price`, or `insist`.)
    - vague-price: Select to negotiate indirectly without stating a specific price (e.g., "That's a bit high...").
        (Condition: Select this *only if* concrete price negotiation is deadlocked.)
    - counter-price: Select to propose a *different* price after the partner has proposed an `init-price` or `counter-price`.
    - insist: Select to re-state your *previous price* after the partner has made a `counter-price`. 
    - supplemental: Select to provide additional information (e.g., product benefits) when the partner's intent was *not* `inquire`.
    - thanks: Select to express your gratitude for reaching an agreement.
        (Condition: Only if your partner's "partner_intent" is "agree" or "thanks")"""
    
    # --- 入力フィールド ---
    dialogue_history = dspy.InputField(desc="The past dialogue history with intent labels for each utterance.")
    partner_utterance = dspy.InputField(desc="The partner's most recent utterance to respond to.")
    partner_intent = dspy.InputField(desc="The intent label of the partner's most recent utterance.")
    partner_role = dspy.InputField(desc="The role of the partner (e.g., Buyer, Seller).")
    agent_role = dspy.InputField(desc="Your role (e.g., Buyer, Seller).")
    agent_strategy = dspy.InputField(desc="Your strategy for selecting an intent. This is a guideline; the 'INTENT SELECTION CRITERIA' above take precedence.")

    # --- 出力フィールド ---
    next_intent = dspy.OutputField(
        desc="The intent label for the agent's next action. Choose exactly one from the following 9 types: "
             "intro, inquire, inform, init-price, vague-price, counter-price, insist, supplemental, thanks"
    )

def load_examples_from_json(filepath):
    """JSONファイルを読み込み、dspy.Exampleのリストを返す"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # JSONの各オブジェクトをdspy.Exampleに変換
        example = dspy.Example(
            dialogue_history=item['dialogue_history'],
            partner_utterance=item['partner_utterance'],
            partner_intent=item['partner_intent'],
            partner_role=item['partner_role'],
            agent_role=item['agent_role'],
            next_intent=item.get('next_intent')
        ).with_inputs("dialogue_history", "partner_utterance", "partner_intent", "partner_role", "agent_role")
        
        examples.append(example)
        
    return examples

def test_manager():

    lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
    )

    dspy.settings.configure(lm=lm)
    manager = dspy.ChainOfThought(NegotiationManager)

    filepath = "archive/da_system/test/manager_val_data.json"
    val_data = load_examples_from_json(filepath)
    #strategy = STRATEGIES["fair"]
    #strategy = STRATEGIES["utility"]
    strategy = STRATEGIES["length"]

    count = 0

    for i, example in enumerate(val_data):
        if example.agent_role == "buyer":
            prediction = manager(
                dialogue_history = example.dialogue_history,
                partner_utterance = example.partner_utterance,
                partner_intent = example.partner_intent,
                partner_role = example.partner_role,
                agent_role = example.agent_role,
                agent_strategy = strategy['buyer_manager_style']
            )
        else:
            prediction = manager(
                dialogue_history = example.dialogue_history,
                partner_utterance = example.partner_utterance,
                partner_intent = example.partner_intent,
                partner_role = example.partner_role,
                agent_role = example.agent_role,
                agent_strategy = strategy['seller_manager_style']
            )
        
        if example.next_intent == prediction.next_intent:
            count += 1
        print(f"--- データ {i+1} ---")
        print("prediction: ", prediction)
        print("pertner_utterance: ", example.partner_utterance)
        print("true_intent: ", example.next_intent)

    print(f"正解率： {count}/50")

    #dialogue_history = [{"role":"buyer", "content":"i'm interested in this item , but i had some questions", "intent":"intro"},{"role":"seller", "content":"geat , ask away.", "intent":"intro"},{"role":"buyer", "content":"do i have to remove it myself?", "intent":"inquire"}, {"role":"seller", "content":"i am renting out the appartment , you \"don't remove anything\"", "intent":"inform"}]
    #partner_utterance = "nice. is it fully furnished?"
    #partner_intent = "inquire"
    #partner_role = "buyer"
    #prediction = manager(dialogue_history=dialogue_history, partner_utterance=partner_utterance, partner_intent=partner_intent, partner_role=partner_role)

    #lm.inspect_history(n=1)

if __name__ == "__main__":
    agent = test_manager()