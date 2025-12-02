# manager_test.py
import os, dspy
from typing import Dict, List, Optional
from dspy.evaluate import Evaluate
import pandas as pd

class NegotiationParser(dspy.Signature):
    """The agent classifies utterances in a price negotiation dialogue into the following labels, taking into account the input utterance and intent.
Choose the appropriate label from the presented options that best matches the meaning of the sentence.
    # classification criteria
    - intro: Greetings or product introductions to start negotiations.
    - inquire: Question about the product.
    - inform: Response to the question.
    - init-price: Initial price offer in a negotiation dialogue.
    - vague-price: Negotiate without disclosing the price.
    - counter-price: Counter price proposal.
    - insist: Offer the same price without changing your mind.
    - agree: Agree to the other party's price offer.
    - disagree: Reject the price offer from the other party and end the negotiation.
    - supplemental: This is supplemental information about the product. If the product description does not fit into "inform", please select this.
    - thanks: Word of thanks.
    - unknown: Texts that do not fall into any other labe."""
    
    # --- 入力フィールド ---
    dialogue_history = dspy.InputField(desc="Dialogue history")
    target_text = dspy.InputField(desc="Target text to classify into labels")

    # --- 出力フィールド ---
    intent = dspy.OutputField(
        desc="A classification label that represents the intent of the target text. Select one of the following 12 types: "
             "intro, inquire, inform, init-price, vague-price, counter-price, insist, agree, disagree, supplemental, thanks, unknown"
    )

def load_examples_from_csv(filepath):
    
    df = pd.read_csv(filepath)
    df = df.drop("Unnamed: 0", axis=1)
    
    examples = []
    dialogue = []

    for i in range(len(df)):
        text = df.loc[i, 'text']
        if text == '<end>':
            dialogue = []
            continue

        example = dspy.Example(
            dialogue_history=dialogue.copy(),
            target_text=text,
            intent=df.loc[i, 'meta_text']
        ).with_inputs("dialogue_history", "target_text")

        dialogue.append(text)
        examples.append(example)
        
    return examples

def test_manager():

    lm = dspy.LM(
        model="ollama/llama3.1",
        provider="ollama",
    )

    dspy.settings.configure(lm=lm)
    manager = dspy.ChainOfThought(NegotiationParser)

    filepath = "archive/da_system/agents/parser/data/cb_dataset_0~999.csv"
    val_data = load_examples_from_csv(filepath)

    count = 0

    for i, example in enumerate(val_data):
        prediction = manager(
            dialogue_history = example.dialogue_history,
            target_text = example.target_text,
        )

        if example.intent == prediction.intent:
            count += 1
        print(f"--- データ {i+1} ---")
        #print("prediction: ", prediction)
        #print("pertner_utterance: ", example.partner_utterance)
        #print("true_intent: ", example.next_intent)

    print(f"正解率： {count}/{len(val_data)}")


if __name__ == "__main__":
    agent = test_manager()