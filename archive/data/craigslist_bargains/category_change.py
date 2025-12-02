# category_change.py
import os, dspy
from typing import Literal
from dspy.evaluate import Evaluate
import pandas as pd

VehiclesType = Literal['bike', 'car']
ElectronicsType = Literal['electronics', 'phone']

class VehiclesClassifier(dspy.Signature):
    """Read the vehicle's product name and description and categorize it as a car or a bike."""
    
    # --- 入力フィールド ---
    title: str = dspy.InputField(desc="Product name")
    description: str = dspy.InputField(desc="Detailed product description")

    # --- 出力フィールド ---
    category: VehiclesType = dspy.OutputField(
        desc="Product category. Output only one word: car or bike"
    )

class ElectronicsClassifier(dspy.Signature):
    """Read the product name and description of the electrical appliance and classify it as either a phone such as a mobile phone or smartphone, or another electronics."""
    
    # --- 入力フィールド ---
    title: str = dspy.InputField(desc="Product name")
    description: str = dspy.InputField(desc="Detailed product description")

    # --- 出力フィールド ---
    category: ElectronicsType = dspy.OutputField(
        desc="Product category. Output only one word: electronics or phone"
    )

def category_change():

    filepath = "archive/data/craigslist_bargains/test.csv"
    df = pd.read_csv(filepath)

    lm = dspy.LM(
        model="ollama/llama3.3:70b",
        provider="ollama",
    )

    vehicle_changer = dspy.Predict(VehiclesClassifier)
    electronics_changer = dspy.Predict(ElectronicsClassifier)

    new_data = []

    with dspy.context(lm=lm):
        for index, row in df.iterrows():
            if row['category'] == 'vehicles':
                modified_row = row.copy()
                context = {
                    'title' : modified_row['title'],
                    'description' : modified_row['description']
                }
                prediction = vehicle_changer(**context)

                modified_row['category'] = prediction['category']
                new_data.append(modified_row)

            elif row['category'] == 'electronics':
                modified_row = row.copy()
                context = {
                    'title' : modified_row['title'],
                    'description' : modified_row['description']
                }
                prediction = electronics_changer(**context)

                modified_row['category'] = prediction['category']
                new_data.append(modified_row)
            else:
                new_data.append(row)

    new_df = pd.DataFrame(new_data)
    new_df.to_csv('new.csv', index=False)

if __name__ == "__main__":
    agent = category_change()