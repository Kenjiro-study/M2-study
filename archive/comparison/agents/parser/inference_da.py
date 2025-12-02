# 作成したパーサーによる推論プログラム(これが完成版！！！)
# 修論バージョン！

import pandas as pd
from glob import glob
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# GPUの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 結果をまとめる用のリスト
results = []

# 作成したモデルの読み込み
checkpoint = "archive/da_system/agents/parser/model/roberta_fold_1/checkpoint-82304" # 好きなモデルを選択
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
model = model.to(device) # GPUにモデルを送る

df = pd.read_csv("archive/da_system/agents/parser/data/cb_dataset_0~999.csv")
df = df.drop("Unnamed: 0", axis=1)
print(df)

pre_text = "[PAD]"
count = 0
true_count = 0

with torch.no_grad():
    model.eval()
    for i in range(len(df)):
        target_text = df.at[i, "text"]
        if target_text == "<end>":
            pre_text = "[PAD]"
            continue
        target_intent = df.at[i, "meta_text"]

        inputs = tokenizer(pre_text, target_text, max_length=512, truncation=True, return_tensors="pt")
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        outputs = model(**inputs)

        # 推論結果の取得
        logits = outputs.logits # ロジットの取得
        probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
        predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
        #print("a: ", predicted_class)
        predicted_class = model.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換
        #print("b: ", predicted_class)
        results.append({'text': target_text, 'predicted_class': predicted_class, 'probabilities': probabilities.cpu().numpy().tolist()}) # 結果の保存

        count += 1
        if predicted_class == target_intent:
            true_count += 1

        pre_text = target_text

result_df = pd.DataFrame(results)
print(result_df)
print(f"true_rate: {true_count} / {count}")
#result_df.to_csv('inference_results_test.csv', index=False) # どの文が何のダイアログアクトに分類されて, 各ラベルの確率は何だったのかを記録