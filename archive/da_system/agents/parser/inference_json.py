# 作成したモデルによる推論プログラム(これが完成版！！！)

import pandas as pd
import linecache
import json
from glob import glob
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.nn.functional import softmax

# GPUの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 結果をまとめる用のリスト
results = []

# 作成したモデルの読み込み
checkpoint = "fold_1/checkpoint-14512" # 好きなモデルを選択
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
model = model.to(device) # GPUにモデルを送る

# jsonファイルを開く
with open("model/data/train-parsed_nonskip.json") as f1:
    data = json.load(f1)

print(len(data))
# 各対話の情報がリストで格納されているので1対話ずつ処理
with torch.no_grad():
    model.eval()
    for i in range(len(data)):
        print(f"Dialogue {i + 1}") # 何対話目まで処理したかの確認
        events = data[i]["events"] # テキストが入っているeventsを取り出す
        for j in range(len(events)):
            text = events[j]["data"] # テキストを取り出す
            if j == 0:
                pre_text = "[PAD]"
            else:
                if type(events[j-1]["data"]) is str:
                    pre_text = events[j-1]["data"] # 一つ前の発話を取得
                else:
                    if type(events[j-2]["data"]) is str:
                        pre_text = events[j-2]["data"]
                    else:
                        pre_text = "[PAD]"

            # テキストのみにダイアログアクトをつける(accept, reject, offer, quitはすでにつけてある)
            if type(text) is str:
                inputs = tokenizer(pre_text, text, max_length=512, truncation=True, return_tensors="pt")
                inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
                outputs = model(**inputs)

                # 推論結果の取得
                logits = outputs.logits # ロジットの取得
                probabilities = softmax(logits, dim=1) # ロジットをソフトマックス関数で確率に変換
                predicted_class = torch.argmax(probabilities, dim=1).item() # 確率が最も高いものを推定ラベルとして決定
                print("a: ", predicted_class)
                predicted_class = model.config.id2label[predicted_class] # ラベル番号をダイアログアクトに変換
                print("b: ", predicted_class)
                results.append({'text': text, 'predicted_class': predicted_class, 'probabilities': probabilities.cpu().numpy().tolist()}) # 結果の保存

                events[j]["metadata"]["intent"] = predicted_class # 推論したダイアログアクトで上書き

        data[i]["events"] = events

# 結果の表示及び保存
result_df = pd.DataFrame(results)
print(result_df)
result_df.to_csv('inference_results_test.csv', index=False) # どの文が何のダイアログアクトに分類されて, 各ラベルの確率は何だったのかを記録

# jsonファイルに保存し直し
with open("train-parsed_test.json", "w") as f2:
    json.dump(data, f2)

f1.close()
f2.close()