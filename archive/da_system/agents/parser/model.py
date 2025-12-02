# 事前学習済みBERTモデルをアノテーションしたcbデータセットでファインチューニングするプログラム(これが完成版!!!)
import torch
import time
import pandas as pd
import linecache
from glob import glob
from datasets import Dataset, Value, Features, ClassLabel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback

"""
追加部分2024/05/21
学習時のruntimeを調べるためのクラス
"""
class RuntimeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("Training started")

    def on_train_end(self, args, state, control, **kwargs):
        end_time = time.time()
        runtime = end_time - self.start_time
        print(f"Training finished in {runtime:.2f} seconds")

#---------------------------GPUの指定-----------------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---------------------tokenizerの準備------------------------#
checkpoint = "roberta-base" # モデルによってここを変更
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#---------アノテーションしたcbデータセットをDataframeで読み込み---------#
df = pd.read_csv("data/cb_dataset_dia.csv") #データによってここを変更
df.drop("Unnamed: 0", axis=1, inplace=True)
#print(df)

"""
追加部分 previous_textを作ろう
"""
#print(df['text'][0])
pre_list = ['[PAD]']
for i in range(df.shape[0]-1):
    if df['text'][i+1] == '<end>':
        continue
    elif df['text'][i] == '<end>':
        pre_list.append('[PAD]')
    else:
        pre_list.append(df['text'][i])

"""
追加部分 textとmeta_textの<end>を消そう
"""
#print(df['text'][0])
text_list = []
meta_list = []
for i in range(df.shape[0]):
    if df['text'][i] == '<end>':
        continue
    else:
        text_list.append(df['text'][i])
        meta_list.append(df['meta_text'][i])

data = dict(text=text_list, pre_text=pre_list, meta_text=meta_list)
df = pd.DataFrame(data=data)

#------------------ダイアログアクトを数値に変換----------------------#
# ダイアログアクトの種類を取得
categories = list(set(df['meta_text']))
#print(categories)

# カテゴリーのID辞書を作成
id2cat = dict(zip(list(range(len(categories))), categories))
cat2id = dict(zip(categories, list(range(len(categories)))))
#print(id2cat)
#print(cat2id)

# DataFrameにカテゴリーID列を追加
df['label'] = df['meta_text'].map(cat2id)

# データセットを本文とカテゴリーID列だけにする
df = df[['text', 'pre_text', 'label']]
#print(df.head())

#print("\n")

#-----Hugging Faceのdatasetsライブラリを使用して, transformersで扱える構造に変換-----#
features = Features({"text": Value("string"), "pre_text": Value("string"), "label": ClassLabel(num_classes=len(categories), names=categories)}) # classlabelの定義
dataset_packed = Dataset.from_pandas(df, features=features) # labelをclasslabelとして保存
#print(dataset_packed)

#---------------交差検証のための分割器の定義(StratifiedKFold)---------------#
num_splits = 5 # 5分割
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42) # データのシャッフルあり, シード固定

# 各foldの性能を格納するリスト
fold_accuracies = []

#--------------------交差検証+ファインチューニング--------------------#
for fold, (train_index, test_index) in enumerate(skf.split(dataset_packed, dataset_packed["label"])):
    print(f"Fold {fold + 1}/{num_splits}")

    # データセットの作成
    train_dataset = dataset_packed.select(train_index)
    test_dataset = dataset_packed.select(test_index)

    # トークナイズ
    def preprocess_function(examples):
        MAX_LENGTH = 512
        return tokenizer(examples["pre_text"], examples["text"], max_length=MAX_LENGTH, truncation=True)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 評価関数の定義
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy':acc, 'f1':f1}
    
    # モデルの初期化
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=12)
    model = model.to(device)

    # -----------------ファインチューニング------------------ #
    training_args = TrainingArguments(
        output_dir = f"model/fold_{fold + 1}",
        evaluation_strategy = 'epoch',
        logging_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit = 1,
        learning_rate = 2e-5,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        num_train_epochs = 8,
        weight_decay = 0.01,
        no_cuda = False
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = tokenizer,
        data_collator = data_collator,
        callbacks=[RuntimeCallback()]
    )

    # ラベル情報の設定(どの数字がどのラベルに対応づいているか)
    id2label = {}
    for i in range(train_dataset.features["label"].num_classes):
        id2label[i] = train_dataset.features["label"].int2str(i)

    label2id = {}
    for i in range(train_dataset.features["label"].num_classes):
        label2id[train_dataset.features["label"].int2str(i)] = i

    trainer.model.config.id2label = id2label
    trainer.model.config.label2id = label2id

    trainer.train()

    # モデルの評価
    results = trainer.evaluate()

    # Accuracyを取得してリストに追加
    accuracy = results["eval_accuracy"]
    fold_accuracies.append(accuracy)

    # 学習させたモデルでtestデータの予測する
    pred_result = trainer.predict(test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
    pred_label = pred_result.predictions.argmax(axis=1).tolist()

    # scikit-learnによる予測結果のレポートのcsvファイルへの出力
    report = classification_report(test_dataset['label'], pred_label, target_names=categories, output_dict=True)
    report_df = pd.DataFrame(report).T
    print(report_df)

    if (fold + 1) == 1:
        fold1_result = report_df
    elif (fold + 1) == 2:
        fold2_result = report_df
    elif (fold + 1) == 3:
        fold3_result = report_df
    elif (fold + 1) == 4:
        fold4_result = report_df
    elif (fold + 1) == 5:
        fold5_result = report_df

    report_df.to_csv(f"model/fold_{fold + 1}/report_roberta.csv") # モデルごとに名前変更

# 各foldの性能を表示
for fold, accuracy in enumerate(fold_accuracies):
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

fold1_result['precision'] = (fold1_result['precision'] + fold2_result['precision'] + fold3_result['precision'] + fold4_result['precision'] + fold5_result['precision']) / 5
fold1_result['recall'] = (fold1_result['recall'] + fold2_result['recall'] + fold3_result['recall'] + fold4_result['recall'] + fold5_result['recall']) / 5
fold1_result['f1-score'] = (fold1_result['f1-score'] + fold2_result['f1-score'] + fold3_result['f1-score'] + fold4_result['f1-score'] + fold5_result['f1-score']) / 5

# 全体の性能を計算して表示
average_accuracy = sum(fold_accuracies) / num_splits
print(f"Average Accuracy: {average_accuracy:.4f}")
print(fold1_result)
fold1_result.to_csv("model/report_roberta_test.csv") # モデルごとに名前変更