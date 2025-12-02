# アノテーションしたcbデータセットのダイアログアクトの割合を調べるプログラム
import pandas as pd
import re

df = pd.read_csv("data/cb_dataset.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

label1 = 0
label2 = 0
label3 = 0
label4 = 0
label5 = 0
label6 = 0
label7 = 0
label8 = 0
label9 = 0
label10 = 0
label11 = 0
label12 = 0

for i in range(df.shape[0]):
    # textとintentを一つ取り出す
    meta = (df.loc[i,"meta_text"])

    if meta == "intro":
        label1 += 1
    elif meta == "init-price":
        label2 += 1
    elif meta == "vague-price":
        label3+= 1
    elif meta == "insist":
        label4 += 1
    elif meta == "counter-price":
        label5+= 1
    elif meta == "inquire":
        label6+= 1
    elif meta == "inform":
        label7 += 1
    elif meta == "agree":
        label8 += 1
    elif meta == "disagree":
        label9 += 1
    elif meta == "supplemental":
        label10 += 1
    elif meta == "thanks":
        label11 += 1
    elif meta == "unknown":
        label12 += 1
    else:
        print("error")

print("        intro : ", label1)
print("   init-price : ", label2)
print("  vague-price : ", label3)
print("       insist : ", label4)
print("counter-price : ", label5)
print("      inquire : ", label6)
print("       inform : ", label7)
print("        agree : ", label8)
print("     disagree : ", label9)
print(" supplemental : ", label10)
print("       thanks : ", label11)
print("      unknown : ", label12)
print("\n")
print("    dialogues : ", label1+label2+label3+label4+label5+label6+label7+label8+label9+label10+label11+label12)