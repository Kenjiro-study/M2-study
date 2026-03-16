import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import argparse

def run_statistical_test(df):
    metrics = ['非機械性', '論理性', '簡潔性', '自然さ', '交渉力']
    targets = ['Search', 'Simple', 'All'] # DAMFと比較する相手
    
    results = []

    print(f"{'指標':<10} | {'比較':<6} | {'p値 (補正前)':<10} | {'p値 (Holm補正後)':<12} | {'有意差'}")
    print("-" * 60)

    summary_data = []

    for metric in metrics:
        # 各エージェントのデータを抽出
        data_a = df[df['Agent'] == 'DAMF'][metric].values
        data_b = df[df['Agent'] == 'Search'][metric].values
        data_c = df[df['Agent'] == 'Simple'][metric].values
        data_d = df[df['Agent'] == 'All'][metric].values

        # 1. Friedman検定 (4群に差があるか？)
        stat, p_friedman = stats.friedmanchisquare(data_a, data_b, data_c, data_d)
        print("stat: ", stat)
        print("p_friedman: ", p_friedman)
        
        # Friedmanで有意差があった場合のみ事後検定へ
        # (傾向を見るため今回は0.05以上でも事後検定の結果を表示させますが、
        #  厳密にはFriedmanが有意でないと事後検定は無効とされることが多いです)
        
        p_vals_raw = []
        comparisons = []
        
        # A vs B, A vs C, A vs D の検定
        for target_agent, target_data in zip(['Search', 'Simple', 'All'], [data_b, data_c, data_d]):
            # Wilcoxonの符号付順位検定
            # alternative='greater' は「Aの方が値が大きい」という仮説
            # 両側検定にしたい場合は 'two-sided' に変更してください
            # 今回は「Aの性能が良いこと」を示したいので片側(greater)または両側を使います。
            # 一般的には「差があるか」を見る両側(two-sided)が無難です。
            s, p = stats.wilcoxon(data_a, target_data, alternative='two-sided')
            p_vals_raw.append(p)
            comparisons.append(f"DAMF vs {target_agent}")

        # Holm補正 (多重比較によるp値の調整)
        reject, p_vals_corrected, _, _ = multipletests(p_vals_raw, method='holm')

        # 結果の格納
        for comp, p_raw, p_corr, rej in zip(comparisons, p_vals_raw, p_vals_corrected, reject):
            sign = "**" if p_corr < 0.01 else ("*" if p_corr < 0.05 else ("†" if p_corr < 0.1 else "n.s."))
            print(f"{metric:<10} | {comp:<6} | {p_raw:.4f}       | {p_corr:.4f}         | {sign}")
            
            summary_data.append({
                "Metric": metric,
                "Comparison": comp,
                "p_raw": p_raw,
                "p_holm": p_corr,
                "Significance": sign
            })
            
    return pd.DataFrame(summary_data)

def main():
    parser = argparse.ArgumentParser(description="negotiation analysis")
    parser.add_argument(
        "--agent",
        default=None,
        help="Custom name for this experiment run"
    )
    args = parser.parse_args()

    data = {
        'Agent': ['DAMF']*22 + ['Search']*22 + ['Simple']*22 + ['All']*22,
        '非機械性': np.concatenate([
            [4,5,3,4,3,2,3,5,5,3,3,4,5,5,5,5,4,3,2,5,2,3], 
            [1,1,3,1,4,5,1,4,2,4,1,4,1,2,3,4,2,4,5,4,2,3], 
            [3,3,3,1,5,5,4,5,4,5,3,4,5,5,5,3,4,3,4,5,1,5], 
            [3,2,1,1,4,2,2,2,3,2,4,3,5,4,4,5,2,1,5,4,5,1]
        ]),
        '論理性': np.concatenate([
            [3,5,4,5,5,4,3,4,4,3,3,4,5,5,5,4,4,5,2,5,2,3], 
            [2,2,2,1,4,5,1,4,2,3,1,4,1,3,3,4,2,2,5,4,1,2], 
            [4,2,2,1,4,5,3,5,3,4,2,3,4,5,4,4,3,2,4,4,1,5], 
            [3,2,1,3,4,4,3,2,3,1,4,3,4,4,4,4,3,2,5,4,5,2]
        ]),
        '簡潔性': np.concatenate([
            [5,5,4,5,5,4,3,4,5,5,4,4,5,5,5,5,5,4,3,5,5,5], 
            [5,5,5,5,4,5,1,5,3,5,3,4,5,5,3,5,4,3,5,3,5,4], 
            [4,5,5,5,4,5,4,5,4,5,3,4,5,5,5,5,4,2,4,4,5,5], 
            [4,4,5,5,4,4,4,3,5,5,4,3,5,5,5,5,5,5,5,4,5,5]
        ]),
        '自然さ': np.concatenate([
            [4,5,2,4,4,5,3,4,4,4,5,5,5,5,3,4,4,3,2,5,1,3], 
            [1,1,4,1,2,3,1,5,2,4,1,5,1,5,2,2,3,3,5,2,2,3], 
            [5,2,3,1,2,4,4,5,5,4,3,4,5,5,5,3,4,4,3,4,1,5], 
            [5,3,1,2,3,5,2,1,4,2,5,4,4,4,1,4,3,1,5,2,5,1]
        ]),
        '交渉力': np.concatenate([
            [3,5,4,4,3,4,4,3,4,3,2,3,5,5,3,4,3,3,3,5,1,2], 
            [2,1,5,3,2,2,1,4,2,4,1,3,1,2,2,3,1,3,3,2,1,3], 
            [3,2,3,1,4,1,4,4,2,2,2,2,3,3,5,4,4,2,3,4,1,5], 
            [3,1,1,2,5,2,2,2,2,3,4,2,5,2,3,2,2,1,4,3,5,1]
        ]),
    }

    df = pd.DataFrame(data)
    if args.agent == "seller":
        df = df[::2] # sellerの結果だけで分析
    elif args.agent == "buyer":
        df = df[1::2] # buyerの結果だけで分析
    
    result = run_statistical_test(df)
    print(result)

if __name__ == "__main__":
    main()