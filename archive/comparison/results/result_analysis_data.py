import pandas as pd
import numpy as np
import io
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =========================================================
# データの入力エリア
# Excelからコピーした内容を、そのままダブルクォーテーション3つの間に貼り付けてください
# =========================================================
csv_text = """
エージェント	合意率	公平性	個人効用	対話の長さ
A	1	1	0.813	12
A	1	0.933	0.412	11
A	1	1	0.778	8
A	1	0.654	0.638	13
A	0			
A	1	0.813	0.509	15
A	0			
A	1	0.42	0.747	9
A	1	0.667	0.921	12
A	1	0.667	0.615	9
A	1	0	0.531	10
A	0			
A	1	0	0.697	4
A	1	0.664	0.33	9
A	1	0.87	0.601	12
A	1	0.676	0.591	11
A	1	0.775	0.667	14
A	1	1	0.474	11
A	1	0	0.426	14
A	0			
A	1	0	0.294	18
A	1	0.254	0.0499	17
B	1	0.378	0.623	12
B	0			
B	1	0.75	0.815	10
B	1	0.739	0.512	13
B	1	0.998	0.262	14
B	1	0.667	0.333	19
B	0			
B	1	0.875	0.318	17
B	0			
B	1	0.758	0.354	9
B	0			
B	0			
B	1	0	0	8
B	0			
B	1	0	0	18
B	1	0.53	0.7	13
B	1	0.6	0.409	18
B	1	0	0	11
B	1	0.36	0.41	18
B	1	0.275	0.0548	17
B	1	0	0.672	14
B	1	0.467	0.148	17
C	1	1	0	14
C	0			
C	1	1	0.895	12
C	1	0.667	0.583	20
C	1	0.913	0	8
C	0			
C	1	0.75	0	8
C	0			
C	0			
C	0			
C	0			
C	0			
C	1	0.788	0.633	6
C	1	0.333	0.767	5
C	1	0.525	0.662	6
C	1	0.667	0.49	17
C	1	0.225	0	12
C	0			
C	1	0	0	8
C	1	0.0999	0.00262	15
C	1	0	0	10
C	0			
D	0			
D	0			
D	0			
D	0			
D	1	0.222	0.512	8
D	1	0.636	0.364	9
D	0			
D	0			
D	1	0.333	0.857	6
D	0			
D	0			
D	1	1	0.429	13
D	1	0.222	0.78	4
D	0			
D	0			
D	0			
D	1	0.667	0.759	16
D	0			
D	1	0.636	0.135	18
D	0			
D	1	0.72	0.018	16
D	1	0.833	0.066	19
"""
# ↑ ここにB, C, Dのデータも続けて貼り付けてください（ヘッダーは最初だけでOKですが、あっても動きます）

# =========================================================
# データ読み込みと検定処理
# =========================================================

def analyze_quantitative_metrics(text_data):
    # 1. テキストデータをDataFrameに変換（Excelコピペはタブ区切りなので sep='\t'）
    df = pd.read_csv(io.StringIO(text_data), sep='\t')
    
    # 2. 列名のマッピング（日本語ヘッダーをコード用の英語名に変換）
    # ※Excelの列名が微妙に違う場合はここを修正してください
    df = df.rename(columns={
        'エージェント': 'Agent',
        '合意率': 'is_agreed',
        '公平性': 'fairness',
        '個人効用': 'utility',
        '対話の長さ': 'length'
    })
    
    print(f"データ読み込み完了: 全{len(df)}行")
    print(df.head()) # 確認用表示

    # --- A. 合意率の検定 ---
    print("\n" + "="*60)
    print("【1. 合意率 (Agreement Rate) の分析】")
    print("="*60)
    
    cross_tab = pd.crosstab(df['Agent'], df['is_agreed'])
    # 合意(1)の列が存在するか確認
    if 1 not in cross_tab.columns:
         print("データエラー: 合意(1)のデータが含まれていません")
         return

    success_counts = cross_tab[1]
    total_counts = cross_tab.sum(axis=1)
    rates = success_counts / total_counts
    
    print(f"{'Agent':<5} | {'合意数':<6} | {'全数':<5} | {'合意率'}")
    for agent in ['A', 'B', 'C', 'D']:
        if agent in success_counts:
            print(f"{agent:<5} | {success_counts[agent]:<6} | {total_counts[agent]:<5} | {rates[agent]:.3f}")
        else:
            print(f"{agent:<5} | データなし")

    # 全体でのカイ二乗検定
    chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
    print(f"\n全体検定 (Chi-square): p = {p:.4f}")
    
    if p < 0.05: # 有意差があれば事後検定
        print(">> 有意差あり。Aとのペア比較 (Fisher正確検定) を実施...")
        p_raw = []
        comparisons = []
        
        a_succ = success_counts['A']
        a_fail = total_counts['A'] - a_succ
        
        for target in ['B', 'C', 'D']:
            if target not in success_counts: continue
            t_succ = success_counts[target]
            t_fail = total_counts[target] - t_succ
            
            table = [[a_succ, a_fail], [t_succ, t_fail]]
            _, p_val = stats.fisher_exact(table, alternative='two-sided')
            p_raw.append(p_val)
            comparisons.append(f"A vs {target}")
            
        reject, p_corr, _, _ = multipletests(p_raw, method='holm')
        
        for comp, pr, pc, rej in zip(comparisons, p_raw, p_corr, reject):
            sign = "**" if pc < 0.01 else ("*" if pc < 0.05 else ("†" if pc < 0.1 else "n.s."))
            print(f"{comp}: p={pc:.4f} ({sign})")
    else:
        print(">> 合意率に有意差なし (n.s.)")

    # --- B. 連続値（効用・公平性・長さ）の検定 ---
    metrics_map = {'utility': '個人効用', 'fairness': '公平性', 'length': '対話長'}

    for metric_col, metric_name in metrics_map.items():
        print("\n" + "="*60)
        print(f"【2. {metric_name} ({metric_col}) の分析 (合意ケースのみ)】")
        print("="*60)
        
        # 欠損値(NaN)を除外してグループ化
        groups = {}
        for agent in ['A', 'B', 'C', 'D']:
            vals = df[df['Agent'] == agent][metric_col].dropna().values
            groups[agent] = vals
            print(f"Agent {agent}: n={len(vals)}, Mean={np.mean(vals):.3f}" if len(vals)>0 else f"Agent {agent}: データなし")
            
        # Kruskal-Wallis検定
        if any(len(g) == 0 for g in groups.values()):
            print("データ不足のため検定スキップ")
            continue
            
        stat, p_kw = stats.kruskal(groups['A'], groups['B'], groups['C'], groups['D'])
        print(f"\nKruskal-Wallis検定: p = {p_kw:.4f}")
        
        # 傾向を見るため常に事後検定を表示
        p_vals_raw = []
        comparisons = []
        
        for target in ['B', 'C', 'D']:
            s, p = stats.mannwhitneyu(groups['A'], groups[target], alternative='two-sided')
            p_vals_raw.append(p)
            comparisons.append(f"A vs {target}")
        
        reject, p_vals_corrected, _, _ = multipletests(p_vals_raw, method='holm')

        print(f"{'比較':<10} | {'p値 (Holm)':<12} | {'有意差'}")
        for comp, p_corr, rej in zip(comparisons, p_vals_corrected, reject):
            sign = "**" if p_corr < 0.01 else ("*" if p_corr < 0.05 else ("†" if p_corr < 0.1 else "n.s."))
            print(f"{comp:<10} | {p_corr:.4f}       | {sign}")

if __name__ == "__main__":
    analyze_quantitative_metrics(csv_text)