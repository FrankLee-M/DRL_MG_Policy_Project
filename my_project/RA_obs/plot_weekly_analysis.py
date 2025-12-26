import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ================== 1. 数据加载函数 (复用之前的逻辑) ==================

def load_and_clean_pjm(file_path):
    """
    读取 PJM (电价) 数据
    支持格式: Date, total_lmp_rt, pnode_name
    """
    print(f"Loading PJM data from: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # 1. 解析时间
    if 'Date' in df.columns:
        df['time'] = pd.to_datetime(df['Date'])
    else:
        # 兼容旧格式
        time_col = df.columns[0]
        df['time'] = pd.to_datetime(df[time_col])
    
    # 2. 处理价格列
    price_col = 'total_lmp_rt'
    if price_col not in df.columns:
        # 尝试找旧格式 RECO 或第二列
        cols = [c for c in df.columns if 'RECO' in c or 'lmp' in c.lower()]
        price_col = cols[0] if cols else df.columns[1]
    
    # 转为数值，处理可能的 '$' 符号
    df['lmhv'] = df[price_col].astype(str).str.replace('$', '', regex=False)
    df['lmhv'] = pd.to_numeric(df['lmhv'], errors='coerce')
    
    df = df.set_index('time').sort_index()
    return df[['lmhv']]

def load_and_clean_pv(file_path):
    """
    读取 PV (光伏) 数据
    支持格式: HOURBEGINNING_TIME, electricity
    """
    print(f"Loading PV data from: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # 1. 确定数值列
    val_col = 'electricity'
    if val_col not in df.columns:
        val_col = df.columns[1]
        
    df['res'] = pd.to_numeric(df[val_col], errors='coerce')
    
    # 2. 解析时间 (处理 ISO 格式)
    time_col = 'HOURBEGINNING_TIME' if 'HOURBEGINNING_TIME' in df.columns else df.columns[0]
    df['time'] = pd.to_datetime(df[time_col], utc=True)
    df['time'] = df['time'].dt.tz_localize(None) # 去除时区以便对齐
    
    df = df.set_index('time').sort_index()
    return df[['res']]

# ================== 2. 核心绘图逻辑 ==================

def plot_weekly_distributions_456():
    # --- 配置路径 (请修改此处) ---
    file_pjm = 'my_project/RA_obs/PJM-HourlyRealTime.csv'  # 替换为你的 PJM 文件路径
    file_pv = 'my_project/RA_obs/ninja_pv.csv'      # 替换为你的 PV 文件路径
    # ---------------------------

    # 1. 读取并合并数据
    try:
        df_lmhv = load_and_clean_pjm(file_pjm)
        df_res = load_and_clean_pv(file_pv)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("Merging data...")
    df_merged = df_lmhv.join(df_res, how='inner')
    
    if df_merged.empty:
        print("Error: Merged dataset is empty. Please check time ranges.")
        return

    # 2. 筛选 4, 5, 6 月
    target_months = [4, 5, 6]
    df_plot = df_merged[df_merged.index.month.isin(target_months)].copy() # type: ignore

    if df_plot.empty:
        print(f"Warning: No data found for months {target_months}.")
        return

    # 3. 特征工程：计算“月内周数” (Week of Month)
    # 逻辑：(Day - 1) // 7 + 1 -> 第1周, 第2周...
    df_plot['month'] = df_plot.index.month
    df_plot['day'] = df_plot.index.day
    df_plot['week_of_month'] = (df_plot['day'] - 1) // 7 + 1
    
    # 为了绘图好看，把 week 转为字符串类别
    df_plot['week_label'] = "Week " + df_plot['week_of_month'].astype(str)

    print(f"Data ready. Months: {df_plot['month'].unique()}")
    
    # 4. 绘图：创建一个 3行 x 2列 的大图
    # 行：4月, 5月, 6月
    # 列：LMHV (Price), RES (PV)
    
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # 设置颜色板
    palette = "viridis" # 或 "tab10", "Set2"

    for i, m in enumerate(target_months):
        # 筛选当前月份数据
        df_m = df_plot[df_plot['month'] == m]
        
        if df_m.empty:
            continue
            
        # --- 左列：LMHV (Price) ---
        ax_price = axes[i, 0]
        # 画分布曲线 (KDE)
        sns.kdeplot(
            data=df_m, x="lmhv", hue="week_label", 
            fill=True, common_norm=False, palette=palette, alpha=0.3, ax=ax_price
        )
        ax_price.set_title(f"Month {m}: LMHV (Price) Weekly Distribution")
        ax_price.set_xlabel("Price ($)")
        
        # --- 右列：RES (PV) ---
        ax_pv = axes[i, 1]
        # 画分布曲线 (KDE)
        sns.kdeplot(
            data=df_m, x="res", hue="week_label", 
            fill=True, common_norm=False, palette=palette, alpha=0.3, ax=ax_pv
        )
        ax_pv.set_title(f"Month {m}: RES (PV) Weekly Distribution")
        ax_pv.set_xlabel("PV Output")

    plt.tight_layout()
    plt.savefig("my_project/RA_obs/weekly_distribution_comparison_456.png")

    # 5. 补充：箱线图视图 (更直观对比中位数和极值)
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 15))
    
    for i, m in enumerate(target_months):
        df_m = df_plot[df_plot['month'] == m]
        if df_m.empty: continue
        
        # LMHV Boxplot
        sns.boxplot(data=df_m, x="week_label", y="lmhv", palette=palette, ax=axes2[i, 0])
        axes2[i, 0].set_title(f"Month {m}: LMHV Price Statistics")
        
        # RES Boxplot
        sns.boxplot(data=df_m, x="week_label", y="res", palette=palette, ax=axes2[i, 1])
        axes2[i, 1].set_title(f"Month {m}: PV Output Statistics")

    plt.tight_layout()
    plt.savefig("my_project/RA_obs/weekly_boxplot_comparison_456.png")

if __name__ == "__main__":
    plot_weekly_distributions_456()