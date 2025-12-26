import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def load_and_clean_pjm(file_path):
    """
    读取 PJM (电价) 数据 - 适配新格式
    格式示例:
    Date,total_lmp_rt,pnode_name
    1/1/2022 0:00,18.46721,PECO
    """
    print(f"Loading PJM data from: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # 1. 解析时间
    # 新格式: "1/1/2022 0:00" 通常 pandas 可以自动推断 (Month/Day/Year)
    # 如果列名是 'Date'
    if 'Date' in df.columns:
        df['time'] = pd.to_datetime(df['Date'])
    else:
        # Fallback: 如果没有 Date 列，尝试使用第一列
        print("Warning: 'Date' column not found, using the first column as time.")
        df['time'] = pd.to_datetime(df.iloc[:, 0])
    
    # 2. 处理价格列
    # 新列名: total_lmp_rt
    # 数据已经是浮点数，不需要去除 '$'
    price_col = 'total_lmp_rt'
    if price_col not in df.columns:
        # Fallback: 如果找不到该列，尝试使用第二列
        print(f"Warning: '{price_col}' not found, using the second column as price.")
        price_col = df.columns[1]
        
    # 确保转换为数值型 (float)
    df['lmhv'] = pd.to_numeric(df[price_col], errors='coerce')
    
    # 3. 设为索引并排序
    # 这一步是为了后续能和 PV 数据对齐
    df = df.set_index('time').sort_index()
    
    # 只返回需要的 'lmhv' 列
    return df[['lmhv']]


def load_and_clean_pv(file_path):
    """
    读取 PV (光伏) 数据
    格式示例:
    HOURBEGINNING_TIME,electricity
    2020-01-01 01:00:00+00:00,0
    """
    print(f"Loading PV data from: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # 1. 确定数值列
    val_col = 'electricity'
    if val_col not in df.columns:
        val_col = df.columns[1]
        
    df['res'] = df[val_col].astype(float)
    
    # 2. 解析时间 (ISO 格式)
    # utc=True 统一时区处理，避免报错
    df['time'] = pd.to_datetime(df['HOURBEGINNING_TIME'], utc=True)
    
    # 如果 PJM 数据没有时区信息，这里可能需要移除时区以便对其
    # 这里假设我们全部转为无时区 (tz_localize(None)) 来对齐
    df['time'] = df['time'].dt.tz_localize(None)
    
    # 3. 设为索引
    df = df.set_index('time').sort_index()
    return df[['res']]

def plot_monthly_distributions():
    # ================= 配置路径 =================
    # 请在这里修改为你的实际文件路径
    file_pjm = 'my_project/RA_obs/PJM_PECO.csv'  # 比如 'PJM_data.csv'
    file_pv = 'my_project/RA_obs/DE_pv.csv'    # 比如 'DE_pv.csv'
    # ===========================================

    # 1. 读取数据
    try:
        df_lmhv = load_and_clean_pjm(file_pjm)
        df_res = load_and_clean_pv(file_pv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("请确保在代码中 'file_pjm' 和 'file_pv' 变量处填写了正确的文件路径。")
        return

    # 2. 合并数据 (Inner Join 确保时间对齐)
    # 这样可以保证我们画的是同一时间段的数据
    print("Merging data...")
    df_merged = df_lmhv.join(df_res, how='inner')
    
    if df_merged.empty:
        print("Error: Merged dataset is empty. Check if dates overlap in your CSV files.")
        return

    # 3. 筛选时间范围: 2023/2024 年的 5-9 月
    target_years = [2023, 2024]
    target_months = [5, 6, 7, 8, 9]

    # 提取年和月
    df_merged['year'] = df_merged.index.year
    df_merged['month'] = df_merged.index.month

    # 筛选
    df_plot = df_merged[
        (df_merged['year'].isin(target_years)) & 
        (df_merged['month'].isin(target_months))
    ].copy()

    if df_plot.empty:
        print(f"Warning: No data found for years {target_years} in months {target_months}.")
        return

    print(f"Data ready for plotting. Total records: {len(df_plot)}")
    print(df_plot.head())

    # 4. 绘图配置 (保持你的参考风格)
    sns.set(style="whitegrid")
    
    # 为了让箱线图看起来不乱，我们可以把 'month' 变成 Categorical 类型
    df_plot['month'] = df_plot['month'].astype(str) # 或者保持 int
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # ================= LMHV (电价) =================
    # 直方图 + KDE
    sns.histplot(
        data=df_plot, x="lmhv", hue="month", 
        element="step", stat="density", common_norm=False, kde=True, 
        palette="tab10", ax=axes[0, 0]
    )
    axes[0, 0].set_title(f"LMHV (Price) Distribution ({target_years}) May-Sep")
    axes[0, 0].set_xlabel("Price Value ($)")
    
    # 箱线图
    sns.boxplot(
        data=df_plot, x="month", y="lmhv", 
        hue="year", # 这里加个 hue="year" 可以对比 2023 vs 2024，如果不需要可去掉
        palette="tab10", ax=axes[0, 1]
    )
    axes[0, 1].set_title("LMHV (Price) Boxplot Statistics")
    
    # ================= RES (光伏) =================
    # 直方图 + KDE
    sns.histplot(
        data=df_plot, x="res", hue="month", 
        element="step", stat="density", common_norm=False, kde=True, 
        palette="tab10", ax=axes[1, 0]
    )
    axes[1, 0].set_title(f"RES (PV) Distribution ({target_years}) May-Sep")
    axes[1, 0].set_xlabel("PV Output")
    
    # 箱线图
    sns.boxplot(
        data=df_plot, x="month", y="res", 
        hue="year", # 对比年份
        palette="tab10", ax=axes[1, 1]
    )
    axes[1, 1].set_title("RES (PV) Boxplot Statistics")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 模拟数据生成（如果你想直接测试代码，可以取消下面注释生成假 CSV）
    # create_dummy_csv() 
    
    plot_monthly_distributions()