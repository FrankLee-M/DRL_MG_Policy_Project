import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import os

# 导入你提供的类和配置
# 假设你的数据处理代码文件名为 data_process.py
try:
    from data_process import DataProcess, get_file_path, range_res, range_lmhv
except ImportError:
    # 如果缺少配置文件，这里提供默认值的 fallback，或者请确保在相应环境中运行
    print("Warning: Could not import config. Using default scale factors 1.0")
    range_res = 1.0
    range_lmhv = 1.0
    # 此时需要你确保 DataProcess 类定义在当前脚本可见，或者手动合并代码
    from data_process import DataProcess, get_file_path


def plot_monthly_distributions():
    # 1. 初始化
    dp = DataProcess(range_res=range_res, range_lmhv=range_lmhv)
    csv_files = get_file_path()
    
    # 确认文件顺序，通常 PJM 是 lambda_hv, ninja_pv 是 res
    # 根据你的代码: filename = csv_files[0] -> get_PJM_data; csv_files[1] -> get_PV_data
    # 建议根据文件名通过 key word 自动判断，比较稳健
    file_lmhv = next((f for f in csv_files if "PJM" in f), csv_files[0])
    file_res = next((f for f in csv_files if "ninja" in f or "pv" in f), csv_files[1])

    print(f"Loading LMHV from: {os.path.basename(file_lmhv)}")
    print(f"Loading RES from: {os.path.basename(file_res)}")

    # 2. 定义时间范围
    months = [4,5, 6, 7, 8, 9,10]
    year = 2022
    
    data_storage = {
        'month': [],
        'lmhv': [],
        'res': []
    }

    # 3. 循环读取每月数据
    for m in months:
        # 获取该月最后一天
        last_day = calendar.monthrange(year, m)[1]
        start_dt = f"{year}-{m:02d}-01 00:00"
        end_dt = f"{year}-{m:02d}-{last_day} 23:00"
        
        # --- 获取 LMHV (电价) ---
        df_lmhv = dp.get_period_data(file_lmhv, start_dt, end_dt)
        if df_lmhv.empty:
            print(f"Warning: No LMHV data for {year}-{m}")
            continue
        # 调用你的处理函数
        lmhv_vals = dp.get_PJM_data(df_lmhv.copy()) 
        
        # --- 获取 RES (光伏) ---
        df_res = dp.get_period_data(file_res, start_dt, end_dt)
        if df_res.empty:
            print(f"Warning: No RES data for {year}-{m}")
            continue
        res_vals = dp.get_PV_data(df_res.copy())

        # 存入列表用于绘图
        # 注意：这里我们假设该月 res 和 lmhv 长度一致，如果不一致需要截断对齐
        min_len = min(len(lmhv_vals), len(res_vals))
        
        data_storage['month'].extend([m] * min_len)
        data_storage['lmhv'].extend(lmhv_vals[:min_len])
        data_storage['res'].extend(res_vals[:min_len])

    # 转换为 DataFrame 方便 Seaborn 绘图
    df_plot = pd.DataFrame(data_storage)

    # 4. 绘图配置
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # --- 绘制 LMHV (电价) ---
    # 直方图 + KDE 叠加
    sns.histplot(data=df_plot, x="lmhv", hue="month", element="step", stat="density", common_norm=False, kde=True, palette="tab10", ax=axes[0, 0])
    axes[0, 0].set_title("LMHV (Price) Distribution Overlap (May-Sep)")
    axes[0, 0].set_xlabel("Price Value")
    
    # 箱线图 (查看极值和统计范围)
    sns.boxplot(data=df_plot, x="month", y="lmhv", palette="tab10", ax=axes[0, 1])
    axes[0, 1].set_title("LMHV (Price) Boxplot Statistics")
    
    # --- 绘制 RES (光伏) ---
    # 直方图 + KDE 叠加
    sns.histplot(data=df_plot, x="res", hue="month", element="step", stat="density", common_norm=False, kde=True, palette="tab10", ax=axes[1, 0])
    axes[1, 0].set_title("RES (PV) Distribution Overlap (May-Sep)")
    axes[1, 0].set_xlabel("PV Output")
    
    # 箱线图
    sns.boxplot(data=df_plot, x="month", y="res", palette="tab10", ax=axes[1, 1])
    axes[1, 1].set_title("RES (PV) Boxplot Statistics")

    plt.tight_layout()
    plt.savefig(f"my_project/RA_obs/monthly_distribution_{year}.png", dpi=300)

if __name__ == "__main__":
    plot_monthly_distributions()