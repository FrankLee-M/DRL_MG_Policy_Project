import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# 1. 读取数据 (请替换为您的文件名)
df = pd.read_csv('my_project/RA_obs/PJM-HourlyRealTime.csv')

# 2. 数据预处理
# 将时间列转换为 datetime 对象
df['dt'] = pd.to_datetime(df['HOURBEGINNING_TIME'])

# 清理价格列：去掉 '$' 符号并转为浮点数
df['price'] = df['RECO'].astype(str).str.replace('$', '').astype(float)

# 提取年份和月份
df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month

# 3. 筛选数据
# 筛选年份 2021-2024 和 月份 4-9
target_years = [2021, 2022, 2023, 2024]
target_months = [1,2,3,4, 5, 6, 7, 8, 9,10,11,12]

df_filtered = df[df['Year'].isin(target_years) & df['Month'].isin(target_months)].copy()

# 4. 绘制子图
# 设置画布大小和子图布局 (2行3列)
fig, axes = plt.subplots(4, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten() # 将二维数组展平，方便循环

for i, month in enumerate(target_months):
    ax = axes[i]
    
    # 筛选当前月份的数据
    data_month = df_filtered[df_filtered['Month'] == month]
    
    # 绘制箱线图
    sns.boxplot(data=data_month, x='Year', y='price', ax=ax, palette="Set2")
    
    # 设置标题和标签
    month_name = calendar.month_name[month] # 获取月份英文名
    ax.set_title(f'{month_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    
    # 只在第一列显示 Y 轴标签，避免杂乱
    if i % 3 == 0:
        ax.set_ylabel('Price ($)')
    else:
        ax.set_ylabel('')
        
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.suptitle('Monthly Price Distribution by Year (2021-2024)', y=1.02, fontsize=16)

plt.savefig('my_project/RA_obs/PJM_Monthly_Price_Distribution_2021_2024.png', dpi=300, bbox_inches='tight')