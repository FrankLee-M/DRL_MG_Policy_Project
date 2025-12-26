import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# 1. 读取数据
# 【重要】请务必将文件名替换为您真实的 csv 文件路径
file_path = 'my_project/RA_obs/data_download/DE_pv.csv' 
df = pd.read_csv(file_path)

# 2. 数据预处理
# 将时间列转换为 datetime 对象 (pandas 会自动处理 +00:00 时区)
df['dt'] = pd.to_datetime(df['HOURBEGINNING_TIME'])

# 提取年份和月份
df['Year'] = df['dt'].dt.year
df['Month'] = df['dt'].dt.month

# 3. 筛选数据
# 设定需要分析的月份 (4月到9月)
target_months = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12]

# 筛选月份
df_filtered = df[df['Month'].isin(target_months)].copy()

# 4. 绘制子图 (2行3列)
fig, axes = plt.subplots(4, 3, figsize=(18, 10)) 
# 如果你想让所有图的纵坐标刻度一致以便对比大小，可以在上面加上 sharey=True
# 例如: fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes = axes.flatten() # 展平数组方便循环

for i, month in enumerate(target_months):
    ax = axes[i]
    
    # 筛选当前月份的数据
    data_month = df_filtered[df_filtered['Month'] == month]
    
    # 绘制箱线图
    # x轴: 年份, y轴: electricity
    sns.boxplot(data=data_month, x='Year', y='electricity', ax=ax, palette="Set2")
    
    # 设置标题和标签
    month_name = calendar.month_name[month]
    ax.set_title(f'{month_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Electricity')
        
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.suptitle('Monthly Electricity Distribution by Year', y=1.02, fontsize=16)

# 保存图片
plt.savefig('my_project/RA_obs/DE_res_distribution_from_2021_2024.png', bbox_inches='tight')

# 显示图片
plt.show()