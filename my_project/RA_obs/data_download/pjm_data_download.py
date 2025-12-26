import gridstatus
import pandas as pd
import time
import os

#FIXME - 无法获取 PJM - API ，不可用❌  
iso = gridstatus.PJM()
start_date = "2021-01-01"
end_date = "2024-12-31"
filename = "pjm_hourly_lmp_filtered_3years.csv"

# 你想要关注的节点名称 (通常是 WESTERN HUB 或 PJM-RTO)
# 注意：PJM API 中 Western Hub 的名称通常是 "WESTERN HUB"
TARGET_NODES = ["WESTERN HUB", "PJM-RTO", "DOMINION"] 

date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print(f"开始下载并筛选 {start_date} 到 {end_date} 的数据...")

for current_date in date_range:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"处理中: {date_str} ...", end=" ")
    
    try:
        # 1. 下载当天的全部数据 (内存中暂存)
        df = iso.get_lmp(date=date_str, market="REAL_TIME_HOURLY")
        
        # 2. 【关键步骤】立刻筛选，只保留感兴趣的节点
        # 使用 isin() 函数筛选 Location 列
        filtered_df = df[df['Location'].isin(TARGET_NODES)]
        
        # 如果当天没有这几个节点的数据（极少见），则跳过
        if filtered_df.empty:
            print("未找到目标节点，跳过")
            continue

        # 3. 追加保存到 CSV
        if not os.path.exists(filename):
            filtered_df.to_csv(filename, index=False, mode='w')
        else:
            filtered_df.to_csv(filename, index=False, mode='a', header=False)
            
        print(f"成功 (保留 {len(filtered_df)} 行)")
        
        time.sleep(0.5) # 稍微快一点，因为处理量小了
        
    except Exception as e:
        print(f"错误: {e}")

print(f"\n下载完成！精简后的文件已保存为: {filename}")