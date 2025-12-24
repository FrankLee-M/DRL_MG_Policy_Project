

# Ninja 输出是单位化的（kW per kWp）: 每千瓦装机容量下的发电量系数，不包含实际规模信息
# 新能源装机 = Nd×用户峰值可调负荷 = 100*2.5 = 250 kW
range_res = 200

# PJM电力市场  $/MW
range_lmhv = 1.0
# start_date,end_date = "2022-01-01 00:00","2022-05-30 23:00"
data_loop_num=1

predict_length = 12  # 预测长度
historical_length = 24  # 历史长度