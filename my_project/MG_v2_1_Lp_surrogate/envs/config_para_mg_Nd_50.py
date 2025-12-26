
import numpy as np
import matplotlib.pyplot as plt

# <负荷/光伏/发电/储能 比例>
# alpha_ave = 0.4
# alpha_pv,H_ave,eta_pv = 0.7,4.5,0.85
# D_peak= 500
# D_valley = 80
# D_ave = alpha_ave* D_peak
# E_day = D_ave*24
# E_pv = E_day*alpha_pv
# P_pv = E_pv/H_ave/eta_pv
# print("Required PV:", P_pv)
# alpha_pg = 1.0 # 0.2~1.0 取决于支撑程度
# P_pg = alpha_pg*D_peak
# print("Required PG",P_pg)
# alpha_ba = 5.0 # 2~8 hour *Peak
# E_ba = alpha_ba * D_peak
# P_ba = 0.5 * D_peak
# print("Required Battery Energy & Power" , E_ba, P_ba)

# data prediction 
# Nd = 10 负荷 范围 2.0-25
# res 范围：  40

def get_complaint_unit_cost(base_cost=5.0):
    """
    返回一个 24 小时的投诉单位成本向量 ($/次)。
    逻辑：
    - 深夜 (0-5): 敏感度低，成本低 (用户在睡觉)。
    - 早高峰 (6-9): 敏感度上升 (起床、做饭)。
    - 白天 (10-16): 中等 (上班/上学不在家，或工商业刚需)。
    - 晚高峰 (17-21): 敏感度最高 (全家团聚、娱乐、做饭)，投诉代价最大。
    - 晚间 (22-23): 逐渐回落。
    """
    # 基础值
    # base_cost = 50.0 
    
    # 24小时系数曲线
    hourly_factors = np.array([
        0.5, 0.5, 0.5, 0.5, 0.6, 0.8,  # 00-05: 深夜 (2.5-4.0元)
        1.2, 1.5, 1.2, 1.0, 1.0, 1.0,  # 06-11: 早高峰 (6.0-7.5元)
        1.0, 1.0, 1.0, 1.0, 1.2, 1.5,  # 12-17: 下午 (5.0-7.5元)
        2.0, 2.5, 2.5, 2.0, 1.5, 1.0   # 18-23: 晚高峰 (10.0-12.5元) !!! 重点惩罚区域
    ])
    
    unit_costs = base_cost * hourly_factors
    return unit_costs

#ANCHOR - base- cost= 5.0 在训练网络时进行缩放
unit_costs = get_complaint_unit_cost(5.0)
complaint_cost_scale = 0.001
# env_config 
# 5月前3周
begin_t = 2880
end_t = 3383

# One-week : 7.7 - 7.15  
# vali_begin_t = 4412
# vali_end_t = 4579

# 5月剩余日子
test_begin_t = 3384  
test_end_t = 3600 

# day_summer = 5184
# day_winter = 360



range_res = 4.0  # basic: 150
range_lmhv = 0.1 


T_slot = 24
historical_length = T_slot
predict_length = int(T_slot/2)


res_penalty = 10
on_grid_price = 0.5
buy_res_price = 1.0

        
# start_date,end_date = "2022-01-01 00:00","2022-05-30 23:00"
data_loop_num=1




lp_max,lp_min =15.0, 3.0
lmhv_kmin,lmhv_kmax = 0.5,2.0

# dynamic price parameter
lp_last = 5.0
delta_lp = 3.0

#  agent parameter 
T_dec = 1  # each step decides 1 time slot decision

# One day/weeek/month/yeat data 
episode_length = 24
# fixed one-day length 


action_dim = 1*T_dec  # lambda_p

obs_res_dim = 36
obs_price_dim = 36
obs_price_rq_dim = 1
obs_dynamic_bound_dim = action_dim *2  # min/max/action_last
obs_user_dim = 2 # [ave_t + Dt-1_agg]

obs_soc_dim = 1
obs_time_dim = 6


# state_dim = obs_res_dim + obs_price_dim + obs_price_rq_dim + action_dim + obs_dynamic_bound_dim + obs_user_dim + obs_soc_dim + obs_time_dim  
# 36*2 + (1+1) + 1 + 6 # res(24+12), lmhv(24+12), [price-roll-quantile],[last_action],[dynamic-bounds(action_dim*2)] ,[ave_t + Dt-1_agg] +  soc + sin/cos(hour,week,month)*2可能还有别的...
# obs_soc_index = - int(obs_soc_dim+obs_time_dim )
# obs_res_index = historical_length
# obs_lmhv_index = 2*historical_length + predict_length

# action_bound_index = -(obs_user_dim + obs_soc_dim + obs_time_dim)
# action_bound_indices = np.arange(action_bound_index-obs_dynamic_bound_dim ,action_bound_index)
# action_last_indices = np.arange(action_bound_indices[0]-action_dim, action_bound_indices[0])

#NOTE -  --- 2. 正向索引计算 (Cumulative Indexing) ---
# 初始化指针，指向 State 向量的开头
current_idx = 0

# [1] Res (资源)
obs_res_start_index = current_idx
current_idx += obs_res_dim

# [2] Price (价格)
obs_price_start_index = current_idx
current_idx += obs_price_dim

# [3] Price Roll Quantile
obs_price_rq_start_index = current_idx
current_idx += obs_price_rq_dim

# [4] Action Last (上一时刻动作)
# 计算索引范围
action_last_indices = np.arange(current_idx, current_idx + action_dim)
current_idx += action_dim

# [5] Dynamic Bounds (动态边界)
# 计算索引范围
action_bound_indices = np.arange(current_idx, current_idx + obs_dynamic_bound_dim)
current_idx += obs_dynamic_bound_dim

# [6] User (用户特征)
obs_user_index = current_idx
current_idx += obs_user_dim

# [7] SoC (电池状态)
obs_soc_index = current_idx  # 直接获取当前位置，这就是正向索引
current_idx += obs_soc_dim

# [8] Time (时间)
obs_time_index = current_idx
current_idx += obs_time_dim

# # [9] Augment (增广状态)
# obs_augment_index = current_idx
# current_idx += augment_dim

# --- 3. 最终总维度 ---
state_dim = current_idx  # 自动计算总和，无需手动加
print(f"State Dim: {state_dim}")
print(f"SoC Index: {obs_soc_index}")

# --- 4. 兼容你的旧变量名 (Optional) ---
#
obs_res_index = obs_res_start_index + historical_length
obs_lmhv_index = obs_price_start_index + historical_length 

# #
# output record:
# print("res-range",range_res)
# print("pg cost matrix",cost_matrix)
# print("pg-capacity",pg_min,pg_max,delta_pg)
# print("Battery-params", "capacity=",capacity,"f_battery=",f_battery,"pb-max=",100)