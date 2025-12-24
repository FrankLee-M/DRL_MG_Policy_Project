import numpy as np
import matplotlib.pyplot as plt

# User model Parameter

########################################
print("mediunm-scale demand reponse - Adjusted for 2.5~10kW Load","\n")
########################################

T_slot = 24
Nd, N_medium, N_large = 35, 10, 5
Nd = Nd + N_medium + N_large


# NOTE -  创建局部随机数生成器，不影响全局随机状态
param_rng = np.random.default_rng(42)

# 目标：当 price=1时 d=9.5; 当 price=15时 d=2.5
# 解析解公式: d* = (ub - price) / (-2*ua)
ua_base = -1.0   
ub_base = 20.0   
uc_base = 0.0

dl_base = 2.5   
ul_base = 10.0   

ua_noise_std = 0.2  # 在 -0.8 ~ -1.2 之间波动，改变对价格的敏感度斜率
ub_noise_std = 3.0  # 在 17 ~ 23 之间波动，改变用户的基本用电需求量
uc_noise_std = 0.2
dl_noise_std = 0.3  # 不同用户的最小负荷不同
ul_noise_std = 1.0  # 不同用户的最大负荷不同

# 在基础值上加入白噪声，生成各用户参数 - 使用局部生成器
ua = np.round(ua_base + param_rng.normal(0, ua_noise_std, Nd), 2)
ub = np.round(ub_base + param_rng.normal(0, ub_noise_std, Nd), 2)
uc = np.round(uc_base + param_rng.normal(0, uc_noise_std, Nd), 2)
dl = np.round(dl_base + param_rng.normal(0, dl_noise_std, Nd), 2)
ul = np.round(ul_base + param_rng.normal(0, ul_noise_std, Nd), 1)

# 修正：
ua = np.minimum(ua, -0.5) 
dl = np.maximum(2.0, dl)
saturation_point = np.round(-ub / (2 * ua), 1)
ul = np.minimum(saturation_point, ul)

# 再次确保 ul > dl
ul = np.maximum(dl + 1.0, ul)

ul_i_t = np.repeat(ul, T_slot).reshape(Nd*T_slot,) 

# 负荷随时间的波动模式 (余弦函数模拟昼夜变化)
time = np.arange(T_slot)
# 保持原有的波动逻辑，这会让 bounds 在一天内动态变化
daily_variation_ul = 1 + 0.15 * np.cos(4 * np.pi * (time - 20) / 24) + 0.07 * np.cos(2 * np.pi * (time - 20) / 24)
daily_variation_dl = 1 + 0.3 * np.cos(4 * np.pi * (time - 20) / 24) + 0.15 * np.cos(2 * np.pi * (time - 20) / 24)

# 计算时间变化后的上下界 - 使用局部生成器
ul_dynamic = np.tile(0.9 * ul, (T_slot, 1)).T * daily_variation_ul + param_rng.normal(0, 0.2, (Nd, T_slot)) 
dl_dynamic = np.tile(dl, (T_slot, 1)).T * daily_variation_dl + param_rng.normal(0, 0.1, (Nd, T_slot)) 

print("min-load",min(np.sum(dl_dynamic,axis=0)),"max-load",max(np.sum(ul_dynamic,axis=0)))

#NOTE - plot user dynamics
# for t in range(Nd):
#     plt.plot(range(24),ul_dynamic[t,:])
#     plt.plot(range(24),dl_dynamic[t,:])
# plt.show()

# 重新排列成 Nd*T_slot 长度的一维数组
ul_i_t = np.minimum(ul_i_t, ul_dynamic.flatten())
dl_i_t = dl_dynamic.flatten()

# 最后的防错检查，防止动态波动导致 dl > ul
dl_i_t = np.minimum(dl_i_t, ul_i_t - 0.5)

di_last = (dl+ul)/2.0
delta_di = (ul-dl)*2.0
price_senstivity = np.ones(Nd, dtype=np.float32) 

# beta_price = 1.0
#⚠️ 暂时先用fixed-one init - 使用局部生成器
last_sat_i = 0.6*np.ones(Nd)+ param_rng.normal(0, 0.1, Nd)
last_ave_sat_level = np.mean(last_sat_i)  # 平均满意度水平 (输入到state)
indices = np.arange(Nd) * T_slot 
last_agg_load = 0.6*np.sum(ul_i_t[indices])

# # --- 验证绘图 (调试用) ---
# # 绘制第一个用户在不同电价下的理论最优负荷，看是否在 bounds 之内
# test_prices = np.linspace(1, 15, 100)
# user_idx = 0
# opt_d = (ub[user_idx] - test_prices) / (-2 * ua[user_idx])
# print(f"User {user_idx} Params: ua={ua[user_idx]}, ub={ub[user_idx]}, dl_base={dl[user_idx]}, ul_base={ul[user_idx]}")
# print(f"User {user_idx} Opt Load at Price 1.0: {opt_d[0]:.2f}")
# print(f"User {user_idx} Opt Load at Price 15.0: {opt_d[-1]:.2f}")

# # Uncomment below to visualize
# plt.figure()
# plt.plot(test_prices, opt_d, label='Optimal Unconstrained')
# plt.axhline(y=ul[user_idx], color='r', linestyle='--', label='Upper Bound (Base)')
# plt.axhline(y=dl[user_idx], color='g', linestyle='--', label='Lower Bound (Base)')
# plt.xlabel('Price')
# plt.ylabel('Load')
# plt.legend()
# plt.title('Check Interior Solution')
# plt.show()