
- **分支名**: archive/ood-train-test-period
- **修改内容**: 修改 训练集 / 测试集 的起止时间
- **实验结果**: train/test ： 表现基本一致，且cvar 估计值 高于 蒙特卡洛采样真值
- **现象**：如果测试集与训练集 分布差异不大，cvar 估计的结果还是比较贴合实际的

在 RA——obs中，撰写了部分 plot_xx_distribution 文件，用于观察 数据分布情况
plot_monnthly_distribution :比较跨年数据，在月内的分布情况
plot_weekly_distribution: 比较当前数据在 周-层面的分布情况
    my_project/RA_obs/ninja_pv.csv
    my_project/RA_obs/PJM-HourlyRealTime.csv

Plot-distribution 后发现，7月相对5/6 月的分布偏移巨大，因此，原实验中，选择用 5-6月训练，7月测试的方案，会有巨大的分布偏移问题
现在选择： 5月 前三周 为训练集；后 1周多 为测试集

#### 测试1: 在测试集上，比较 RL/rule-based/random strategy的结果
=== Final Results cost_0p8_seed_100
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 4.18         | 0.89           
Rule Baseline   | 3.71         | 0.49           
Random          | 2.54         | 0.35   

#### 测试2: fixed_state cost distribution : 观察约束 cvar(\fai_0|s_0,a_0) <= delta 满足情况：
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar_fixed_ood_period/2025_12_26_170209/data/verification_test_start_state/all_runs_distribution_test_False.png)

## 💡 测试3: fixed_state-all-step cost distritbution :  观察约束 any_t : cvar(\fai_t|s_t,a_t) <= delta 满足情况
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar_fixed_ood_period/2025_12_26_170209/data/verification_test_step_all_test)
train/test ： 表现基本一致，且cvar 估计值 高于 蒙特卡洛采样真值


#### 测试4:buffer-level cost distribution ：
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar_fixed_ood_period/2025_12_26_170209/data/verification_buffer)



# 下一步：
    - week-level 完全不需要 week & month -index ；可以选择隐去这部分 obs 信息
    - 解决 historical & prediction 数据 划分的问题
        历史数据 (History): t (当前) + t-1 (上一刻) + t-24 (昨日此刻)
        预测数据 (Forecast): 未来 24 小时 (全量) / 要覆盖 剩余的 Episode 长度（即 24 - t）
    - 重新划分 训练集与测试集 进行训练 //  不需要对训练集最后一天 “正常的” 预测数据进行修正
    - ⚠️ 最终版需要用真实的 预测模型，替代 真实值+噪声注入 的版本
    - 修正 env 中 unit-cost 缩放的问题。应该直接在 config-para 文件修改？


