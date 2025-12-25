# 实验记录：MGv2.2 CVaR 约束下的 Cost Limit 测试

**日期**: 2025-12-24
**分支**: feat/cvar-constraint (Commit Hash: <在这里填入当时的commit hash>)


## 1. 实验目标
在 MGv2.2 环境下，引入 surrogate 模型，设定多个 cost-limit 值，观察 RL Agent 在满足约束和最大化收益之间的平衡，以及 累计折扣成本的分布情况

## 2. 实验环境与配置
- **环境文件**: `MG_v2_2_Lp_surrogate/envs/surrogate/env_mg_surrogate_v2_2.py`
- **运行脚本**: `MG_v2_2_Lp_surrogate_CVaR/train/run_sac_training_env_surrogate_cvar.py`
- **Tensorboard Logs**: `tensorboard_logs/MG_v2.2_LpCVaR/sac_training_env_surrogate_cvar/2025_12_24_xxx` (本地路径，未上传)


## 3. 实验1 ：观察 drop-rate 对 cvar 低估问题的改善

#### 测试3: fixed_state-all-step cost distritbution :  观察约束 any_t : cvar(\fai_t|s_t,a_t) <= delta 满足情况：
test_set = False ： 训练集估计情况 （测试集存在分布偏移问题，后续解决这个问题）
timestamp|drop-rate 
2025_12_24_184139 | 0.05
2025_12_24_222400 | 0.1
2025_12_24_222459 | 0.2
cvar 低估问题逐渐改善，但仍然存在，见[fig](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/2025_12_24_222459/data/verification_test_step_all_test/cvar_tracking_comparison_all_runs_test_False.png)



# 下一步
1. 增大尾部分位点数量：num_quantiles=64（保持 n_critics=2 不变）
3. 分析 分布偏移的原因
4. 假设 服从高斯分布？但是实际仿真发现不符合，cost 分布是 双峰/多峰的 




