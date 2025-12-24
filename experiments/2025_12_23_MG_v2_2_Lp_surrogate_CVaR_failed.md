# 实验记录：MGv2.2 CVaR 约束下的 Cost Limit 测试

**日期**: 2025-12-23
**分支**: feat/cvar-constraint (Commit Hash: <在这里填入当时的commit hash>)

# [已作废] 实验记录：MGv2.2 CVaR (基于 Env surrogate v1.2)
> ⚠️ **警告 (2025-12-23)**: 
> env_surrogate_v1_2.py: 模型拟合时，采用了几乎全年的数据，包括了“测试集”，存在数据泄漏风险。
> **本实验结论已失效，仅作参考。**
> 最新实验请参考：

## 1. 实验目标
在 MGv2.2 环境下，引入 surrogate 模型，设定多个 cost-limit 值，观察 RL Agent 在满足约束和最大化收益之间的平衡，以及 累计折扣成本的分布情况

## 2. 实验环境与配置
- **环境文件**: `MG_v2_1_Lp_surrogate/envs/surrogate/env_mg_surrogate_v1_2.py`
- **运行脚本**: `MG_v2_2_Lp_surrogate_CVaR/train/run_sac_training_env_surrogate_cvar.py`
- **Tensorboard Logs**: `tensorboard_logs/MG_v2.2_LpCVaR/sac_training_env_surrogate_cvar` (本地路径，未上传)
[logs](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/213328)


## 3. 实验1 ：设定多个 cost-limit 值，观察结果


#### 测试1: 在测试集上，比较 RL/rule-based/random strategy的结果
[test](../my_project/MG_v2_2_Lp_surrogate_CVaR/test/test_sac_env_surrogate_cvar.py)
结果符合实验设定，cost_limit 越低，RL-agent 的 avg-complaints 越低，但 avg-reward 也越低

    === Final Results cost_0p5_seed_100
    Strategy        | Avg Reward   | Avg Complaints 
    ----------------------------------------------
    RL Agent        | 3.81         | 0.88           
    Rule Baseline   | 3.48         | 1.42           
    Random          | 2.46         | 0.55           
    --- Starting Evaluation ---

    === Final Results cost_0p4_seed_100
    Strategy        | Avg Reward   | Avg Complaints 
    ----------------------------------------------
    RL Agent        | 3.68         | 0.67           
    Rule Baseline   | 3.48         | 1.42           
    Random          | 2.46         | 0.55           
    --- Starting Evaluation ---

    === Final Results cost_0p8_seed_100
    Strategy        | Avg Reward   | Avg Complaints 
    ----------------------------------------------
    RL Agent        | 3.99         | 1.38           
    Rule Baseline   | 3.48         | 1.42           
    Random          | 2.46         | 0.55           
    --- Starting Evaluation ---

    === Final Results cost_0p6_seed_100
    Strategy        | Avg Reward   | Avg Complaints 
    ----------------------------------------------
    RL Agent        | 3.90         | 1.10           
    Rule Baseline   | 3.48         | 1.42           
    Random          | 2.46         | 0.55           



#### 测试2: fixed_state cost distribution : 观察约束 cvar(\fai_0|s_0,a_0) <= delta 满足情况：
[test](../my_project/MG_v2_2_Lp_surrogate_CVaR/test/test_fixed_state_cost_distribution.py)
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/213328/data/verification_test_start_state)
分别在 训练集/测试集 观察 固定点： fai_0 | s_0,a_0  的 情况
训练集：cvar-hat 偏离 cvar-real， 但是呈现出 高估
测试集：cvar-hat 毫无规律的 低谷 cvar-rel ，随着 delta 的增大，整体的偏移 规律是存在的

#### 测试3: fixed_state-all-step cost distritbution :  观察约束 any_t : cvar(\fai_t|s_t,a_t) <= delta 满足情况：
[test](../my_project/MG_v2_2_Lp_surrogate_CVaR/test/test_any_state_cost_distribution.py)
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/213328/data/verification_test_step_all)
训练集：cvar 估计值可以很好跟随真实值
测试集：完全偏离=》过拟合

#### 测试4:buffer-level cost distribution ：
[test](../my_project/MG_v2_2_Lp_surrogate_CVaR/test/test_buffer_level_cvar.py)
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/213328/data/verification_buffer)
基本满足

# 下一步
1. 严格划分训练集、验证集、测试集
2. 修改 surrogate 模型
3. cost-critic ：double-net  target-net 有点问题？
4. lam ReLu 更新
3. 分析 分布偏移的原因
4. 实在不行，就直接假设 服从高斯分布？但是实际仿真发现不符合，cost 分布是 双峰/多峰的 




