# 实验记录：MGv2.2 CVaR 约束下的 Cost Limit 测试

**日期**: 2025-12-25
**分支**: feat/cvar-constraint (Commit Hash: <在这里填入当时的commit hash>)


## 1. 实验目标
在 MGv2.2 环境下，引入 surrogate 模型，设定多个 cost-limit 值，观察 RL Agent 在满足约束和最大化收益之间的平衡，以及 累计折扣成本的分布情况

## 2. 实验环境与配置
- **环境文件**: `MG_v2_2_Lp_surrogate/envs/surrogate/env_mg_surrogate_v2_2.py`
- **运行脚本**: `MG_v2_2_Lp_surrogate_CVaR/train/run_sac_training_env_surrogate_cvar.py`
- **Tensorboard Logs**: `tensorboard_logs/MG_v2.2_LpCVaR/sac_training_env_surrogate_cvar/2025_12_25_xxx` (本地路径，未上传)


## 3. 实验1 ：增大尾部分位点数量：num_quantiles=32/64,观察cvar-hat 在 训练集上的低估问题

### 测试3: fixed_state-all-step cost distritbution :  观察约束 any_t : cvar(\fai_t|s_t,a_t) <= delta 满足情况：
test_set = False ： 训练集估计情况 （测试集存在分布偏移问题，实验2继续解决这个问题）
timestamp|num_quantiles｜drop-rate = 0.0 
2025_12_25_102944 ｜ 32
2025_12_25_103009 ｜ 64 
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/2025_12_25_102944/data/verification_test_step_all_test/)
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/2025_12_25_103009/data/verification_test_step_all_test/)

去呗不大，num_quantiles 数量提升带来的 cvar 低估，没有 提升 drop-rate 带来的提升大

## 4 实验2 ：测试验证 训练集->测试集 分布偏移 的原因

### 猜测1: week/month -index 影响了actor 判断
观察 训练集 & 测试集 obs-dim 分布情况，发现 最后 4 维 obs 差异巨大，例如  dim 83/84 是month-index，训练集包含 5-6月，测试集 只有7 月  
[test](../my_project/MG_v2_2_Lp_surrogate_CVaR/test/test_vecnorm_ood.py)
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar/2025_12_24_184139/data/verification_vecnorm_ood)


#### 实验1 ：去除 vecnorm 环境包装，不再对 obs 进行归一化
[results](../tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar_without_norm/2025_12_25_114106/data/verification_test_step_all_test/)
没有提升

把obs[-6:] 全部置0 => 仍然没有改善，需要 重新训练，把 week & month 信息全部隐去
实验结果将在 exp/fix-ood-obs 分支 

#### 实验2: 训练阶段隐去 week/month index 
分别在 env_surrogate & env_true 进行实验，排除 surrogate 模型对结果的影响
2025_12_25_155623 ｜ surrogate 
2025_12_25_162103 ｜ true
没有任何改善，具体内容在 exp/fix-ood-obs 

### 猜测2: 测试集本身偏离训练集过多 
git checkout exp/fix-ood-train-test-period
#### 实验1: 选择 train /test 分布比较接近的阶段-[解决ood问题]
例如，5月 前三周 为训练集；后1周为 测试集，具体内容在 exp/fix-ood-train-test-period
tensorboard_logs/MG_v2_2_Lp_surrogate_CVaR/sac_training_env_surrogate_cvar_fixed_ood_period
train/test ： 表现基本一致，且cvar 估计值 高于 蒙特卡洛采样真值




# 下一步：（main）
    - week-level 完全不需要 week & month -index ；可以选择隐去这部分 obs 信息
    - 解决 historical & prediction 数据 划分的问题
        历史数据 (History): t (当前) + t-1 (上一刻) + t-24 (昨日此刻)
        预测数据 (Forecast): 未来 24 小时 (全量) / 要覆盖 剩余的 Episode 长度（即 24 - t）
    - 重新划分 训练集与测试集 进行训练 //  不需要对训练集最后一天 “正常的” 预测数据进行修正
    - ⚠️ 最终版需要用真实的 预测模型，替代 真实值+噪声注入 的版本
    - 修正 env 中 unit-cost 缩放的问题。应该直接在 config-para 文件修改？

## NEXT （feat/cvar-constraint）
1. 增大尾部分位点数量：num_quantiles=64（保持 n_critics=2 不变）
    N=64（保持 n_critics=2 不变）
    仍低估 → N=100
    仍低估且算力够 → n_critics=5，N=64
    仍低估 → 上 尾部加密 τ（比继续堆 N / critics 更有效）
    数据问题（尖峰事件很稀）→ 对高 cost 轨迹做 priority replay / 更多 rollout（否则再多 quantile 也学不到尖峰）
4. 假设 服从高斯分布？但是实际仿真发现不符合，cost 分布是 双峰/多峰的 




