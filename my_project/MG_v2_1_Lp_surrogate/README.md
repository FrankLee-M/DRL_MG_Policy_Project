# ⚠️
1. 没有证据，不改代码：问题必须能被复现、观测或量化；感觉、猜测和“可能有问题”不足以触发修改。
2. 先证明“必须改代码”：优先排除配置、数据、使用方式和上游输入问题；代码不是默认责任方。
3. 先写清“改完会多什么”：每次修改必须对应一个可验证的结果（正确性、性能、稳定性、能力），否则停止。
4. 允许并鼓励中途止损：一旦发现前提不成立或问题不在此处，立即停止；“不改”是有效结论。
5. 最小化修改路径：能加校验不重构，能加日志不改逻辑，能局部改不全局动。
6. 禁止无关顺手改：任何与当前目标无直接关系的改动，一律推迟或拒绝。

# 还未修改
1. 严格将一年的数据划分为训练集+验证集+测试集，但是 用户侧模型拟合阶段是否需要？
2. 如果引入dynamic-psi 上述结果是否依然有效？

# MG v2.1 – Surrogate Model

## Internal Code Variants

 env_mg_surrogate_v0_0.py | 动态电价 + 拟合 负荷响应 + 投诉数量|
 env_mg_surrogate_v0_1.py | 动态电价 + 拟合 负荷响应 + 投诉数量 + 概率预测 |

 env_mg_surrogate_v1_0.py | 动态电价 + 拟合 负荷响应 + 投诉成本|
 env_mg_surrogate_v1_1.py | 用户侧参数动态变化 + 动态电价 + 拟合 负荷响应 + 投诉成本|
 env_mg_surrogate_v1_2.py | (继承 v1_0)|改变 complaint-cost 的基础，缩放 0.0001
 env_mg_surrogate_v1_3.py | (继承 v1_0)｜峰谷电价|
 env_mg_surrogate_v2_0.py | (继承 v1_2): 仅用夏季训练数据做模型拟合的结果 [final-version]


## 实验 1:比较不同 surrogate 模型的测试结果

[run](../MG_v2.1_LpCVaR/train/run_sac_training_env_surrogate.py)
[test-v0](../MG_v2_1_LpCVaR/test/test_sac_env_surrogate.py) 
[test-v1/v2](../MG_v2_1_LpCVaR/test/test_sac_env_surrogate_v1.py) 
其他版本（v1_0/v1_2），RL-agent ave-reward 在 4.00 左右；模型差别不大；数据量非常充足

v1_1: is_user_dynamic_true : 至少比 rule-based 要好很多 
=== Final Results (Average over 100 episodes) ===
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 3.72         | 14012.95       
Rule Baseline   | 3.30         | 11249.95       
Random          | 2.28         | 5325.40    

v1_3：
=== Final Results (Average over 100 episodes) ===
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 3.99         | 14432.70       
Rule Baseline   | 3.48         | 14178.40       
Random          | 2.42         | 5285.35        


v2_0: 仅用夏季训练数据做模型拟合的结果
=== Final Results (Average over 100 episodes) ===
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 3.98         | 14630.10       
Rule Baseline   | 3.48         | 14178.40       
Random          | 2.42         | 5285.35            


## 实验2: 采用 seed = None 保留结果随机性，观察 reward & cost 分布情况
[run](../MG_v2.1_LpCVaR/train/run_sac_training_env_surrogate_v1.py)
[test](../MG_v2_1_LpCVaR/test/test_fixed_state_distribution.py)
[results](../../tensorboard_logs/MG_v2_1_LpCVaR/sac_training_env_surrogate_v1_0/162146/data/verification_test/verification_seed_None.png)
整体呈现 高斯分布， 因为 res / user-response 增加的噪声，都是 高斯分布噪声
reward 分布 min-max 差值在 0.3*10^4 = 3000 ； cost 分布 min-max 在 5000～7000；

绘图分析：
[analyze](../MG_v2_1_LpCVaR/tools/analyze_complaint.py)  
ep_return 有明显的上升趋势，但 episode_cost 整体波动幅度非常大，且很快持平

