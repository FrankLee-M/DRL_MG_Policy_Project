# ⚠️
1. 需要修改没有证据，不改代码：问题必须能被复现、观测或量化；感觉、猜测和“可能有问题”不足以触发修改。
2. 先证明“必须改代码”：优先排除配置、数据、使用方式和上游输入问题；代码不是默认责任方。
3. 先写清“改完会多什么”：每次修改必须对应一个可验证的结果（正确性、性能、稳定性、能力），否则停止。
4. 允许并鼓励中途止损：一旦发现前提不成立或问题不在此处，立即停止；“不改”是有效结论。
5. 最小化修改路径：能加校验不重构，能加日志不改逻辑，能局部改不全局动。
6. 禁止无关顺手改：任何与当前目标无直接关系的改动，一律推迟或拒绝。

# 还未修改
1. 严格将一年的数据划分为训练集+验证集+测试集，但是 用户侧模型拟合阶段是否需要？/ ⚠️ 需要。修改！
2. 如果引入dynamic-psi 上述结果是否依然有效？
3. 把 lambda 更新 改为 ReLu（）版本，不做奖励


# MG v2.2 – Surrogate Model + CVaR 约束 
cost-critic ：double-net  target-net 有点问题？

## 使用 env_surrogate v2_2 的实验结果
#### 测试1: 在测试集上，比较 RL/rule-based/random strategy的结果
=== Final Results cost_0p5_seed_100
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 3.47         | 0.56           
Rule Baseline   | 3.48         | 1.42           
Random          | 2.46         | 0.55           
--- Starting Evaluation ---

=== Final Results cost_0p8_seed_100
Strategy        | Avg Reward   | Avg Complaints 
----------------------------------------------
RL Agent        | 3.95         | 1.23           
Rule Baseline   | 3.48         | 1.42           
Random          | 2.46         | 0.55    


# 接下来：
1. 严格划分训练集、验证集、测试集
2. 修改 surrogate 模型
3. cost-critic ：double-net  target-net 有点问题？
4. lam ReLu 更新
3. 分析 分布偏移的原因
4. 实在不行，就直接假设 服从高斯分布