
- **分支名**: archive/failed-ood-obs (见下文归档操作)
- **修改内容**: 将 obs 的 week/month index 全部置 0
- **实验结果**: 无效。
- **现象**: 测试阶段的 cvar 估计仍然大幅偏离实际情况

## 5 实验3: 训练阶段隐去 week/month index 
分别在 env_surrogate & env_true 进行实验，排除 surrogate 模型对结果的影响
2025_12_25_155623 ｜ surrogate 
2025_12_25_162103 ｜ true
没有任何改善，具体内容在 exp/fix-ood-obs 

⚠️ 这个版本运行时，存在bug：
surrogate 模型采用 先前生成的版本，即，未对 obs 置0 的版本；
但是不影响实验的最终结果，因为用 true env 仿真的结果仍然具有较大的 OOD 

⚠️ 重新保存了surrogate 模型，但是用的 conda - base，可能存在 sklearn 不兼容的问题，重新用 rlenv 环境 拟合模型即可
