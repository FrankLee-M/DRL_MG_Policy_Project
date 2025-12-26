

**日期**: 2025-12-26
**分支**: fix/env-setup

1. Fix: train/test 数据划分
2. Fix：obs：week & month -index 置0 
3. Fix: config-env 中 unit-cost-base

# 修正以下内容：（main）
    - week-level 完全不需要 week & month -index ；可以选择隐去这部分 obs 信息
    - 解决 historical & prediction 数据 划分的问题
        历史数据 (History): t (当前) + t-1 (上一刻) + t-24 (昨日此刻)
        预测数据 (Forecast): 未来 24 小时 (全量) / 要覆盖 剩余的 Episode 长度（即 24 - t）
    - 重新划分 训练集与测试集 进行训练 //  不需要对训练集最后一天 “正常的” 预测数据进行修正
    - ⚠️ 最终版需要用真实的 预测模型，替代 真实值+噪声注入 的版本
    - 修正 config-env 中 unit-cost 缩放的问题。应该直接在 config-para 文件修改？
