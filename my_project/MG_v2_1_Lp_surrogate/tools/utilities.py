import pandas as pd
import numpy as np
import os

def format_tag(value):
    """Helper to format numbers for filenames (e.g., 2.0 -> 2p0)"""
    if value is None:
        return "None"
    return str(value).replace(".", "p").replace("-", "neg")

def calculate_cvar(data, alpha=0.9):
    """
    计算分布右尾的 CVaR (Conditional Value at Risk)。
    对于 Cost 而言，关注的是数值最大的那部分风险（Worst-case）。
    alpha=0.9 意味着计算最大的 10% 数据的平均值。
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # 计算切分点索引，例如 100 个数据，alpha=0.9，cutoff=90
    cutoff_index = int(alpha * n)
    
    # 边界保护
    if cutoff_index >= n:
        return sorted_data[-1]
    
    # 取尾部 (1-alpha) 的数据（即最大的那部分 Cost）
    tail_losses = sorted_data[cutoff_index:]
    return np.mean(tail_losses)

# reigion ：find env function
def _peel_vec_shell(maybe_vec_env):
    """
    把 VecNormalize / VecCheckNan / VecFrameStack 之类的 .venv 外壳剥掉，
    返回最内层的 VecEnv（DummyVecEnv/SubprocVecEnv）或单环境。
    """
    cur = maybe_vec_env
    seen = set()
    while hasattr(cur, "venv") and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.venv
    return cur
def get_single_env_from_training_env(training_env, index: int = 0):
    """
    从 SB3 的 self.training_env 里拿到第 index 个“单环境”（仍可能是多层 gym.Wrapper）。
    """
    core = _peel_vec_shell(training_env)
    if hasattr(core, "envs"):            # VecEnv
        return core.envs[index]
    return core                          # 非 VecEnv，直接就是单环境
def get_base_env(training_env, index: int = 0):
    """
    拿到“最初的环境”（最内层 base env，例如你的 MgSatFBEnv）。
    优先用 .unwrapped；若不可用，就沿 .env 逐层向里找。
    """
    single = get_single_env_from_training_env(training_env, index=index)

    # 1) Gym 规范：任何 Wrapper 的 .unwrapped 都应指向 base env
    base = getattr(single, "unwrapped", None)
    if base is not None:
        return base

    # 2) 兜底：手动沿 .env 向里找
    seen = set()
    cur = single
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if not hasattr(cur, "env"):
            return cur
        cur = cur.env
    return cur
# endregion

def compute_cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return np.nan
    sorted_vals = np.sort(values)
    tail_count = max(1, int(np.ceil((1 - alpha) * len(sorted_vals))))
    tail_vals = sorted_vals[-tail_count:]
    return float(np.mean(tail_vals))


def find_wrapper_anywhere(root, target_cls):
    """
    在 SB3/Gym 的各种封装层里递归寻找某个 wrapper/环境实例。
    支持属性：.env（Gym Wrapper）、.venv（VecNormalize等）、.envs（VecEnv子环境列表）。
    用法：cvar = find_wrapper_anywhere(env, CVARLagrangianWrapper)
    """
    seen = set()
    stack = [root]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        # 命中
        if isinstance(cur, target_cls):
            return cur

        # 可能的下游指针
        for attr in ("env", "venv", "wrapped_env"):  # wrapped_env 以防自定义
            if hasattr(cur, attr):
                stack.append(getattr(cur, attr))

        # VecEnv: 子环境列表
        if hasattr(cur, "envs") and isinstance(cur.envs, (list, tuple)):
            stack.extend(cur.envs)

    return None


# 训练后数据导出
# 导出系统功率相关的数据
def export_training_data(callback, save_dir):
    """导出系统功率相关数据到CSV文件"""
    
    # # 1. 导出电价数据 (lmhv) - 单独导出
    # if callback.lmhv_list and len(callback.lmhv_list) > 0:
    #     lmhv_data = pd.DataFrame(callback.lmhv_list, columns=['lmhv'])
    #     lmhv_data.to_csv(os.path.join(save_dir, "lmhv_data.csv"), index=False)
    #     print("Electricity price data exported.")
    
    # 2. 导出动作数据 (lp, pb，pg，soc) - 单独导出
    action_data = {}
    if callback.action_lps and len(callback.action_lps) > 0:
        action_data['lp'] = callback.action_lps
    
    if callback.action_pbs and len(callback.action_pbs) > 0:
        action_data['pb'] = callback.action_pbs
    
    if callback.action_pgs and len(callback.action_pgs) > 0:
        pg_array = np.array(callback.action_pgs)
        if pg_array.ndim == 2:  # 如果是二维数组
            for i in range(pg_array.shape[1]):
                action_data[f'pg_{i}'] = pg_array[:, i]
  # 添加电池SOC数据 (soc)
    if callback.soc_list and len(callback.soc_list) > 0:
        action_data['soc'] = callback.soc_list
    
    if action_data:
        action_df = pd.DataFrame(action_data)
        action_df.to_csv(os.path.join(save_dir, "action_data.csv"), index=False)
        print("Action data exported.")
    

    # 3. 导出系统功率数据 (res, pg, pb, phv, last_agg_load) - 合并到一个文件
    system_data = {}
    
    # 添加可再生能源数据 (res)
    if callback.res_list and len(callback.res_list) > 0:
        system_data['res'] = callback.res_list
    
    # 添加发电机功率数据 (pg) - 处理多维数据
    if callback.action_pgs and len(callback.action_pgs) > 0:
        pg_array = np.array(callback.action_pgs)
        if pg_array.ndim == 2:  # 如果是二维数组
            for i in range(pg_array.shape[1]):
                system_data[f'pg_{i}'] = pg_array[:, i]
        else:  # 如果是一维数组
            system_data['pg'] = callback.action_pgs
    
    # 添加电池功率数据 (pb)
    if callback.action_pbs and len(callback.action_pbs) > 0:
        system_data['pb'] = callback.action_pbs
    
    # 添加电网交换功率数据 (phv)
    if callback.phv_list and len(callback.phv_list) > 0:
        system_data['phv'] = callback.phv_list
    
    # 添加总负荷数据 (last_agg_load)
    if callback.last_agg_load_list and len(callback.last_agg_load_list) > 0:
        system_data['last_agg_load'] = callback.last_agg_load_list
    
    if system_data:
        system_df = pd.DataFrame(system_data)
        system_df.to_csv(os.path.join(save_dir, "system_data.csv"), index=False)
        # print("System power data exported to system_data.csv")
    
    # 4. 导出用户需求数据 (dit) - 单独导出
    if callback.dit_list and len(callback.dit_list) > 0:
        # dit是多维数据，需要特殊处理
        dit_array = np.array(callback.dit_list)
        if dit_array.ndim == 2:  # 如果是二维数组
            dit_columns = [f'user_demand_{i}' for i in range(dit_array.shape[1])]
            dit_df = pd.DataFrame(dit_array, columns=dit_columns)  # type: ignore
        else:  # 如果是一维数组
            dit_df = pd.DataFrame(callback.dit_list, columns=['dit']) # type: ignore
        dit_df.to_csv(os.path.join(save_dir, "demand_data.csv"), index=False)
        print("User demand data exported.")
    
    # 导出用户满意度数据
    if callback.sat_i_list and len(callback.sat_i_list) > 0:
        sat_array = np.array(callback.sat_i_list)
        # 生成列名（假设有100个用户）
        columns = [f'user_{i}' for i in range(sat_array.shape[1])]
        sat_df = pd.DataFrame(sat_array, columns=columns) # type: ignore

        sat_df.to_csv(os.path.join(save_dir, "sat_data.csv"), index=False) # type: ignore



    # # 5. 导出奖励数据 - 单独导出
    # if callback.reward_list and len(callback.reward_list) > 0:
    #     reward_data = pd.DataFrame(callback.reward_list, columns=['reward'])
    #     reward_data.to_csv(os.path.join(save_dir, "reward_data.csv"), index=False)
    #     print("Reward data exported.")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def normalize_manual(raw_obs, vec_env):
    """
    使用 VecNormalize 的统计数据手动归一化
    """
    # 1. 获取训练好的均值和方差
    # 注意：VecNormalize 可能会嵌套，需要找到最外层的那个
    # 如果你直接用了 vec_env = VecNormalize(...)，那就是它
    mean = vec_env.obs_rms.mean
    var = vec_env.obs_rms.var
    epsilon = 1e-8  # 防止除以0

    # 2. 执行归一化公式: (x - mean) / std
    norm_obs = (raw_obs - mean) / np.sqrt(var + epsilon)
    
    # 3. 截断 (Clip)，和训练时保持一致 (clip_obs=10.)
    norm_obs = np.clip(norm_obs, -10., 10.)
    
    return torch.tensor(norm_obs, dtype=torch.float32)


def plot_cost_distribution(model,env, action_dim,test_index_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 切换到评估模式

    env.train_mode = False  # 切换到测试模式
    env.reset()
    q_high_list =[]
    q_low_list =[]
    baseline_list=[]
    for t in test_index_list:
        env.index_t = t
        obs = env._get_obs()  # 更新环境状态
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)  # 增加 Batch 维度: [state_dim] -> [1, state_dim]
        a_high = torch.ones(1, action_dim).to(device) * 0.9
        a_low = torch.ones(1, action_dim).to(device) * (-0.5)
    
        with torch.no_grad():
            # 输出形状: [1, 20] -> flatten -> [20]
            # 这些值就是 Critic 预测的 Cost 分布的 20 个采样点
            q_high = model(obs_tensor, a_high).cpu().numpy().flatten()
            q_low  = model(obs_tensor, a_low).cpu().numpy().flatten()
            q_high_list.append(q_high)
            q_low_list.append(q_low)
    
    q_high_all = np.concatenate(q_high_list)
    q_low_all = np.concatenate(q_low_list)
    
    print("Expectation-High:",q_high_all.mean(),"Expectation-Low:",q_low_all.mean())

    # --- 绘图 ---
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 绘制 KDE
    sns.kdeplot(q_high_all, fill=True, color='red', label='Summer + High Price (Risky)', cut=0)
    sns.kdeplot(q_low_all,  fill=True, color='green', label='Summer + Low Price (Safe)', cut=0)
    
    # 绘制 Rug Plot (数据点分布)
    sns.rugplot(q_high_all, color='red', alpha=0.1, height=0.1) # alpha 调低一点，因为点可能很多
    sns.rugplot(q_low_all, color='green', alpha=0.1, height=0.1)

    # 辅助线
    plt.axvline(0, color='black', linestyle='--', label='Baseline (Regret=0)')
    
    plt.title(f"Predicted Cost Distribution (Test Steps: {test_index_list})")
    plt.xlabel("Relative Cost Value (Regret)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()