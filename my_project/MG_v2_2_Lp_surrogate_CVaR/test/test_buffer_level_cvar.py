import os,sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
project_version = os.path.basename(os.path.normpath(parent_dir))


from tools.utilities import get_base_env, compute_cvar
from envs.config_para_mg_Nd_50 import begin_t, end_t, test_begin_t, test_end_t
from envs.action_wrapper import ActionClipperWrapper_OffPolicy
from envs.surrogate.env_mg_surrogate_v2_0 import MgSurrogateEnv
from algorithms.customerized_sac_v2_2 import DSAC


# NOTE - 测试 cvar 估计 vs cvar 真值 (任意时刻 t)
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default="2025_12_26_170209")
    parser.add_argument("--check_time_step", type=int, default=4)
    parser.add_argument("--test_set", action="store_true")
    parser.add_argument("--exp_name", type=str, default="sac_training_env_surrogate_cvar_fixed_ood_period")
    return parser.parse_args()
args = get_args()

env_version = "v2_0"
EXP_NAME = args.exp_name
project_version = os.path.basename(os.path.normpath(parent_dir))

BASE_LOG_DIR = f"./tensorboard_logs/{project_version}"
TIMESTAMP = args.timestamp  # <--- 请修改为你实际运行生成的时间戳
K_ROLLOUTS = 200  # K >= 100 to ensure statistical significance
# --- 新增：指定要检测的时间步 t ---
CHECK_TIME_STEP = args.check_time_step  # 例如：检测第24个时间步的 Cost 分布 (t=0即为初始状态)
test_set = args.test_set  # If False, test on training set

#region
if test_set:
    test_bt = test_begin_t
    test_et = test_end_t
else:
    test_bt = begin_t
    test_et = end_t
#endregion    


#NOTE - 固定 test_bt
def make_test_vec_norm_env(seed: int, param_path: str):
    """Creates a test environment and wraps it with loaded VecNormalize stats."""
    env = MgSurrogateEnv(
        reward_scale=0.0001,
        train_mode=False
    )
    env = ActionClipperWrapper_OffPolicy(env)
    env = DummyVecEnv([lambda: env])  # type: ignore
    
    env = VecNormalize.load(param_path, env)
    env.training = False     # Do not update stats during test
    env.norm_reward = False  # Return raw rewards
    return env

def collect_buffer_and_predict(env, model, buffer_size, beta):
    """
    1. 运行策略收集 buffer_size 个状态。
    2. 使用 Critic 批量预测这些状态的 CVaR。
    3. 返回 shape 为 (buffer_size, ) 的 CVaR 数组。
    """
    obs = env.reset()
    
    # 存储所有的 Observations
    obs_list = []
    
    # --- Phase 1: Rollout & Collection ---
    # 这一步是为了获取 "在该策略下可能会遇到的状态分布"
    print(f"  > Collecting {buffer_size} samples...")
    for _ in range(buffer_size):
        # 记录当前状态 (Normalized)
        # obs 是 (1, state_dim)，存入 list
        obs_list.append(obs.copy())
        
        # 执行动作 (Actor 决定)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        
        # 如果 done 了，DummyVecEnv 会自动 reset，我们只需继续循环即可
    
    # 将 list 转为 numpy array -> (N, state_dim)
    obs_array = np.vstack(obs_list)
    
    # --- Phase 2: Batch Prediction (Actor & Critic) ---
    print(f"  > Predicting Risk for {len(obs_array)} states...")
    
    # 转为 Tensor
    obs_tensor = torch.as_tensor(obs_array, device=model.device).float()
    
    model.policy.set_training_mode(False)
    
    with torch.no_grad():
        # 1. 获取对应状态下的动作 (Batch Action)
        # 注意：这里使用 model.actor 而不是 model.predict，以保持梯度图逻辑(虽然这里no_grad)并直接获得 Tensor
        actions_tensor = model.actor(obs_tensor) 
        
        # 定义内部函数：批量计算 CVaR
        def compute_batch_cvar_torch(quantiles_tensor, beta):
            """
            Input: (Batch, N_quantiles)
            Output: (Batch, )
            """
            batch_size, n_quantiles = quantiles_tensor.shape
            
            # 1. Sort along the quantile dimension (dim=1)
            sorted_q, _ = torch.sort(quantiles_tensor, dim=1)
            
            # 2. Determine tail size (alpha = 1 - beta)
            # 例如 beta=0.9, tail 就是最高的 10%
            tail_count = max(1, int(np.ceil((1.0 - beta) * n_quantiles)))
            
            # 3. Mean of the tail for each sample in batch
            # 取最后 tail_count 个值的平均
            cvar_batch = sorted_q[:, -tail_count:].mean(dim=1)
            
            return cvar_batch

        quantiles = model.cost_critic(obs_tensor, actions_tensor)
        # 2. Critic 预测分位数并计算 CVaR
        # Robust check: If output is a list/tuple, stack them (B, N, Q)
        if isinstance(quantiles, (list, tuple)):
            quantiles = torch.stack(quantiles, dim=1)
        
        # If shape is just (Batch, Quantiles) (single critic case), unsqueeze to (Batch, 1, Quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles.unsqueeze(1)
        
        sorted_q, _ = torch.sort(quantiles, dim=-1)

        # 2. Determine the tail size for CVaR
        n_quantiles = sorted_q.shape[-1]
        tail = max(1, int(np.ceil((1.0 - beta) * n_quantiles)))

        # 3. Calculate CVaR for EACH critic (mean of the tail)
        # Shape: (Batch, n_critics)
        cvar_per_critic = sorted_q[..., -tail:].mean(dim=-1)

        # 4. Conservative Estimate: Max CVaR across all critics
        # Shape: (Batch,) -> scalar after .item() since batch size is 1 here
        conservative_cvar = cvar_per_critic.max(dim=1).values



    # 转回 Numpy 数组返回
    return conservative_cvar.cpu().numpy()

def plot_buffer_distribution(run_id, cvar_data, cost_limit, output_dir,beta):
    """画出 Buffer 中所有状态的 CVaR 分布图"""
    mean_cvar = np.mean(cvar_data)
    
    plt.figure(figsize=(10, 6))
    
    # 直方图 + KDE
    sns.histplot(cvar_data, stat='density', kde=True, color='teal', alpha=0.5, label='State Risk Distribution')
    
    # 辅助线
    plt.axvline(cost_limit, color='red', linestyle='--', linewidth=2, label=f'Constraint Limit ({cost_limit})')
    plt.axvline(mean_cvar, color='blue', linestyle='-', linewidth=2, label=f'Mean CVaR ({mean_cvar:.2f})')
    
    plt.title(f"E-Buffer Constraint Verification\nRun: {run_id} | Batch Size: {len(cvar_data)}")
    plt.xlabel(f"Predicted CVaR (alpha={beta})")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f"buffer_constraint_{run_id}.png")
    plt.savefig(save_path)
    plt.close()
    
    return mean_cvar

def main():
    # 1. Setup Paths
    exp_root = os.path.join(BASE_LOG_DIR, EXP_NAME)
    timestamp = TIMESTAMP
    data_root = os.path.join(exp_root, timestamp, "data")
    config_root = os.path.join(data_root, "configs")
    output_dir = os.path.join(data_root, "verification_buffer")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Checking E-Buffer Constraint: {EXP_NAME}/{timestamp}")
    
    config_files = glob.glob(os.path.join(config_root, "*", "*.json"))
    config_files.sort()
    
    summary_results = []
    
    for cfg_path in config_files:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
            
        run_id = config.get("run_id")
        cost_limit = float(config.get("cost_limit", 0))
        seed = int(config.get("seed", 0))
        model_path = config.get("model_path")
        param_path = os.path.join(data_root, "params", run_id, f"{run_id}.pkl")
        
        print(f"\n--- Processing Run: {run_id} (Limit: {cost_limit}) ---")
        
     
        # 1. Load Env & Model
        env = make_test_vec_norm_env(seed, param_path)
        model = DSAC.load(model_path, env=env)
        
        # 2. Collect Data & Predict
        # 这一步生成 buffer 并计算每个点的 CVaR

        cvar_batch = collect_buffer_and_predict(env, model, buffer_size=256, beta=0.9)
        
        # 3. Analyze & Plot
        mean_cvar = plot_buffer_distribution(run_id, cvar_batch, cost_limit, output_dir,beta=0.9)
        
        # 4. Check Satisfaction
        is_satisfied = mean_cvar <= cost_limit
        # 也可以检查 worst-case (max cvar)
        max_cvar = np.max(cvar_batch)
        
        print(f"  > Limit: {cost_limit}")
        print(f"  > Mean CVaR: {mean_cvar:.4f} [{'SATISFIED' if is_satisfied else 'VIOLATED'}]")
        print(f"  > Max CVaR in Buffer: {max_cvar:.4f}")
        
        summary_results.append({
            "run_id": run_id,
            "cost_limit": cost_limit,
            "mean_cvar_pred": mean_cvar,
            "max_cvar_pred": max_cvar,
            "std_cvar_pred": np.std(cvar_batch),
            "is_satisfied": is_satisfied
        })
        
        env.close()
            
       
    # Save Summary
    if summary_results:
        df = pd.DataFrame(summary_results)
        csv_path = os.path.join(output_dir, f"buffer_constraint_summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to {csv_path}")

if __name__ == "__main__":
    main()