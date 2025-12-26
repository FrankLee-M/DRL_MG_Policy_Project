import os,sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, cast
from datetime import datetime
import torch
import copy 
# Stable Baselines3
# from stable_baselines3 import SAC
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


#NOTE - 测试 cvar 估计 vs cvar 真值

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
        train_mode=False,
        test_stochasitc=False, 
        test_begin_t=test_bt,
        test_end_t=test_et
    )
    env = ActionClipperWrapper_OffPolicy(env)
    env = DummyVecEnv([lambda: env])  # type: ignore
    
    env = VecNormalize.load(param_path, env)
    env.training = False     # Do not update stats during test
    env.norm_reward = False  # Return raw rewards
    return env


#region
def get_cost_critic_prediction(model, obs, action) -> Tuple[np.ndarray, float]:
    """
    Step 3: Predict
    Get the distribution (quantiles) from the EnsembleCostCritic and calculate conservative CVaR_pred.
    
    Adaptation:
    - Handles output shape (Batch, n_critics, n_quantiles).
    - Calculates CVaR for each critic independently.
    - Returns the MAX CVaR across critics (Conservative/Robust safety estimate).
    """
    model.policy.set_training_mode(False)
    beta = model.cvar_beta
    
    # Convert to torch tensors
    obs_tensor = torch.as_tensor(obs, device=model.device).float()
    action_tensor = torch.as_tensor(action, device=model.device).float()

    with torch.no_grad():
        # EnsembleCostCritic .forward() usually returns shape: (Batch, n_critics, n_quantiles)
        quantiles = model.cost_critic(obs_tensor, action_tensor)
        
        # Robust check: If output is a list/tuple, stack them (B, N, Q)
        if isinstance(quantiles, (list, tuple)):
            quantiles = torch.stack(quantiles, dim=1)
        
        # If shape is just (Batch, Quantiles) (single critic case), unsqueeze to (Batch, 1, Quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles.unsqueeze(1)

        # 1. Sort quantiles along the last dimension (quantile dim)
        # Shape: (Batch, n_critics, n_quantiles)
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

        # 5. Prepare data for visualization
        # We assume the first critic is representative for the distribution shape visualization
        # to avoid mixing multiple distributions in one histogram.
        # Alternatively, use quantiles.cpu().numpy().flatten() to see all critics mixed.
        q_numpy = quantiles[:, 0, :].cpu().numpy().flatten()
            
    return q_numpy, float(conservative_cvar.cpu().item())
#endregion



def run_k_rollouts(env, agent, fixed_action, gamma,snapshot_state,test_episodes=K_ROLLOUTS) :
    """
    Step 4: Rollout
    Reset environment K times to the SAME state s_t, execute fixed action a_t, 
    then run policy to end.
    """
    real_rewards = []
    real_costs = []
    env.reset()
    raw_env = get_base_env(env)
    
    # obs = env.reset()
    # target_time_index = raw_env.index_t 
    
    for k in range(test_episodes):
        # Restore the EXACT State s_t
        raw_env.__dict__.update(copy.deepcopy(snapshot_state))
        unique_seed = ( k * 11 + 123456) % (2**32 - 1)
        raw_env.env_rng = np.random.default_rng(unique_seed)
        raw_env.user_rng = np.random.default_rng(unique_seed+1)
        
        
        # print(raw_env.obs)
        # 2. Execute Fixed Action a_t (Step 1)
        obs, reward, done, info = env.step(fixed_action)
        
        cum_reward = reward
        current_gamma = gamma
        
        cum_cost  = info[0].get("complaint_cost",0.0)
        
        # 3. Continue running Policy (Step 2...T)
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward , done, info = env.step(action)
            cum_reward += current_gamma * reward
            cum_cost += current_gamma * info[0].get("complaint_cost",0.0)

            current_gamma *= gamma
            
            
            
        real_rewards.append(cum_reward)
        real_costs.append(cum_cost)
        
    return real_rewards,real_costs

#region
def plot_distribution(run_id, real_rewards_k_rollouts, real_costs_k_rollouts,  save_dir):
    """
    
    Step 6: Compare - Plotting
    """
    real_rewards_k_rollouts = np.array(real_rewards_k_rollouts).flatten()
    real_costs_k_rollouts = np.array(real_costs_k_rollouts).flatten()
    plt.figure(figsize=(10, 6))
    
    # Subplot 1: Rewards
    plt.subplot(2,1,1)
    sns.kdeplot(real_rewards_k_rollouts, label='Real Rewards (Rollouts)', fill=True, color='blue', alpha=0.5)
    plt.ylabel('Density')
    plt.xlabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Costs
    plt.subplot(2,1,2)
    sns.kdeplot(real_costs_k_rollouts, label='Real Costs (Rollouts)', fill=True, color='red', alpha=0.5) 
    plt.ylabel('Density')
    plt.xlabel('Cost')     
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Run {run_id}') # Use suptitle for global title
    plt.tight_layout() # Adjust spacing
    
    filename = os.path.join(save_dir, f"verification_{run_id}.png")
    plt.savefig(filename)
    plt.close()
#endregion

def main():
    # 1. Locate Experiment Directory
    exp_root = os.path.join(BASE_LOG_DIR, EXP_NAME)
    timestamp = TIMESTAMP
    data_root = os.path.join(exp_root, timestamp, "data")
    config_root = os.path.join(data_root, "configs")
    
    output_dir = os.path.join(data_root, f"verification_test_start_state")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating Experiment: {EXP_NAME}/{timestamp}")
    print(f"Output Directory: {output_dir}")

    # 2. Discover Configurations
    config_files = glob.glob(os.path.join(config_root, "*", "*.json"))
    results = []
    cost_list = []
    # Sort config files to be deterministic
    config_files.sort()

    for cfg_path in config_files:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        
        run_id = config.get("run_id", "unknown")
        cost_limit = float(config.get("cost_limit", 0))
        seed = config.get("seed", 0)
        model_path = config.get("model_path")
        param_path = os.path.join(data_root, "params", run_id, f"{run_id}.pkl")
        gamma = config.get("gamma")
        print(f"--- Processing Run: {run_id} ---")
        
      
        # 1. Step 1: Select State (Implicitly via Reset)
        dummy_env = make_test_vec_norm_env(seed, param_path)
        s_t= dummy_env.reset() # This is s_t (normalized)
        raw_dummy_env = get_base_env(dummy_env)
        snapshot_state = copy.deepcopy(raw_dummy_env.__dict__)
        
        # Load Model
        model = DSAC.load(model_path, env=dummy_env)
        
        # 2. Step 2: Fix Policy & Get Action a_t
        a_t, _ = model.predict(s_t, deterministic=True) # type: ignore
        
        # 3. Step 3: Predict (Cost-Critic)
        # Pass numpy arrays, model handles tensor conversion internally in our helper
        q_numpy,cvar_hat= get_cost_critic_prediction(model, s_t, a_t)
            
        
        # print(f"  > reward-expectation Prediction (Critic): {reward_pred:.4f}")
        
        # 4. Step 4: Rollout (Real World)
        # Define a lambda to recreate env exactly for each rollout
        env = make_test_vec_norm_env(seed, param_path)
        
        print(f"  > Running {K_ROLLOUTS} Monte Carlo rollouts from s_t...")
        real_rewards,real_costs = run_k_rollouts(env, model, a_t, gamma,snapshot_state)
        cvar_real = compute_cvar(np.array(real_costs), model.cvar_beta)

        # # 6. Step 6: Compare & Plot
        # plot_distribution(run_id,  real_rewards,real_costs,output_dir)
        
        cost_list.append({
            "run_id": run_id,            
            "real_cost":real_costs,
            "cvar_hat":cvar_hat,
            "cvar_real":cvar_real
            
            
        })
        
        

        
        dummy_env.close()
    
    plot_cost_distribution(cost_list,output_dir)

            
def plot_cost_distribution(cost_list, save_dir, cols=1):
    """
    将多个 Run 的 Cost 分布绘制在同一张大图的子图中。
    
    Args:
        cost_list: 包含字典的列表 [{'run_id':..., 'real_cost':..., 'cvar_hat':..., 'cvar_real':...}]
        save_dir: 保存路径
        cols: 每行显示的子图数量 (默认 2 列)
    """
    if not cost_list:
        print("Warning: cost_list is empty.")
        return

    num_plots = len(cost_list)
    rows =int(num_plots / cols) # 向上取整计算行数
    
    # 设置画布大小：高度随行数增加，宽度固定
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # 如果只有一个子图，axes 不是数组，需要转成数组以便统一处理
    if num_plots == 1:
        axes = np.array([axes])
    
    # 将多维 axes 数组扁平化 (例如 2x2 变成长度为4的一维数组)，方便循环
    axes_flat = axes.flatten()

    for i, data in enumerate(cost_list):
        ax = axes_flat[i]
        
        # 1. 提取数据
        run_id = data['run_id']
        real_costs = np.array(data['real_cost']).flatten()
        cvar_hat = data['cvar_hat']
        cvar_real = data['cvar_real']
        
        # 2. 绘制 KDE 分布
        sns.kdeplot(real_costs, ax=ax, fill=True, color='red', alpha=0.3, label='Cost Dist')
        
        # 3. 绘制竖线 (Vertical Lines)
        # 真实 CVaR (黑色虚线)
        ax.axvline(x=cvar_real, color='black', linestyle='--', linewidth=2, 
                   label=f'Real: {cvar_real:.2f}')
        # 估计 CVaR (蓝色点划线)
        ax.axvline(x=cvar_hat, color='blue', linestyle='-.', linewidth=2, 
                   label=f'Est: {cvar_hat:.2f}')
        
        # 4. 子图装饰
        ax.set_title(f'Run ID: {run_id}')
        ax.set_xlabel('Cost')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')

    # 5. 隐藏多余的空白子图 (如果总数不是 cols 的倍数)
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # 6. 全局布局调整与保存
    plt.suptitle(f'Cost Distribution & CVaR Estimation across {num_plots} Runs', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_set_str = str(test_set)   
    filename = os.path.join(save_dir, f"all_runs_distribution_test_{test_set_str}.png")
    plt.savefig(filename, bbox_inches='tight') # bbox_inches防止标题被切掉
    plt.close()
    
    print(f"Saved combined plot to {filename}")
    
if __name__ == "__main__":
    main()