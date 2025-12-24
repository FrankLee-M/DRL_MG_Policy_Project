import os, sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, cast
from datetime import datetime
import torch
import copy  # <--- Added: 用于深拷贝环境状态

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

# NOTE - 测试 cvar 估计 vs cvar 真值 (任意时刻 t)

env_version = "v2_0"
EXP_NAME = f"sac_training_env_surrogate_cvar"
project_version = os.path.basename(os.path.normpath(parent_dir))

BASE_LOG_DIR = f"./tensorboard_logs/{project_version}"
TIMESTAMP = "184139"  # <--- 请修改为你实际运行生成的时间戳
K_ROLLOUTS = 1000  # K >= 100 to ensure statistical significance

# --- 新增：指定要检测的时间步 t ---
CHECK_TIME_STEP = 4  # 例如：检测第24个时间步的 Cost 分布 (t=0即为初始状态)

test_set = False  # If False, test on training set
if test_set:
    test_bt = test_begin_t
    test_et = test_end_t
else:
    test_bt = begin_t
    test_et = end_t


# 固定 test_bt
def make_test_env(seed: int, param_path: str):
    """Creates a test environment and wraps it with loaded VecNormalize stats."""
    env = MgSurrogateEnv(
        reward_scale=0.0001,
        train_mode=False,
        test_stochasitc=False, 
        test_begin_t=test_bt,
        test_end_t=test_et
    )
    # env.seed(seed) # Ensure base env is seeded
    env = ActionClipperWrapper_OffPolicy(env)
    env = DummyVecEnv([lambda: env])  # type: ignore

    if os.path.exists(param_path):
        env = VecNormalize.load(param_path, env)
        env.training = False  # Do not update stats during test
        env.norm_reward = False  # Return raw rewards
    else:
        print(f"Warning: Norm params not found at {param_path}, using raw env.")
    env.seed(seed)
    return env

#region
import torch
import numpy as np
from typing import Tuple

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
    Step 4: Rollout from arbitrary state s_t
    
    Fix: Instead of deepcopying the VecEnv (which causes errors), 
    we snapshot the underlying environment's dictionary state and restore it.
    """
    real_rewards = []
    real_costs = []
    
    # 1. 获取底层环境实例 (MgSurrogateEnv)
    env.reset()
    raw_env = get_base_env(env)
    

    
    # print(f"    > Starting {test_episodes} rollouts from step t={start_t}...")

    for k in range(test_episodes):
       
        
        raw_env.__dict__.update(copy.deepcopy(snapshot_state))
        unique_seed = ( k * 11 + 123456) % (2**32 - 1)
        raw_env.env_rng = np.random.RandomState(unique_seed)
        raw_env.user_rng = np.random.RandomState(unique_seed+1)
            
        # 4. 执行固定动作 a_t (Step t -> t+1)
        # 此时 env 内部已经是 t 时刻的状态，直接 step 即可
        obs, reward, done, info = env.step(fixed_action)
        
        cum_cost = info[0].get("complaint_cost", 0.0)
        current_gamma = gamma 
        
        # 5. 继续运行 Policy 直到结束
        step_in_loop = 0
        while not done:
            step_in_loop+=1
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            cum_cost += current_gamma * info[0].get("complaint_cost", 0.0)

            current_gamma *= gamma
        # print('step_in_loop:',step_in_loop)
        real_costs.append(cum_cost)
        
        
    return real_rewards, real_costs

#region
def plot_cost_distribution(cost_list, save_dir, cols=2):
    """
    将多个 Run 的 Cost 分布绘制在同一张大图的子图中。
    """
    if not cost_list:
        print("Warning: cost_list is empty.")
        return

    num_plots = len(cost_list)
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    if num_plots == 1:
        axes = np.array([axes])
    
    axes_flat = axes.flatten()

    for i, data in enumerate(cost_list):
        ax = axes_flat[i]
        
        run_id = data['run_id']
        real_costs = np.array(data['real_cost']).flatten()
        cvar_hat = data['cvar_hat']
        cvar_real = data['cvar_real']
        time_step = data.get('time_step', 0)
        
        sns.kdeplot(real_costs, ax=ax, fill=True, color='red', alpha=0.3, label='Real Cost Dist')
        
        ax.axvline(x=cvar_real, color='black', linestyle='--', linewidth=2, 
                   label=f'Real: {cvar_real:.2f}')
        ax.axvline(x=cvar_hat, color='blue', linestyle='-.', linewidth=2, 
                   label=f'Est: {cvar_hat:.2f}')
        
        ax.set_title(f'Run: {run_id} @ Step {time_step}')
        ax.set_xlabel('Future Discounted Cost')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(f'Cost Distribution (t={CHECK_TIME_STEP}) & CVaR Estimation across {num_plots} Runs', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_set_str = str(test_set)   
    filename = os.path.join(save_dir, f"all_runs_dist_step_{CHECK_TIME_STEP}_{test_set_str}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot to {filename}")
#endregion
from collections import defaultdict
def plot_all_points_comparison (cost_list, save_dir,cols=2):
    
    grouped_data = defaultdict(list)
    for entry in cost_list:
        run_id = entry['run_id']
        grouped_data[run_id].append(entry)

    # 获取所有唯一的 run_id 并排序（保证绘图顺序固定）
    unique_run_ids = sorted(list(grouped_data.keys()))
    num_plots = len(unique_run_ids)
    
    # --- 2. 设置画布 ---
    rows = int(np.ceil(num_plots / cols))
    # 动态调整高度：每行高度 5，宽度每列 8
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    
    # 兼容处理：如果只有1个图，axes不是数组，强制转为数组
    if num_plots == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # --- 3. 循环绘制每个 Run ---
    for i, run_id in enumerate(unique_run_ids):
        ax = axes_flat[i]
        run_entries = grouped_data[run_id]
        
        # 关键：按 check_time_step 排序，确保折线是按时间顺序连接的
        run_entries.sort(key=lambda x: x['check_time_step'])
        
        # 提取绘图数据 (X轴和Y轴)
        time_steps = [d['check_time_step'] for d in run_entries]
        cvar_reals = [d['cvar_real'] for d in run_entries]
        cvar_hats = [d['cvar_hat'] for d in run_entries]
        
        # --- 绘图 ---
        # 真实 CVaR (黑色实线)
        ax.plot(time_steps, cvar_reals, color='black', marker='o', linestyle='-', 
                label='Real CVaR (Rollouts)', linewidth=2, markersize=5, alpha=0.8)
        
        # 估计 CVaR (蓝色虚线)
        ax.plot(time_steps, cvar_hats, color='blue', marker='x', linestyle='--', 
                label='Est. CVaR (Critic)', linewidth=2, markersize=6, alpha=0.8)
        
        # --- 子图装饰 ---
        ax.set_title(f'Run ID: {run_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (t)')
        ax.set_ylabel('CVaR Value')
        
        # 设置 X 轴刻度为整数
        if len(time_steps) > 0:
            ax.set_xticks(time_steps)
            
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # (可选) 计算平均绝对误差 MAE 并显示在图上
        if len(cvar_reals) > 0:
            mae = np.mean(np.abs(np.array(cvar_reals) - np.array(cvar_hats)))
            ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- 4. 清理与保存 ---
    # 删除多余的空子图
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.suptitle(f'CVaR Estimation Tracking across {num_plots} Runs', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, f"cvar_tracking_comparison_all_runs_test_{test_set}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined comparison plot to {filename}")
    
    
def main():
    # 1. Locate Experiment Directory
    exp_root = os.path.join(BASE_LOG_DIR, EXP_NAME)
    timestamp = TIMESTAMP
    data_root = os.path.join(exp_root, timestamp, "data")
    config_root = os.path.join(data_root, "configs")
    
    output_dir = os.path.join(data_root, f"verification_test_step_{CHECK_TIME_STEP}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating Experiment: {EXP_NAME}/{timestamp}")
    print(f"Target Time Step: {CHECK_TIME_STEP}")
    print(f"Output Directory: {output_dir}")

    # 2. Discover Configurations
    config_files = glob.glob(os.path.join(config_root, "*", "*.json"))
    results = []
    cost_list = []

    config_files.sort()

    for cfg_path in config_files:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        
        run_id = config.get("run_id", "unknown")
        # cost_limit = float(config.get("cost_limit", 0))
        seed = config.get("seed", 0)
        model_path = config.get("model_path")
        param_path = os.path.join(data_root, "params", run_id, f"{run_id}.pkl")
        gamma = config.get("gamma")
        print(f"--- Processing Run: {run_id} ---")
        
        def check_one_point():
            
            dummy_env = make_test_env(seed, param_path)
            model = DSAC.load(model_path, env=dummy_env)
            
            # 2. Step 1: Run Policy to reach state s_t
            obs = dummy_env.reset()
            done = False
            current_step = 0
            
            # 运行直到达到 CHECK_TIME_STEP
            while current_step < CHECK_TIME_STEP:
                action, _ = model.predict(obs, deterministic=True) # type: ignore
                obs, _, done, _ = dummy_env.step(action)
                current_step += 1
          
            # 3. Step 2: Fix Policy & Get Action a_t (for the immediate next step)
            s_t = obs
            a_t, _ = model.predict(s_t, deterministic=True) # type: ignore
            
            raw_dummy_env = get_base_env(dummy_env)
            snapshot_state = copy.deepcopy(raw_dummy_env.__dict__)
        
            # 4. Step 3: Predict (Cost-Critic) at (s_t, a_t)
            q_numpy, cvar_hat = get_cost_critic_prediction(model, s_t, a_t)
            print(f"  > CVaR Prediction (Critic) at t={CHECK_TIME_STEP}: {cvar_hat:.4f}")
            
            # 5. Step 4: Rollout (Real World) from s_t
            # 我们传入当前的 env 对象作为快照，run_k_rollouts_from_t 会处理 deepcopy
            env = make_test_env(seed, param_path)
            real_rewards, real_costs = run_k_rollouts(env, model, a_t, gamma,snapshot_state)
            
            cvar_real = compute_cvar(np.array(real_costs), model.cvar_beta)
            print(f"  > CVaR Real (Rollouts): {cvar_real:.4f}")

            cost_list.append({
                "run_id": run_id,            
                "real_cost": real_costs,
                "cvar_hat": cvar_hat,
                "cvar_real": cvar_real,
                "time_step": CHECK_TIME_STEP
            })
            
            env.close()
    
            plot_cost_distribution(cost_list, output_dir)
         
        def check_all_point():   
            for check_sp in range(23):
                # 1. Make Env and Load Model
                # 我们需要一个基础环境来跑前缀 trajectory
                dummy_env = make_test_env(seed, param_path)
                model = DSAC.load(model_path, env=dummy_env)
                
                # 2. Step 1: Run Policy to reach state s_t
                obs = dummy_env.reset()
                done = False
                current_step = 0
                
                # 运行直到达到 CHECK_TIME_STEP
                while current_step < check_sp:
                    action, _ = model.predict(obs, deterministic=True) # type: ignore
                    obs, _, done, _ = dummy_env.step(action)
                    current_step += 1
               
                s_t = obs
                a_t, _ = model.predict(s_t, deterministic=True) # type: ignore
                raw_dummy_env = get_base_env(dummy_env)
                snapshot_state = copy.deepcopy(raw_dummy_env.__dict__)

                # 4. Step 3: Predict (Cost-Critic) at (s_t, a_t)
                q_numpy, cvar_hat = get_cost_critic_prediction(model, s_t, a_t)
                # print(f"  > CVaR Prediction (Critic) at t={check_sp}: {cvar_hat:.4f}")
                
                # 5. Step 4: Rollout (Real World) from s_t
                # 我们传入当前的 env 对象作为快照，run_k_rollouts_from_t 会处理 deepcopy
                env = make_test_env(seed, param_path)
                real_rewards, real_costs = run_k_rollouts(env, model, a_t, gamma,snapshot_state)
                
                cvar_real = compute_cvar(np.array(real_costs), model.cvar_beta)
                # print(f"  > CVaR Real (Rollouts): {cvar_real:.4f}")

                cost_list.append({
                    "run_id": run_id,            
                    "real_cost": real_costs,
                    "cvar_hat": cvar_hat,
                    "cvar_real": cvar_real,
                    "check_time_step": check_sp
                })
                
                env.close()
                dummy_env.close()
        
        # check_one_point()
        check_all_point()
    
    output_dir = os.path.join(data_root, "verification_test_step_all_test")
    os.makedirs(output_dir, exist_ok=True)
    plot_all_points_comparison(cost_list,output_dir)


if __name__ == "__main__":
    main()