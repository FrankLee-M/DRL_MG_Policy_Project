#NOTE - 观察是否因为 均值/方差 在 训练集/测试集 差异巨大 导致 Out-of_distribution

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
import copy

# Stable Baselines3
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
    parser.add_argument("--exp_name", type=str, default="sac_training_env_surrogate_cvar_fixed_ood_period")
    return parser.parse_args()

args = get_args()
env_version = "v2_0"
EXP_NAME = args.exp_name
project_version = os.path.basename(os.path.normpath(parent_dir))

BASE_LOG_DIR = f"./tensorboard_logs/{project_version}"
TIMESTAMP = args.timestamp  # <--- 请修改为你实际运行生成的时间戳
K_ROLLOUTS = 1000  # 收集多少步数据用于分布对比

def make_test_env(seed: int, param_path: str):
    """Creates a test environment and wraps it with loaded VecNormalize stats."""
    env = MgSurrogateEnv(
        reward_scale=0.0001,
        train_mode=False,
        test_stochasitc=False, 
        test_begin_t=test_begin_t,
        test_end_t=test_end_t
    )
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

def make_train_env(seed: int, param_path: str):
    """Creates a training environment (for comparison) and wraps it with loaded VecNormalize stats."""
    # 注意：这里使用 train_mode=True 和训练集的时间范围
    env = MgSurrogateEnv(
        reward_scale=0.0001,
        train_mode=True, 
        test_stochasitc=True, # 训练时通常带有随机性
        test_begin_t=begin_t,
        test_end_t=end_t
    )
    env = ActionClipperWrapper_OffPolicy(env)
    env = DummyVecEnv([lambda: env])  # type: ignore

    if os.path.exists(param_path):
        # 关键点：加载和测试集完全一样的 Normalizer 参数，且设置 training=False
        # 这样我们就能看到：用这套参数去归一化训练数据，结果是否符合 N(0,1)
        env = VecNormalize.load(param_path, env)
        env.training = False 
        env.norm_reward = False
    else:
        print(f"Warning: Norm params not found at {param_path}, using raw env.")
        
    env.seed(seed)
    return env

def collect_observations(env, model, steps=1000):
    """Run the environment and collect normalized observations."""
    obs = env.reset()
    obs_list = []
    
    print(f"Collecting {steps} steps from environment...")
    for _ in range(steps):
        # obs 已经是归一化后的了 (如果 VecNormalize 正常工作)
        obs_list.append(obs.copy())
        
        # 使用模型选择动作，或者随机动作都可以，这里用模型
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        
    return np.vstack(obs_list)

def main():
    # 1. Locate Experiment Directory
    exp_root = os.path.join(BASE_LOG_DIR, EXP_NAME)
    timestamp = TIMESTAMP
    data_root = os.path.join(exp_root, timestamp, "data")
    config_root = os.path.join(data_root, "configs")
    
    output_dir = os.path.join(data_root, f"verification_vecnorm_ood")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating Experiment: {EXP_NAME}/{timestamp}")
    print(f"Output Directory: {output_dir}")

    # 2. Discover Configurations
    config_files = glob.glob(os.path.join(config_root, "*", "*.json"))
    if not config_files:
        print("No config files found!")
        return

    config_files.sort()
    cfg_path = config_files[0]
    
    with open(cfg_path, 'r') as f:
        config = json.load(f)
    
    run_id = config.get("run_id", "unknown")
    seed = config.get("seed", 0)
    model_path = config.get("model_path")
    param_path = os.path.join(data_root, "params", run_id, f"{run_id}.pkl")
    if not os.path.exists(param_path):
        param_path = os.path.join(data_root, "params", f"{run_id}.pkl")

    print(f"--- Processing Run: {run_id} ---")
    
    # 3. 准备环境
    dummy_test_env = make_test_env(seed, param_path)
    dummy_train_env = make_train_env(seed, param_path) 
    
    model = DSAC.load(model_path, env=dummy_test_env)
    
    # 4. 收集数据 
    print(">>> Collecting Training Data (Normalized)...")
    norm_train_obs = collect_observations(dummy_train_env, model, steps=K_ROLLOUTS)
    
    print(">>> Collecting Test Data (Normalized)...")
    norm_test_obs = collect_observations(dummy_test_env, model, steps=K_ROLLOUTS)

    # ==========================================
    # 5. 画图对比 (修改版：每8个维度一张图)
    # ==========================================
    dim_count = norm_train_obs.shape[1]
    print(f"Total Dimensions to plot: {dim_count}")
    
    # 设定每张图包含的子图数量 (2行4列 = 8个)
    plots_per_fig = 8
    rows = 2
    cols = 4
    
    # 计算需要多少张大图
    num_chunks = (dim_count + plots_per_fig - 1) // plots_per_fig
    
    print(f"Generating {num_chunks} figures (Grouping {plots_per_fig} dims per figure)...")

    for chunk_idx in range(num_chunks):
        start_dim = chunk_idx * plots_per_fig
        end_dim = min((chunk_idx + 1) * plots_per_fig, dim_count)
        
        # 创建大图画板，尺寸设大一点方便查看
        fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
        axes = axes.flatten() # 展平方便索引
        
        # 遍历当前 chunk 内的维度
        plot_idx = 0
        for current_dim in range(start_dim, end_dim):
            ax = axes[plot_idx]
            
            # 画 Train
            sns.histplot(norm_train_obs[:, current_dim], ax=ax, color="blue", label='Train', kde=True, stat="density", element="step", alpha=0.3)
            # 画 Test
            sns.histplot(norm_test_obs[:, current_dim], ax=ax, color="red", label='Test', kde=True, stat="density", element="step", alpha=0.3)
            
            # 辅助线
            ax.axvline(x=-3, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=3, color='k', linestyle='--', alpha=0.3)
            ax.set_title(f"Dim {current_dim}", fontsize=10)
            
            # 仅在第一个子图显示图例，避免遮挡
            if plot_idx == 0:
                ax.legend(loc='upper right')
            
            plot_idx += 1
            
        # 隐藏多余的空子图 (如果最后一组不足8个)
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])
            
        # 调整布局并保存
        fig.suptitle(f"Distribution Shift Check: Dims {start_dim} - {end_dim-1}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore # 留出标题空间
        
        save_name = f"dist_shift_group_{chunk_idx+1:02d}_dims_{start_dim}_{end_dim-1}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path)
        plt.close() # 关闭画布释放内存
        
        print(f"Saved: {save_name}")

    print("All plots generated.")

if __name__ == "__main__":
    main()