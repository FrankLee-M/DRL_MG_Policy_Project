import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import json
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from datetime import datetime
from joblib import Parallel, delayed

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
 

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from envs.callbacks import EpisodeReturnCallback
from envs.action_wrapper import ActionClipperWrapper_OffPolicy
from tools.utilities import format_tag
from envs.config_para_mg_Nd_50 import (begin_t,T_slot,end_t)
from envs.env_mg import MgComplaintFBEnv
from algorithms.customerized_sac_v2_2 import DSAC
"""
training_env_cvar v0:
    - 使用真实的 user-response 数据进行训练
    - 使用 DSAC 算法在 MG Surrogate 环境中进行训练，目标是优化 CVaR 性能指标。
    - 主要测试 cost-critic 在 不同参数设置下的表现。
        - drop_rate: 0.05, 0.1，0.2
        - num_quantiles: 20, 50
        - n_critics: 2, 5

"""

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # 定义你的参数，并给一个默认值
    parser.add_argument("--drop_rate", type=float, default=0.05, help="drop rate")
    parser.add_argument("--num_quantiles", type=int, default=20, help="num quantiles")
    parser.add_argument("--n_critics", type=int, default=2, help="number of critics")
    return parser.parse_args()

args = get_args()
total_timesteps = 100_000
learning_rate=3e-4
# seed=42
ALGORITHM = "SAC"
reward_scale = 0.0001 
gamma = 0.99
# default version 
env_version = "v1_0"
project_version = os.path.basename(os.path.normpath(parent_dir))

gamma = 0.99
# cost_limit = 2.0
num_quantiles = args.num_quantiles
n_critics = args.n_critics
lam_lr = 1e-5
nu_lr = 1e-2
n_epochs = 3
ent_coef = 0.0
drop_rate = args.drop_rate

# ====== 全局统一时间戳 (所有实验共享) ======
timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
exp_name = f"sac_training_env_surrogate_cvar_fixed_ood"
base_root = f"./tensorboard_logs/{project_version}/{exp_name}/{timestamp}"
base_dirs = {
    "tensorboard_logs": os.path.join(base_root, "logs"),
    "models": os.path.join(base_root, "data/models"),
    "params": os.path.join(base_root, "data/params"),
    "configs": os.path.join(base_root, "data/configs"),
    "training_data": os.path.join(base_root, "data/training_data")
}

for d in base_dirs.values():
    os.makedirs(d, exist_ok=True)

print(f"tensorboard serve --logdir ./{base_dirs}/training_data --port 6006")

def run_experiment(cost_limit,seed):
    """
    Runs a single DSAC experiment with a specific cost_limit and seed.
    """
    
    # 1. Create unique tags and paths for this specific run
    c_tag = format_tag(cost_limit)

    seed_tag = format_tag(seed)
    run_id = f"cost_{c_tag}_seed_{seed_tag}"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Run: {run_id}")


    current_log_dir = os.path.join(base_dirs["tensorboard_logs"], run_id)
    os.makedirs(current_log_dir, exist_ok=True)
    
    current_csv_dir = os.path.join(base_dirs["training_data"], run_id)
    os.makedirs(current_csv_dir, exist_ok=True)
    csv_path = os.path.join(current_csv_dir, f"{run_id}.csv")

    model_dir = os.path.join(base_dirs["models"], run_id)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{run_id}.zip")

    
    param_dir = os.path.join(base_dirs["params"], run_id)
    os.makedirs(param_dir, exist_ok=True)
    param_path = os.path.join(param_dir, f"{run_id}.pkl")


    config_dir = os.path.join(base_dirs["configs"], run_id)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{run_id}.json")
    
    
    
    # 2. Environment Setup
    # Note: DSAC usually runs on 1 env, using DummyVecEnv for compatibility
    env = MgComplaintFBEnv( 
            reward_scale=reward_scale, 
            is_record=True, 
            train_mode=True, 
            begin_t=begin_t, 
            end_t=end_t
        )

    monitored_env = Monitor(env)
    wrapped_env = ActionClipperWrapper_OffPolicy(monitored_env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    vev_norm_env = VecNormalize(venv=vec_env,gamma=gamma,norm_obs=True,norm_reward=False)
    
    # 3. Model Initialization (DSAC)
    # Pass the dynamic cost_limit and seed here
    model = DSAC(
        "MlpPolicy",
        env=vev_norm_env,
        cost_limit=cost_limit,       # <--- Dynamic Parameter
        num_quantiles=num_quantiles,
        n_critics=n_critics,
        drop_rate = drop_rate,
        u_lr=nu_lr,
        lambda_lr=lam_lr,
        tensorboard_log=current_log_dir,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        seed=seed,                    # <--- Dynamic Parameter
        gamma=gamma,
    )

    # 4. Callback
    
    callback = EpisodeReturnCallback(verbose=0, csv_path=csv_path)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    model.save(model_path)
    vev_norm_env.save(param_path)

   
    config = {
        "description": f"DSAC parallel run for {run_id}",
        "run_id": run_id,
        "reward_scale": reward_scale,
        "seed": seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "total_timesteps": total_timesteps,
        "algorithm": ALGORITHM,
        "env": "MgComplaintFBEnv",
        "log_dir": current_log_dir,
        "csv_path": csv_path,
        "model_path": model_path,
        "param_path":param_path,
        "timestamp": timestamp,
        "env_version":env_version,
        "algorithm":"DSAC-v2-2",
        "n_critics":n_critics,
        "num_quantiles":num_quantiles,
        "cost_limit":cost_limit,
        "drop_rate":drop_rate
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Run: {run_id}")


# ==============================================================================
# Main Execution Block (Parallel)
# ==============================================================================
if __name__ == "__main__":

    cost_limits = [0.8]  # Example cost limits
    seeds = [100]         # Example seeds

    print(f"Starting Parallel Experiments: {exp_name}")
    print(f"Output Root: {base_root}")
    print(f"Parameters: Cost Limits={cost_limits}, Seeds={seeds}")
    
    # Create job list
    jobs = []
    for c_lim in cost_limits:
        for s in seeds:
            jobs.append(delayed(run_experiment)(c_lim, s))

    # Execute in parallel
    # n_jobs=-1 uses all available CPU cores. Adjust if memory is tight.
    Parallel(n_jobs=-1)(jobs)
    print(f"All experiments finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")