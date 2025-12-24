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


"""
training_env_surrogate
v0：固定一组 seed ，观察 surrogate 模型 是否有效 
"""



total_timesteps = 100_000
learning_rate=3e-4
# seed=42
ALGORITHM = "SAC"
reward_scale = 0.0001 
gamma = 0.99
env_version = "v2_0"
project_version = os.path.basename(os.path.normpath(parent_dir))

def make_base_env(env_version: str):
    if env_version == "v0_0":
        from envs.surrogate.env_mg_surrogate_v0_0 import MgSurrogateEnv
    elif env_version == "v0_1":
        from envs.surrogate.env_mg_surrogate_v0_1 import MgSurrogateEnv
    elif env_version == "v1_0":
        from envs.surrogate.env_mg_surrogate_v1_0 import MgSurrogateEnv
    elif env_version == "v1_1":
        from envs.surrogate.env_mg_surrogate_v1_1 import MgSurrogateEnv
    elif env_version == "v1_2":
        from envs.surrogate.env_mg_surrogate_v1_2 import MgSurrogateEnv
    elif env_version == "v1_3":
        from envs.surrogate.env_mg_surrogate_v1_3 import MgSurrogateEnv
    return   MgSurrogateEnv( 
        reward_scale=reward_scale, 
        is_record=True, 
        train_mode=True, 
        begin_t=begin_t, 
        end_t=end_t
    )


env = make_base_env(env_version)    


# ====== 全局统一时间戳 (所有实验共享) ======
# 分情况：tensorboard_logs or debug_tensorboard_logs
timestamp = datetime.now().strftime("%H%M%S")
exp_name = f"sac_training_env_surrogate_{env_version}"
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

def run_experiment(seed):
    """
    Runs a single DSAC experiment with a specific cost_limit and seed.
    """
    
    # 1. Create unique tags and paths for this specific run
    seed_tag = format_tag(seed)
    run_id = f"seed_{seed_tag}"
    
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

    monitored_env = Monitor(env)
    wrapped_env = ActionClipperWrapper_OffPolicy(monitored_env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    vev_norm_env = VecNormalize(venv=vec_env,gamma=gamma,norm_obs=True,norm_reward=False)
    
    # 3. Model Initialization (DSAC)
    # Pass the dynamic cost_limit and seed here
    model = SAC(
        "MlpPolicy",
        env=vev_norm_env,
        tensorboard_log=current_log_dir,
        learning_rate=learning_rate,
        seed=seed,                    # <--- Dynamic Parameter
        gamma=gamma
    )

    # 4. Callback
    
    callback = EpisodeReturnCallback(verbose=0, csv_path=csv_path)

  
    model.learn(total_timesteps=total_timesteps, callback=callback)



    model.save(model_path)
    vev_norm_env.save(param_path)

    # 7. Save Configuration

    
    config = {
        "description": f"DSAC parallel run for {run_id}",
        "run_id": run_id,
        "reward_scale": reward_scale,
        "seed": seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "total_timesteps": total_timesteps,
        "algorithm": ALGORITHM,
        "env": "MgSatFBEnv-surrogate",
        "log_dir": current_log_dir,
        "csv_path": csv_path,
        "model_path": model_path,
        "param_path":param_path,
        "timestamp": timestamp,
        "env_version":env_version
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Run: {run_id}")


# ==============================================================================
# Main Execution Block (Parallel)
# ==============================================================================
if __name__ == "__main__":

    # Define the parameters to sweep
    seeds = [100,200,300]         # Example seeds

    print(f"Starting Parallel Experiments: {exp_name}")
    print(f"Output Root: {base_root}")
    print(f"Parameters: Seeds={seeds}")
    
    # Create job list
    jobs = []
    for s in seeds:
        jobs.append(delayed(run_experiment)(s))

    # Execute in parallel
    # n_jobs=-1 uses all available CPU cores. Adjust if memory is tight.
    Parallel(n_jobs=-1)(jobs)

    print(f"All experiments finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    