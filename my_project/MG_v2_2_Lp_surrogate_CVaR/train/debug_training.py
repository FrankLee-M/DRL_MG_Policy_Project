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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from envs.callbacks import EpisodeReturnCallback
from envs.surrogate.env_mg_surrogate_v2_0 import MgSurrogateEnv
from envs.action_wrapper import ActionClipperWrapper_OffPolicy
from tools.utilities import format_tag
from envs.config_para_mg_Nd_50 import (begin_t,T_slot,end_t)
from algorithms.customerized_sac_v2_2 import DSAC

###########################################
# environment ： MgSatFBEnv-surrogate
#NOTE -   Training-Summer-DAYs (5 月-  6月 )
# env_mg 增加 baseline- compliant-number
##########################################
total_timesteps = 10_000
learning_rate=3e-4
# seed=42
ALGORITHM = "SAC"
reward_scale = 0.0001 
gamma = 0.99
seed = 42

    


# 2. Environment Setup
# Note: DSAC usually runs on 1 env, using DummyVecEnv for compatibility
env = MgSurrogateEnv(
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
    learning_rate=learning_rate,
    seed=seed,                    # <--- Dynamic Parameter
    gamma=gamma,
    n_critics=2,
    drop_rate=0.05,
    lambda_lr=1e-5,
    num_quantiles=20
)

# 4. Callback

callback = EpisodeReturnCallback(verbose=1)


model.learn(total_timesteps=total_timesteps, callback=callback)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Run")
