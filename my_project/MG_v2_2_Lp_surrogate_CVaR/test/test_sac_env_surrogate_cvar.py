import os,sys
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, cast
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from tools.utilities import get_base_env, compute_cvar
from envs.config_para_mg_Nd_50 import begin_t, end_t, test_begin_t, test_end_t
from envs.action_wrapper import ActionClipperWrapper_OffPolicy
from envs.surrogate.env_mg_surrogate_v2_0 import MgSurrogateEnv
from algorithms.customerized_sac_v2_2 import DSAC

'''
可用于 
    run_sac_training_env_surrogate_v0.py 中 env-version：“v1/v2” 的测试
Strategy        | Avg Reward   | Avg Complaints-cost 
相较于 v0 版本：
-[ADD] 保存算法比较结果 到 csv 文件
-[DELETE] 删除对 全部 episode 测试结果的csv文件保存
-[Modify] 
    修改 complaint-num 为 complaint-cost 
    seed 参数获取，直接获取，不需要int保护
-⚠️ 此版本保存的图片只是 最后一次测试的结果
'''
env_version = "v2_0"
EXP_NAME = f"sac_training_env_surrogate_cvar" 
project_version = os.path.basename(os.path.normpath(parent_dir))

BASE_LOG_DIR = f"./tensorboard_logs/{project_version}"
TIMESTAMP = "2025_12_24_184139"  # <--- 请修改为你实际运行生成的时间戳
K_ROLLOUTS = 100  # K >= 100 to ensure statistical significance

test_set=True  # If False, test on training set
if test_set:
    test_bt = test_begin_t
    test_et = test_end_t
else:
    test_bt = begin_t
    test_et = end_t


def make_test_env(seed: int, param_path: str, cost_limit: float):
    """
    创建测试环境。
    关键修改：必须包含 AugmentedStateWrapper 且 initial_e = cost_limit
    """
    # 1. Base Environment (Test Mode)
    env = MgSurrogateEnv( 
        reward_scale=0.0001, 
        is_record=True, 
        train_mode=False,
        test_stochasitc=True, 
        test_begin_t=test_bt,
        test_end_t=test_et
    )
    env = ActionClipperWrapper_OffPolicy(env)
    env = DummyVecEnv([lambda: env]) # type: ignore
    
    if os.path.exists(param_path):
        # 加载训练好的均值和方差
        env = VecNormalize.load(param_path, env)
        env.training = False     # 测试模式：停止更新统计量
        env.norm_reward = False  # 测试模式：返回原始 Reward 以便统计
    
    return env


def evaluate_model(env, agent, run_id,test_episodes=K_ROLLOUTS):
    """
    对比 RL Agent, Baseline A (Rule-based), 和 Baseline B (Random)
    """
    print("--- Starting Evaluation ---")
    
    # 存储结果，增加 Random
    results = {
        'RL': {'rewards': [], 'profits': [], 'complaint_cost': [], 'prices': [], 'demands': []},
        'Baseline': {'rewards': [], 'profits': [], 'complaint_cost': [], 'prices': [], 'demands': []},
        'Random': {'rewards': [], 'profits': [], 'complaint_cost': [], 'prices': [], 'demands': []}
    }
    
    # --- 循环测试 --- 通过设置 seed 重置环境的 初始状态！
    for episode in range(test_episodes):
        
        # ==========================================
        # 1. 测试 RL 策略
        # ==========================================
        env.seed(episode)  
        obs = env.reset()
        # raw_env = get_base_env(env)
        # print("RL-env-reset to index-t:",raw_env.index_t)
        done = False
        ep_reward = 0
        ep_complaint_cost = 0
        
        trace_prices = []
        trace_demands = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True) 
            next_obs, reward, done, infos = env.step(action)
            info = infos[0]
            
            # VecEnv 返回的 reward 是数组，通常只有一个环境所以直接累加
            ep_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            ep_complaint_cost += info.get('complaint_cost', 0)
            
            if episode == test_episodes - 1:
                trace_prices.append(info.get('lp', 0))
                trace_demands.append(info.get('last_agg_load', 0))
            
            obs = next_obs
            
        results['RL']['rewards'].append(ep_reward)
        results['RL']['complaint_cost'].append(ep_complaint_cost)
        if episode == test_episodes - 1:
            results['RL']['prices'] = trace_prices
            results['RL']['demands'] = trace_demands

        # ==========================================
        # 2. 测试 Baseline A 策略 (Rule-Based: 1.5倍定价)
        # ==========================================
        raw_env = get_base_env(env)
        raw_env.reset(seed=episode) 
        # print("Rule-env-reset to index-t:",raw_env.index_t)
        
        done = False
        ep_reward_base = 0
        ep_complaint_base = 0
        
        trace_prices_b = []
        trace_demands_b = []
        
        while not done:
            reward, terminated, turncated, info = raw_env.base_policy_step()
            done = terminated or turncated
            ep_reward_base += reward
            ep_complaint_base += info.get('complaint_cost', 0)
            
            if episode == test_episodes - 1:
                trace_prices_b.append(info.get('lp', 0))
                trace_demands_b.append(info.get('last_agg_load', 0))

        results['Baseline']['rewards'].append(ep_reward_base)
        results['Baseline']['complaint_cost'].append(ep_complaint_base)
        if episode == test_episodes - 1:
            results['Baseline']['prices'] = trace_prices_b
            results['Baseline']['demands'] = trace_demands_b

        # ==========================================
        # 3. 测试 Baseline B 策略 (Random Strategy)
        # ==========================================
        # 使用 VecEnv 接口以保持动作空间一致性
        env.seed(episode) 
        obs = env.reset()
        # raw_env = get_base_env(env)
        # print("Random-env-reset to index-t:",raw_env.index_t)
        
        done = False
        ep_reward_rand = 0
        ep_complaint_rand = 0
        
        trace_prices_r = []
        trace_demands_r = []
        
        while not done:
            # 随机采样动作
            # env.action_space.sample() 返回单个动作，VecEnv step 需要列表
            random_action = [env.action_space.sample() for _ in range(env.num_envs)]
            
            next_obs, reward, done, infos = env.step(random_action)
            info = infos[0]
            
            ep_reward_rand += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            ep_complaint_rand += info.get('complaint_cost', 0)
            
            if episode == test_episodes - 1:
                trace_prices_r.append(info.get('lp', 0))
                trace_demands_r.append(info.get('last_agg_load', 0))
            
            obs = next_obs
            
        results['Random']['rewards'].append(ep_reward_rand)
        results['Random']['complaint_cost'].append(ep_complaint_rand)
        if episode == test_episodes - 1:
            results['Random']['prices'] = trace_prices_r
            results['Random']['demands'] = trace_demands_r

    # --- 打印统计结果 ---
    print(f"\n=== Final Results {run_id}")
    return results

def plot_results(results,save_path):
    mean_rl_r = np.mean(results['RL']['rewards'])
    mean_base_r = np.mean(results['Baseline']['rewards'])
    mean_rand_r = np.mean(results['Random']['rewards'])
    
    mean_rl_c = np.mean(results['RL']['complaint_cost'])
    mean_base_c = np.mean(results['Baseline']['complaint_cost'])
    mean_rand_c = np.mean(results['Random']['complaint_cost'])

    print(f"{'Strategy':<15} | {'Avg Reward':<12} | {'Avg Complaints':<15}")
    print("-" * 46)
    print(f"{'RL Agent':<15} | {mean_rl_r:<12.2f} | {mean_rl_c:<15.2f}")
    print(f"{'Rule Baseline':<15} | {mean_base_r:<12.2f} | {mean_base_c:<15.2f}")
    print(f"{'Random':<15} | {mean_rand_r:<12.2f} | {mean_rand_c:<15.2f}")
    
        # 3. 构造数据列表
    summary_data = [
        {
            'Strategy': 'RL Agent', 
            'Avg Reward': mean_rl_r, 
            'Avg Complaints': mean_rl_c
        },
        {
            'Strategy': 'Rule Baseline', 
            'Avg Reward': mean_base_r, 
            'Avg Complaints': mean_base_c
        },
        {
            'Strategy': 'Random', 
            'Avg Reward': mean_rand_r, 
            'Avg Complaints': mean_rand_c
        }
    ]




    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['RL']['prices'], label='RL', color='red')
    plt.plot(results['Baseline']['prices'], label='Rule', color='gray', linestyle='--')
    plt.plot(results['Random']['prices'], label='Random', color='green', alpha=0.5)
    plt.title("Pricing Strategy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results['RL']['demands'], label='RL', color='blue')
    plt.plot(results['Baseline']['demands'], label='Rule', color='gray', linestyle='--')
    plt.plot(results['Random']['demands'], label='Random', color='green', alpha=0.5)
    plt.title("Demand Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"comparison_strategy.png"))
    
    return summary_data

def save_to_csv(results,output_dir):
    summary_rows = []

    # 遍历每个策略 (RL, Baseline, Random)
    for strategy, metrics in results.items():
        # 获取 Episode 数量 (比如 100)
        num_episodes = len(metrics['rewards'])
        
        for i in range(num_episodes):
            summary_rows.append({
                'Strategy': strategy,
                'Episode': i,
                'Reward': metrics['rewards'][i],
                'Complaints': metrics['complaint_cost'][i],
                # 如果有 profit 也可以加进来
                # 'Profit': metrics['profits'][i] 
            })

    # 创建 DataFrame 并保存
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "eval_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary data saved to {summary_path}")

    # --- 效果示例 ---
    # Strategy | Episode | Reward | Complaints
    # RL       | 0       | 4.01   | 0.0
    # RL       | 1       | 3.98   | 0.1
    # ...


def main():
    exp_root = os.path.join(BASE_LOG_DIR, EXP_NAME)
    data_root = os.path.join(exp_root,TIMESTAMP , "data")
    config_root = os.path.join(data_root, "configs")
    output_dir = os.path.join(data_root, "test_summary") 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluating Experiment: {EXP_NAME}/{TIMESTAMP}")
    print(f"Output Directory: {output_dir}")    
    
    config_files = glob.glob(os.path.join(config_root, "**", "*.json"), recursive=True)
    if not config_files:
        print(f"No config files found in {config_root}")
        return
    
    all_summary_data = []
    for cfg_path in config_files:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
        
        run_id = config.get("run_id")
        if not run_id:
            run_id = os.path.basename(os.path.dirname(cfg_path))

        cost_limit = float(config.get("cost_limit", 0))
        seed = config.get("seed")
        # if seed != None:
        #     seed = int(seed)
        # seed = int(config.get("seed", 0))
        
        model_path = os.path.join(data_root, "models", run_id, f"{run_id}.zip")
        param_path = os.path.join(data_root, "params", run_id, f"{run_id}.pkl")
        
        if not os.path.exists(model_path):
            print(f"[Skip] Model not found: {model_path}")
            continue
            
        dummy_env = make_test_env(seed, param_path, cost_limit)
    
        model = DSAC.load(model_path, env=dummy_env)
        
        results = evaluate_model(dummy_env, model,run_id) # type: ignore
        summary_data = plot_results(results,output_dir)
        all_summary_data.extend(summary_data)
    
    final_df = pd.DataFrame(all_summary_data)
    save_path = os.path.join(output_dir,'all_configs_comparison_results.csv')
    final_df.to_csv(save_path, index=False)

    print(f"\n[Done] 所有配置的对比结果已汇总保存至: {save_path}")

        # save_to_csv(results,output_dir)
        

if __name__ == "__main__":
    main()