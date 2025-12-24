import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from tools.utilities import get_base_env
import pandas as pd
import os
import json

class EpisodeReturnCallback(BaseCallback):
    """
    自定义回调函数，用于输出每个 episode 的累计回报。
    """
    def __init__(self, verbose=1, csv_path: str | None = None, record_keys: list[str] | None = None):
        super().__init__(verbose)
        
        # generic recorder (schema-free, can stream to CSV)
        self.record_keys = record_keys
        self.record_buffer = []
        self.csv_path = csv_path
        self.episode_index = 0
        self.step_in_episode = 0
    

    def _on_step(self) -> bool:

        base_env = get_base_env(self.training_env, index=0)

        if base_env.is_record:
            infos = self.locals['infos']
            # 遍历 infos 列表
           
            for info in infos:
              
                # schema-free record of whatever appears in info
                filtered = {}
                keys_to_use = self.record_keys or list(info.keys())
                for k in keys_to_use:
                    if k in info:
                        v = info[k]
                        if isinstance(v, (np.ndarray, list, tuple)):
                            try:
                                v = np.asarray(v).tolist()
                            except Exception:
                                v = str(v)
                        elif isinstance(v, (np.floating, np.integer)):
                            v = float(v)
                        filtered[k] = v
                filtered["global_step"] = int(self.num_timesteps)
                filtered["episode_index"] = int(self.episode_index)
                filtered["step_in_episode"] = int(self.step_in_episode)
                self.record_buffer.append(filtered)
                self.step_in_episode += 1

        
        if self.locals['dones'][0]:
            # advance episode counter first
            self.episode_index += 1
            self.step_in_episode = 0
            # every 100 episodes, write a full snapshot without clearing buffer
            if self.episode_index % 50 == 0:
                self._flush_csv_snapshot()


            
        return True  # 继续训练


    # Optional: set CSV output path at runtime
    def set_csv_output(self, path: str):
        self.csv_path = path

    def _flush_csv_if_needed(self):
        if not self.csv_path:
            return
        if not self.record_buffer:
            return
        df = pd.DataFrame(self.record_buffer)
        out_path = self.csv_path
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        file_exists = os.path.exists(out_path)
        df.to_csv(out_path, mode='a', header=not file_exists, index=False)
        self.record_buffer.clear()
    
    # Write a full snapshot of current in-memory buffer to CSV (overwrite), do NOT clear buffer
    def _flush_csv_snapshot(self):
        if not self.csv_path:
            return
        if not self.record_buffer:
            return
        df = pd.DataFrame(self.record_buffer)
        out_path = self.csv_path
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # overwrite with full snapshot to avoid duplication
        df.to_csv(out_path, mode='w', header=True, index=False)
            

import numpy as np
import os
import matplotlib.pyplot as plt
import gymnasium as gym

class FixedStartEvalCallback(BaseCallback):
    """
    固定起点评估回调函数。
    每隔 eval_freq 步，在 fixed_start_indices 指定的时间点上运行测试，
    并将平均 Reward 和 Satisfaction 记录到 TensorBoard。
    """
    def __init__(self, 
                 eval_env, 
                 fixed_start_indices: list, 
                 eval_freq: int = 2000, 
                 verbose: int = 1,
                 deterministic = True,
                 best_model_save_path="./best_model_path"):
        """
        :param eval_env: 用于评估的环境 (必须设置 train_mode=False)
        :param fixed_start_indices: 固定的测试时间点列表 (hour_index)
        :param eval_freq: 每隔多少个训练步数执行一次评估
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.fixed_start_indices = fixed_start_indices
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.warm_ups = 10_000
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        
    def _on_step(self) -> bool:
        # 按照频率执行评估
        if self.n_calls >= self.warm_ups and self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward, mean_sat = self._run_fixed_eval()
            
            # 1. 记录到 TensorBoard (SB3 标准方式)
            self.logger.record("eval/fixed_set_reward", mean_reward)
            self.logger.record("eval/fixed_set_sat", mean_sat)
            
            # 2. 打印日志
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Fixed Eval Reward = {mean_reward:.2f} +/- {std_reward:.2f} | Sat = {mean_sat:.2f}")

            # 3. 保存最佳模型 (可选)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}!")

                save_path_model = os.path.join(self.best_model_save_path, "best_model.zip")
                self.model.save(save_path_model)

                save_path_vecnorm = os.path.join(self.best_model_save_path, "best_vecnormalize.pkl")
                
                if isinstance(get_base_env(self.training_env), VecNormalize):
                    self.training_env.save(save_path_vecnorm)  # type: ignore
                
        return True
    def _run_fixed_eval(self):
        rewards = []
        satisfactions = []
        
        # 获取最底层的 MgSatFBEnv 以修改 begin_t
        # VecNormalize -> DummyVecEnv -> ActionWrapper -> Monitor -> MgSatFBEnv
        # VecEnv 通常有 .envs 属性列表，或者通过 get_attr 获取属性
        
        # 备份原始 begin_t (注意：VecEnv 可能是多环境，这里假设只有1个)
        # 这是一个稍微 hack 的方法来穿透 VecEnv 设置属性
        original_begin_t = self.eval_env.get_attr('begin_t', indices=0)[0]

        for start_t in self.fixed_start_indices:
            # 1. 修改 Env 内部参数
            self.eval_env.set_attr('begin_t', start_t, indices=0)
            
            # 2. Reset (VecEnv reset 只返回 obs)
            # 此时 User RNG 已重置，Index_t 已重置
            obs = self.eval_env.reset()
            
            done = False
            episode_reward = 0.0
            episode_sats = []
            
            # VecEnv 的 done 自动 reset，所以我们需要自己判断何时停止
            # 对于 Fixed Evaluation，我们知道长度是 episode_length (24)
            # 最好显式循环 24 次，或者通过 info 判断
            
            steps_count = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                
                # VecEnv step
                obs, reward, dones, infos = self.eval_env.step(action)
                
                # reward 是一个数组 [r1]
                episode_reward += reward[0]
                
                # info 是一个列表 [info1]
                if 'ave_sat' in infos[0]:
                    episode_sats.append(infos[0]['ave_sat'])
                
                steps_count += 1
                if dones[0]: # Episode 结束
                    break
            
            rewards.append(episode_reward)
            if len(episode_sats) > 0:
                satisfactions.append(np.mean(episode_sats))
        
        # 还原
        self.eval_env.set_attr('begin_t', original_begin_t, indices=0)

        return np.mean(rewards), np.std(rewards), np.mean(satisfactions)
    
def get_monthly_fixed_indices(month_days_list=[1, 15, 25], total_hours=8760, data_range=None):
    """
    生成索引列表。
    :param data_range: tuple (min_idx, max_idx)，例如 (2880, 4343)。
                       如果提供，只返回落在这个范围内的索引。
    """
    fixed_indices = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    current_year_hour = 0
    for month_days in days_in_month:
        for day in month_days_list:
            if day <= month_days:
                hour_idx = current_year_hour + (day - 1) * 24
                
                # 范围检查
                is_valid = True
                if data_range:
                    if not (data_range[0] <= hour_idx <= data_range[1] - 24):
                        is_valid = False
                elif hour_idx >= total_hours - 24:
                    is_valid = False
                    
                if is_valid:
                    fixed_indices.append(hour_idx)
                    
        current_year_hour += month_days * 24
        
    return fixed_indices

# fixed_indices = get_monthly_fixed_indices(data_range=(2880,4343))
# print(fixed_indices)

def get_daily_start_indices(begin_t, end_t):
    """
    获取指定时间范围内，所有 hour_index == 0 的 t_index 列表。
    
    参数:
    begin_t (int): 验证集开始的 t_index
    end_t (int): 验证集结束的 t_index (包含)
    
    返回:
    list: 所有对应午夜（0点）的 t_index 列表
    """
    midnight_indices = []
    
    # 1. 找到范围内第一个 0 点
    # 计算 begin_t 当前的小时数 (0-23)
    current_hour = begin_t % 24
    
    if current_hour == 0:
        # 如果开始时间正好是 0 点
        first_midnight = begin_t
    else:
        # 如果不是，找到下一个 0 点（补齐剩余小时数）
        first_midnight = begin_t + (24 - current_hour)
    
    # 2. 从第一个 0 点开始，以 24 为步长生成索引，直到超过 end_t
    # range(start, stop, step) -> stop 是不包含的，所以用 end_t + 1
    for t in range(first_midnight, end_t + 1, 24):
        midnight_indices.append(t)
        
    return midnight_indices
