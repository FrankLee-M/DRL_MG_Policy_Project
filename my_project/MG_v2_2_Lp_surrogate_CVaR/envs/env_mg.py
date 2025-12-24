
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional
from typing import Optional, Any, Dict


from my_project.RA_obs.data_process import lambda_hv_data,res_data

from envs.action_wrapper import ActionClipperWrapper_OffPolicy
from envs.config_para_mg_Nd_50 import *
from envs.env_inner_user import  User_SFB_Model

'''
MG_Model With "True" User Complaint Feedback in state
Aligned with envs/surrogate/env_mg_surrogate_v2.py
Complaint-number & complaint cost feeadback 
'''
###########################################
# 修改 reset 函数
print("MG_Model in Version-251219：Adding obs-noise in step func")
##########################################


#region get_price_rolling_quantile 
def get_price_rolling_quantile(index_t, window:int = 168):
    current_price = lambda_hv_data[index_t].copy()

    if index_t >= window:
        window_prices = lambda_hv_data[index_t-window+1:index_t+1].copy()
        roll_quantile = (window_prices <= current_price).mean()
    else:
        # 窗口不足时，直接用0.5或者用当前已有窗口
        window_prices = lambda_hv_data[:index_t+1]
        roll_quantile = (window_prices <= current_price).mean() if len(window_prices) > 0 else 0.5
        
    return np.array([roll_quantile])
#endregion


#region get_observations
def get_observations(t_start):
    res_pre = res_data[t_start:t_start+predict_length].copy()
    res_his = res_data[t_start-historical_length:t_start].copy()
    
    lmhv_pre = lambda_hv_data[t_start:t_start+predict_length].copy()
    lmhv_his = lambda_hv_data[t_start-historical_length:t_start].copy()
    
    return  np.concatenate((res_his, res_pre)),np.concatenate((lmhv_his, lmhv_pre))
#endregion

#region calculate_time_indices
def calculate_time_indices(t_index):
    """
    t_index : 0-8759 // one-year data
    return hour_index, remaining_days, week_index, month_index
    0 <= hour_index <= 23
    0 <= remaining_days <= 30
    0 <= week_index <= 51
    0 <= month_index <= 11
    """
    # 定义每个月的天数（非闰年）
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 计算当天的小时数
    hour_index = t_index % 24 
    
    # 计算天数
    day_index = t_index // 24
    
    # 计算周数（每周7天）
    week_index = day_index // 7
    
    # 计算月数和剩余天数
    month_index = 0
    remaining_days = day_index
    while remaining_days >= month_days[month_index]:
        remaining_days -= month_days[month_index]
        month_index += 1
    
    return hour_index, remaining_days, week_index, month_index 
#endregion

# hour_index, remaining_days, week_index, month_index = calculate_time_indices(4344)             
def encode_time_index(t_index):
    index_hour, _, index_week, index_month = calculate_time_indices(t_index)
    angles = np.array([
        2 * np.pi * index_hour / 23,
        2 * np.pi * index_week / 51,
        2 * np.pi * index_month / 11
    ])
    sin_cos = np.empty(6, dtype=np.float32)
    sin_cos[::2] = np.sin(angles)
    sin_cos[1::2] = np.cos(angles)
    return sin_cos
 
#!SECTION ： Esp model with user satisfaction feedback in state
class MgComplaintFBEnv(gym.Env):
    
    """
    Initialize the class instance with the envirnment.

    env_config Parameters:
        begin_t:(int) The start day of the training/testing ; hour-level indexed form Jan.1st 0 am
        random_obs : (bool) Add noise to observations
        is_record : (bool) record action data 
        User_model: (cls)  The user model class, default is UserModel() 

    Fixed parameters:
    -----------
    T_dec : (int) The decision dimension
    T_ep : (int) The length of one training episode; usualy day/week/month/year
    T_slot : (int) The length data used for prediction/observation; usually 24 hour
    
    state_dim : (int) observations 
            [res*T_slot,lmhv*T_slot, soc, index_hour, index_day,index_week,index_month]      
    action_dim :(int) [lambda_p] at t 
    
    Reset parameters:
    soc : the state of battery in step-loop function  
    index_t : the time index in year-hour-index

    lp_last, pg_last, pb_last : the last step decision variables, used in ramp rate constraint

    """
    metadata = {"render_modes": [], }  
    
    def __init__(self,
                 is_complaint_model: bool = True,
                 reward_scale = 1.0,
                 train_mode = True, # train or test mode
                 begin_t :int = begin_t,
                 end_t :int =  end_t,
                 test_begin_t =  test_begin_t, 
                 test_end_t = test_end_t, # one-week test data
                 is_record:bool = True,
                 is_user_dynamic:bool = False,
                #  reward_mode:int = 0,
                 random_obs: bool = False, 
                 render_mode: Optional[str] = None,
                 user_model = User_SFB_Model,
                 test_stochasitc:bool = False,
                #   test_seed:int=  100,
                 realization_sigma: float = 0.05,  # [新增] 实际执行时的波动标准差 (5%)
                 user_uncertainty: float = 0.05,   # [新增] 用户响应的波动           
                 ):
        
        self.unit_costs = unit_costs
        
        self.realization_sigma = realization_sigma
        self.user_uncertainty = user_uncertainty
        
        self.is_complaint_model = is_complaint_model
        self.test_stochasitc = test_stochasitc
        # self.test_seed = test_seed    
       #region reward parameter         
        self.reward_scale = reward_scale           
        self.on_grid_price = on_grid_price
        self.is_user_dynamic = is_user_dynamic     
        # self.reward_mode = reward_mode
        self.range_lmhv =range_lmhv
        self.range_res = range_res
        # endregion reward parameter

        
        # user model
        self.User = user_model(is_ps_dynamic=self.is_user_dynamic)  

        # train settings
        self.train_mode = train_mode  # 是否测试模式/和策略的test mode 一致
        if self.train_mode:
            self.begin_t = begin_t
            self.end_t = end_t
            self.max_episode_steps = episode_length
        else:
            self.begin_t = test_begin_t
            self.end_t = test_end_t
            self.max_episode_steps = episode_length
        
       
        self.index_t = None
        self.random_obs = random_obs
        self.is_record = is_record
        
        self._step_in_loop = 0  # The count of steps in episode.
        self._current_hour_index = None  # The count of steps in 24-hour-index.
        self.env_rng = None
        self.user_rng = None
      
        
    # fixed parameter imported from env_parameter.py   
        self.T_dec = T_dec
        self.T_slot = T_slot
        self.action_dim = 1
        self.state_dim = state_dim 
     

        self.lp_max = lp_max
        self.lp_min = lp_min

       
        # self.dit_last = di_last
        self.lp_last = None  # last lambda_p, used in ramp rate constraint
        self.pg_last = None  
        self.pb_last = None
        
        self.delta_lp = delta_lp
    
       
        
        self.lmhv_kmin = lmhv_kmin
        self.lmhv_kmax = lmhv_kmax

        
     
        
        self.obs_soc_index = obs_soc_index
        self.obs_res_index = obs_res_index
        self.obs_lmhv_index = obs_lmhv_index
#FIXME - 此处 ？obs_time_dim or obs_time_index ？
        self.obs_time_index = obs_time_dim
        
        self.action_last_indices = action_last_indices
        self.action_bound_indices = action_bound_indices

        self.observation_space = spaces.Box(
            low=np.full((self.state_dim,), -np.inf),  # 根据需求调整
            high=np.full((self.state_dim,), np.inf),   # 根据需求调整
            shape=(self.state_dim,),
        )


        self.action_space = spaces.Box(np.full(self.action_dim, -1),np.full(self.action_dim, 1))
        self.current_min_action = self.action_space.low.copy()
        self.current_max_action =  self.action_space.high.copy()
       
        self.render_mode = render_mode
        self.screen = None 
        self.action_last = None
        self.action_delta = np.array([delta_lp])
        self.complaint_cost = 0.0
        
        #NOTE - 数据增强
        
        self._obs_shifted = False
        # On-demand initialized base views for observations and episode shift (in hours)
        self._obs_base_res = None
        self._obs_base_lmhv = None
        self._obs_shift_hours = 0

        self._buffer_his_res = deque(maxlen=historical_length)
        self._buffer_his_lmhv = deque(maxlen=historical_length)



        
    def reset(self,
              seed: Optional[int] = None, 
              options: Optional[dict] = None):
        """
        Standard Gymnasium reset with seed handling.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.env_rng = np.random.default_rng(seed)
            self.user_rng = np.random.default_rng(seed + 1)
            
        # 如果没有传入 seed 且生成器尚未初始化 (第一次调用)，则进行默认随机初始化
        elif self.env_rng is None or self.user_rng is None:
            self.env_rng = np.random.default_rng()
            self.user_rng = np.random.default_rng()
            
        if self.train_mode:
            # 训练模式：随机采样起始时间
            sample_t = self.env_rng.integers(self.begin_t, self.end_t - 23)
            sample_t = sample_t - sample_t % 24 
            self.index_t = np.clip(sample_t, self.begin_t, self.end_t - 23)
            self._current_hour_index = 0

        else:
            # 测试模式
            if not self.test_stochasitc:
                # 固定测试场景 (例如总是从第一天开始)
                self.index_t = self.begin_t
            else:
                # 随机测试场景 (使用我们刚刚重置过的 env_rng)
                sample_t = self.env_rng.integers(self.begin_t, self.end_t - 23)
                sample_t = sample_t - sample_t % 24 
                self.index_t = np.clip(sample_t, self.begin_t, self.end_t - 23)

            self._current_hour_index = 0


        # 本环境 不考虑ESS ，仅考虑 电价-lp 决策
        self.soc = 0.5
        # 初始化 上一时刻的决策 
        #NOTE - lp_last & action_last 不需要归一化，用于 _compute_new_action_bound 
        self.lp_last = self.env_rng.uniform(self.lp_min, self.lp_max)
        self.action_last = np.array([self.lp_last])


        self.User._random_init_user_model(self.user_rng,self._current_hour_index)       
        
        self.complaint_cost = 0.0
        self.dit_last = self.User.dit_last.copy()  # pyright: ignore[reportOptionalMemberAccess]
        self.last_agg_load =np.sum(self.dit_last)

#NOTE - 数据增强
        def _get_padded_slice(arr, start_idx, end_idx, pad_value=0.0):
            '''
            带填充的安全切片,如果索引超出边界则用 pad_value 填充
            '''
            arr = np.asarray(arr)
            total_len = len(arr)
            out_len = end_idx - start_idx
            out = np.full((out_len,), pad_value, dtype=arr.dtype)
            
            src_start = max(0, start_idx)
            src_end = min(total_len, end_idx)
            if src_start < src_end:
                dest_start = src_start - start_idx
                out[dest_start:dest_start + (src_end - src_start)] = arr[src_start:src_end]
            return out

        s_idx = self.begin_t - historical_length
        e_idx = self.end_t + self.max_episode_steps+predict_length
        self._obs_base_res = _get_padded_slice(res_data, s_idx, e_idx, pad_value=0.0)
        self._obs_base_lmhv = _get_padded_slice(lambda_hv_data, s_idx, e_idx, pad_value=0.0)
        
        # 训练时每个 episode 采用随机循环平移；测试为 0 平移
        # if self.train_mode:
        #     K_days = self.env_rng.integers(7,14)  # pyright: ignore[reportOptionalOperand, reportOptionalMemberAccess]
        #     self._obs_shift_hours = int(K_days) * 24
        # else:
        #     self._obs_shift_hours = 0
        # ⚠️后期可以修改
        self._obs_shift_hours = 0

        # 维护历史数据
        self._buffer_his_res.clear()
        self._buffer_his_lmhv.clear()


        self.obs = self._get_obs()
        self._step_in_loop = 0    
        # NOTE: 构造 info，将 wrapper 需要的信息放入 info 中
        info = {
            "action_last": self.action_last,
            "action_delta": self.action_delta,
            "min_action": self.current_min_action,
            "max_action": self.current_max_action
        }
        

        return self.obs ,info          
    

    #region get_observations
    def clip_obs_data(self,quantile_window = 168, loc=1.0,scale=0.01):
        # 重新抽取数据 ： +5% 随机扰动
        bias_data = self.env_rng.normal(loc = loc,scale =scale)  # pyright: ignore[reportOptionalMemberAccess]
        # 基于 _get_padded_slice 生成的 base view 获取观测数据
        bias_index = self.index_t - self.begin_t + historical_length  # pyright: ignore[reportOptionalOperand]
        
        # 提取观测窗口 [历史 historical_length | center ｜ 预测 predict_length]，使用模索引避免复制与滚动
        center = bias_index + self._obs_shift_hours
        idx = np.arange(center - historical_length, center + predict_length, dtype=np.int64)
         
        res_window = np.take(self._obs_base_res, idx)  # type: ignore
        lmhv_window = np.take(self._obs_base_lmhv, idx)  # type: ignore
        res_obs = self.range_res * bias_data * res_window
        lmhv_obs = self.range_lmhv * bias_data * lmhv_window

        len_his = len(self._buffer_his_res)
        if len_his > 0:
            # # 安全保护，防止历史长度超出 obs 窗口
            # len_his = min(len_his, len(res_obs))
            # 覆盖末尾的历史段
            res_obs[24-len_his:24] = list(self._buffer_his_res)[:len_his]
            lmhv_obs[24-len_his:24] = list(self._buffer_his_lmhv)[:len_his]

        # 价格分位在滚动后的同一时间点上计算（就地计算，避免复制）
        base_len = len(self._obs_base_res)  # type: ignore
        cur_pos = center % base_len
        current_price = float(self._obs_base_lmhv[cur_pos])  # type: ignore
        w_idx = (np.arange(cur_pos - (quantile_window - 1), cur_pos + 1, dtype=np.int64) % base_len)
        w_prices = np.take(self._obs_base_lmhv, w_idx, mode='wrap')  # type: ignore
        roll_quantile = (w_prices <= current_price).mean()
        price_quantile = np.array([roll_quantile])
        return res_obs,lmhv_obs,price_quantile
    #endregion

    def _get_obs(self):
        # 获取观测数据并缩放 + 5% 随机扰动
        obs_res, obs_lmhv, lmhv_rq= self.clip_obs_data(loc=1.0,scale=0.0025)
        
        # obs_res  = obs_res * self.range_res
        # obs_lmhv = obs_lmhv * self.range_lmhv

        # 获取时间状态
        time_state = encode_time_index(self.index_t)

        # user 相关状态 /
        # user_state = np.array([self.last_ave_sat_level, self.last_agg_load])
        user_state = np.array([self.complaint_cost, self.last_agg_load])
        if not self.is_complaint_model:
            user_state = np.array([0.0, self.last_agg_load])
        
        soc_state = np.array([self.soc])
      
        # 确保所有数组都是1维的
        action_last_flat = np.atleast_1d(self.action_last.copy())  # pyright: ignore[reportOptionalMemberAccess]
      
        temp_obs_for_bounds = np.concatenate([
            obs_res, obs_lmhv, lmhv_rq, 
            action_last_flat.copy(),np.zeros(self.action_dim*2),  # pyright: ignore[reportOptionalMemberAccess]
            user_state, soc_state, time_state
        ], dtype=np.float32)
        
        # new_low, new_high = self._compute_new_action_bound(temp_obs_for_bounds)
        self._compute_new_action_bound(temp_obs_for_bounds)
        # self.current_min_action, self.current_max_action = new_low, new_high

        # 更新动作边界
        temp_obs_for_bounds[action_bound_indices] = np.concatenate([self.current_min_action, self.current_max_action])

        return temp_obs_for_bounds



    def _compute_new_action_bound(self,imcomplete_obs):
        """
        计算新的动作边界,并更新 self.current_min_action, self.current_max_action
        obs =  [res*(his_24+pre_12),lmhv*(his_24+pre_12), soc, sin/cos[index_hour,index_week,index_month]]
        action = [lambda_p,p_b], 其中，lambda_p,p_b 受到ramp rate constrain
        """

        lphv = imcomplete_obs[self.obs_lmhv_index].copy()
        
        
        # 直接就地修改
        # lambda_p (索引0)
        if self.lp_last is not None:
            lp_min_candidates = np.array([self.lmhv_kmin * lphv, self.lp_min, self.lp_last - self.delta_lp])
            lp_max_candidates = np.array([self.lmhv_kmax * lphv, self.lp_max, self.lp_last + self.delta_lp])
            self.current_min_action[0] = np.max(lp_min_candidates)
            self.current_max_action[0] = np.min(lp_max_candidates)
        else:
            self.current_min_action[0] = max(self.lmhv_kmin * lphv, self.lp_min)
            self.current_max_action[0] = min(self.lmhv_kmax * lphv, self.lp_max)

        # 数值保护：确保下界不超过上界（若出现，按中点或 clip）
        invalid = self.current_min_action > self.current_max_action
        if np.any(invalid):
            # 简单策略：将下界裁剪为不超过上界（也可取中点 self.current_min_action[invalid] = self.current_max_action[invalid]）
            self.current_min_action[invalid] = np.minimum(self.current_min_action[invalid], self.current_max_action[invalid])
   
        
    def step(self,action):

        res_forecast = self.obs[self.obs_res_index]
        lmhv_forecas = self.obs[self.obs_lmhv_index]
        
        # 观测随机性
        noise_res = self.env_rng.normal(1.0, self.realization_sigma) # type: ignore
        noise_price = self.env_rng.normal(1.0, self.realization_sigma) # type: ignore # 可选：电价是否波动
        
        res = res_forecast*noise_res
        lambda_hv = lmhv_forecas*noise_price
        # print(noise_price)
#NOTE - 记录历史数据，用于观测补全
        self._buffer_his_lmhv.append(lambda_hv)
        self._buffer_his_res.append(res)

        lp =np.clip( action[0],lambda_hv*self.lmhv_kmin,lambda_hv*self.lmhv_kmax)

        #one step response from users: 可以引入用户响应的不确定性
        d_i_t,sat_i_t,complaint_number = self.User.user_response([lp],self._current_hour_index%24)  # pyright: ignore[reportOptionalOperand]
        d_t =  np.sum(d_i_t)
        complaint_cost = self.unit_costs[self._current_hour_index%24]*complaint_number # type: ignore
        # sat_i_t 仅作为监测使用，不作为 MG-env observation 
        # d_i_t_base,_,complaint_num_base = self.get_baseline_feedback(lp_base=lambda_hv*1.5,hour_index=self._current_hour_index%24) # pyright: ignore[reportOptionalOperand]
        
        # cvar_sat = self.cvar_users(sat_i_t, beta=0.9)
        # print("cvar_sat:",cvar_sat)
                
        
        
        phv = d_t  - res
        hv_pos = np.maximum(0,phv)
        hv_neg = np.maximum(0,-phv) 
        
       
        electricity_income = d_t*lp
            
        hv_cost =  lambda_hv*hv_pos  - self.on_grid_price*hv_neg 
        
        reward =  electricity_income  - hv_cost 
        
        reward = reward*self.reward_scale
        
   
        
        # update state
        # _current_hour_index ：indext 对应的 当天 hour-index
        self._current_hour_index += 1  # pyright: ignore[reportOperatorIssue]
        self._step_in_loop += 1

        self.index_t += 1  # pyright: ignore[reportOperatorIssue]
        self.lp_last = lp
 

        self.action_last = np.array([lp])
        
        self.dit_last = d_i_t
        self.last_agg_load = d_t
        self.monitor_psi = self.User.ps_i.copy()  # ps_i 后续可能会改变，需要 copy  # pyright: ignore[reportOptionalMemberAccess]
        self.ave_sat_level = np.mean(sat_i_t)  # 更新平均满意度水平
        
        # 扩大
        # ⚠️ 在 不考虑complaint约束的情况下，complaint_rate 相当于是 状态噪声，不利于策略训练，此处需要 置 0 
        # self.complaint_rate = 0.0
        # self.complaint_cost = max(0.0,complaint_number-complaint_num_base)/self.User.Nd * 1.0
        self.complaint_cost = complaint_cost
        self.complaint_rate = complaint_number/self.User.Nd

        self.obs = self._get_obs()
        
        info = {
            "action_last": self.action_last,
            "action_delta": self.action_delta,
            "min_action": self.current_min_action,
            "max_action": self.current_max_action
        }
        # 记录额外信息
        if self.is_record:
            info.update ({
                "res": res.tolist(),
                "lmhv"  : lambda_hv,
                "lp": lp,
                "pg":0,
                "pb": 0,
                "phv": phv,
                "d_t":d_t,
                # "dit": d_i_t.tolist(),
                "soc": self.soc,
                # "obs": (self.obs).tolist(),
                "reward": reward,
                # "sat_i":sat_i_t.tolist(),
                "ave_sat": self.ave_sat_level,
                "last_agg_load": self.last_agg_load,
                # "user_psi": self.monitor_psi.tolist(),
                # "cvar_sat": cvar_sat,
                "complaint_number":complaint_number,
                "complaint_rate":self.complaint_rate,
                "complaint_cost":self.complaint_cost
            })


     
        # truncated = False
        
        #NOTE -  无论从何时开始，运行 step_in_loop=24 后，episode 结束 
        # truncated = False if self._current_step < self.max_episode_steps else True  #type: ignore
        terminated = False if self._step_in_loop < self.max_episode_steps else True  #type: ignore
           
        return self.obs, reward, terminated, False, info
    
    
    def get_baseline_feedback(self, lp_base,hour_index):
        """
        lp: 设定为 hv-price * 1.5 
        获取在给定电价下，用户的反馈
        主要用于基线对比
        """
         # pyright: ignore[reportOptionalOperand]
        return self.User.user_response([lp_base],hour_index)
   
    def base_policy_step(self):
        '''
        base-policy: lp = 1.5*lambda_hv
        '''
        res_forecast = self.obs[self.obs_res_index]
        lmhv_forecas = self.obs[self.obs_lmhv_index]
        
        # 观测随机性
        noise_res = self.env_rng.normal(1.0, self.realization_sigma) # type: ignore
        noise_price = self.env_rng.normal(1.0, self.realization_sigma) # type: ignore # 可选：电价是否波动
        
        res = res_forecast*noise_res
        lambda_hv = lmhv_forecas*noise_price
        # print(noise_price)
#NOTE - 记录历史数据，用于观测补全
        self._buffer_his_lmhv.append(lambda_hv)
        self._buffer_his_res.append(res)

        lp_base = 1.5*lambda_hv
        lp_min_candidates = np.array([self.lmhv_kmin * lambda_hv, self.lp_min, self.lp_last - self.delta_lp]) # type: ignore
        lp_max_candidates = np.array([self.lmhv_kmax * lambda_hv, self.lp_max, self.lp_last + self.delta_lp]) # type: ignore
            
        lp =np.clip(lp_base,np.max(lp_min_candidates),np.min(lp_max_candidates))


        # #one step response from users: 可以引入用户响应的不确定性
        d_i_t,sat_i_t,complaint_number = self.User.user_response([lp],self._current_hour_index%24)  # pyright: ignore[reportOptionalOperand]
        d_t =  np.sum(d_i_t)
        complaint_cost = self.unit_costs[self._current_hour_index%24]*complaint_number # type: ignore
        # # sat_i_t 仅作为监测使用，不作为 MG-env observation 
        # d_i_t_base,_,complaint_num_base = self.get_baseline_feedback(lp_base=lambda_hv*1.5,hour_index=self._current_hour_index%24) # pyright: ignore[reportOptionalOperand]
        
        # cvar_sat = self.cvar_users(sat_i_t, beta=0.9)
        # print("cvar_sat:",cvar_sat)
                
        # 
        
        phv = d_t  - res
        hv_pos = np.maximum(0,phv)
        hv_neg = np.maximum(0,-phv) 
        
       
        electricity_income = d_t*lp
            
        hv_cost =  lambda_hv*hv_pos  - self.on_grid_price*hv_neg 
        
        reward =  electricity_income  - hv_cost 
        
        reward = reward*self.reward_scale
        
   
        
        # update state
        # _current_hour_index ：indext 对应的 当天 hour-index
        self._current_hour_index += 1  # pyright: ignore[reportOperatorIssue]
        self._step_in_loop += 1

        self.index_t += 1  # pyright: ignore[reportOperatorIssue]
        self.lp_last = lp
 

        self.action_last = np.array([lp])
        
        self.dit_last = d_i_t
        self.last_agg_load = d_t
        self.monitor_psi = self.User.ps_i.copy()  # ps_i 后续可能会改变，需要 copy  # pyright: ignore[reportOptionalMemberAccess]
        self.ave_sat_level = np.mean(sat_i_t)  # 更新平均满意度水平
        
        # 扩大
        # ⚠️ 在 不考虑complaint约束的情况下，complaint_rate 相当于是 状态噪声，不利于策略训练，此处需要 置 0 
        # self.complaint_rate = 0.0
        # self.complaint_cost = max(0.0,complaint_number-complaint_num_base)/self.User.Nd * 1.0
        self.complaint_cost = complaint_cost
        self.complaint_rate = complaint_number/self.User.Nd

        self.obs = self._get_obs()
        
        info = {
            "action_last": self.action_last,
            "action_delta": self.action_delta,
            "min_action": self.current_min_action,
            "max_action": self.current_max_action
        }
        # 记录额外信息
        if self.is_record:
            info.update ({
                "res": res.tolist(),
                "lmhv"  : lambda_hv,
                "lp": lp,
                "pg":0,
                "pb": 0,
                "phv": phv,
                "d_t":d_t,
                # "dit": d_i_t.tolist(),
                "soc": self.soc,
                # "obs": (self.obs).tolist(),
                "reward": reward,
                # "sat_i":sat_i_t.tolist(),
                "ave_sat": self.ave_sat_level,
                "last_agg_load": self.last_agg_load,
                # "user_psi": self.monitor_psi.tolist(),
                # "cvar_sat": cvar_sat,
                "complaint_number":complaint_number,
                "complaint_rate":self.complaint_rate,
                "complaint_cost":self.complaint_cost
            })


     
        # truncated = False
        
        #NOTE -  无论从何时开始，运行 step_in_loop=24 后，episode 结束 
        # truncated = False if self._current_step < self.max_episode_steps else True  #type: ignore
        terminated = False if self._step_in_loop < self.max_episode_steps else True  #type: ignore
           
        return reward, terminated, False, info
    

if __name__ == "__main__":
    
    import argparse
    import numpy as np
    env = MgComplaintFBEnv(is_user_dynamic=False,train_mode=True) 
    env = ActionClipperWrapper_OffPolicy(env)
    
    # With action wrapper 
    #NOTE -  action-scale-to [-1,1]
    
    for seed in range(10):
        action = 1.0* np.ones([action_dim,]) 
        env.reset(seed = seed)
        obs, reward, terminated,truncted, info = env.step(action)
        # print("Observation:", obs)
        print("Reward:", reward)
        # print("Done:", terminated)
        print("complaint_cost:", info['complaint_cost'])    
        
 
    