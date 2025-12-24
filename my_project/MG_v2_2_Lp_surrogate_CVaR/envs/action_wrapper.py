from gymnasium import ActionWrapper
import numpy as np



class ActionClipperWrapper_OffPolicy(ActionWrapper):
    def __init__(self, env):
        super(ActionClipperWrapper_OffPolicy, self).__init__(env)
        # 初始化当前动作边界
        self.dynamic_min_action = self.env.unwrapped.action_space.low.copy()  #type: ignore
        self.dynamic_max_action = self.env.unwrapped.action_space.high.copy()  #type: ignore
        
        # 增量与上一步动作（由环境提供）
        self.action_delta = self.env.unwrapped.action_delta.copy()  #type: ignore
        self.action_last = None  #type: ignore

        # # 缓存索引，减少属性查找，同时支持可选的增量索引
        # self._bound_idx = self.env.unwrapped.action_bound_indices  #type: ignore
        # self._last_idx = self.env.unwrapped.action_last_indices  #type: ignore
        # # 当前环境，ation_delta 保持不变，所以不需要 _delta_idx，如果后续发生变化，可用
        # self._delta_idx = getattr(self.env.unwrapped, "action_delta_indices", None)

    def reset(self, **kwargs):
        """
        reset 时,通过obs 获取环境中更新的 动作上下界
        """
        obs,info = self.env.reset(**kwargs)
        self.action_last = info["action_last"]
        self.action_delta = info["action_delta"]
        self.update_action_bound(info["min_action"], info["max_action"])
        
        return obs,info

    
    def action(self, action):
        """
        在动作 [-1,1] 传递给环境之前进行还原。
        action 为改变的增量！
        """

        # 保证为 ndarray 并进行形状校验/一般不会出问题，这是网络的输出
        # action = np.asarray(action)
        # if action.shape != self.action_last.shape:
        #     raise ValueError(f"Action shape {action.shape} does not match expected shape {self.action_last.shape}.")

        action_change_delta  = self.action_delta * action
        unscale_action = self.action_last + action_change_delta
        unscale_action = np.clip(unscale_action, self.dynamic_min_action, self.dynamic_max_action)
        # print("action at current step is",action[0])
        return unscale_action
        
    def step(self, action):
   
        next_obs, reward, terminated, truncated, info = super().step(action)
      
        self.action_last = info["action_last"]
        self.action_delta = info["action_delta"]
        self.update_action_bound(info["min_action"], info["max_action"])
        
        return next_obs, reward, terminated, truncated, info
    
    def update_action_bound(self, new_min, new_max):
        """
        更新当前动作边界。
        """
        self.dynamic_min_action = new_min
        self.dynamic_max_action = new_max
    
    
        
class ActionClipperWrapper_OnPolicy(ActionClipperWrapper_OffPolicy):
    def __init__(self, env):
        # supper (class).__init__(self, env) : 调用 class  父类 的 __init__。”
        # 此处，直接调用 自己的父类 ActionClipperWrapper_OffPolicy 的 __init__
        super().__init__(env)
    
    def action(self, action):
        """
        在动作传递给环境之前进行剪裁。
        policy 输出 采样得到的action，通过 tanh 限制在【-1，1】，再进行rescale
        """

        action = np.tanh(action)
        action_change_delta  = self.action_delta * action
        unscale_action = self.action_last + action_change_delta
        unscale_action = np.clip(unscale_action, self.dynamic_min_action, self.dynamic_max_action)
        # print("action at current step is",action[0])
        # unscale_action = self.dynamic_min_action + (0.5 * (action + 1.0) * (self.dynamic_max_action - self.dynamic_min_action))
        
        return unscale_action
        

    