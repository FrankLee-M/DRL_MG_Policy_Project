import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from typing import Any, ClassVar, Optional, TypeVar, Union
from gymnasium import spaces
from typing import Optional, NamedTuple, List, Dict, Any, Union
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize

'''
based on V2_1
v2_2 :
    - 替换 多个单独定义的 cost-critic 为 EnsembleCostCritic

'''

class CustomReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    extra_field: torch.Tensor
    discounts: Optional[torch.Tensor] = None 

class CustomReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space, # type: ignore
            action_space, # type: ignore
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        # 初始化存储额外字段的数组
        self.extra_fields = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # 从 infos 提取数据
        extra_data = np.array([info.get("complaint_cost", 0.0) for info in infos]).reshape((self.n_envs, 1))
        self.extra_fields[self.pos] = extra_data
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(
        self, 
        batch_inds: np.ndarray, 
        env: Optional[VecNormalize] = None
    ) -> CustomReplayBufferSamples:
        standard_samples = super()._get_samples(batch_inds, env)
        
        data = self.extra_fields[batch_inds, 0, :]
        data_tensor = torch.as_tensor(data).to(self.device)
        
        # 安全获取 discounts
        discounts = getattr(standard_samples, "discounts", None)

        return CustomReplayBufferSamples(
            observations=standard_samples.observations,
            actions=standard_samples.actions,
            next_observations=standard_samples.next_observations,
            dones=standard_samples.dones,
            rewards=standard_samples.rewards,
            extra_field=data_tensor,
            discounts=discounts
        )

class DistributionalCostCritic(nn.Module):
    """
    Cost Critic (风险网络)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_quantiles=32):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quantiles) 
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x)

#ANCHOR - 新增 EnsembleCostCritic 类
class EnsembleCostCritic(nn.Module):
    """
    管理多个 DistributionalCostCritic
    """
    def __init__(self, state_dim, action_dim, num_quantiles, n_critics=2, hidden_dim=256):
        super().__init__()
        # 使用 ModuleList 注册所有子网络
        self.critics = nn.ModuleList([
            DistributionalCostCritic(state_dim, action_dim, hidden_dim, num_quantiles)
            for _ in range(n_critics)
        ])
    
    def forward(self, obs, action):
        # 返回形状: [Batch, n_critics, num_quantiles]
        return torch.stack([critic(obs, action) for critic in self.critics], dim=1)

class DSAC(SAC):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    
    def __init__(self, 
                policy: Union[str, type[SACPolicy]],
                env: Union[GymEnv, str],
                ###  DSAC 额外参数 ###
                 n_critics=2,        #ANCHOR - 新增 critic 数量参数 (TQC 建议 2 或 5)
                 cost_limit=10.0,      # 截图中的 Limit
                 num_quantiles=32,     # M
                 lambda_lr=1e-3,       # Lambda 的学习率
                 lambda_init=1.0,      # 初始 Lambda
                 u_lr=1e-2,          # u 的学习率
                 u_init=5.0,         # u 的初始猜测值
                 cvar_beta=0.9,     # VaR 的置信度 (1 - 风险容忍度)
                 drop_rate = 0.0,
                 tau_hat = None,
                ###  SB3 SAC 标准参数 ###
                learning_rate: Union[float, Schedule] = 3e-4,
                buffer_size: int = 1_000_000, 
                learning_starts: int = 100,
                batch_size: int = 256,
                tau: float = 0.005,
                gamma: float = 0.99,
                train_freq: Union[int, tuple[int, str]] = 1,
                gradient_steps: int = 1,
                action_noise: Optional[ActionNoise] = None,
                replay_buffer_class: Optional[type[ReplayBuffer]] = None,
                replay_buffer_kwargs: Optional[dict[str, Any]] = None,
                optimize_memory_usage: bool = False,
                n_steps: int = 1,
                ent_coef: Union[str, float] = "auto",
                target_update_interval: int = 1,
                target_entropy: Union[str, float] = "auto",
                use_sde: bool = False,
                sde_sample_freq: int = -1,
                use_sde_at_warmup: bool = False,
                stats_window_size: int = 100,
                tensorboard_log: Optional[str] = None,
                policy_kwargs: Optional[dict[str, Any]] = None,
                verbose: int = 0,
                seed: Optional[int] = None,
                device: Union[torch.device, str] = "auto",
                _init_setup_model: bool = True,
                 
        ):
        
        self.u_lr = u_lr
        self.u_init = u_init
        self.lambda_lr = lambda_lr
        self.lambda_init = lambda_init
        self.num_quantiles = num_quantiles
        self.n_critics = n_critics #ANCHOR - 保存 n_critics
        self.drop_rate = drop_rate
        self.tau_hat = tau_hat

        
        super().__init__( 
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=CustomReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            ent_coef = ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,             
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            )
        
        self.cost_limit = cost_limit
        self.cvar_beta = cvar_beta
        
    
    def _setup_model(self) -> None:
        super()._setup_model()
        
        state_dim = self.observation_space.shape[0] # type: ignore
        action_dim = self.action_space.shape[0] # type: ignore
        
        #ANCHOR - 使用 EnsembleCostCritic 初始化
        self.cost_critic = EnsembleCostCritic(
            state_dim, action_dim, num_quantiles=self.num_quantiles, n_critics=self.n_critics
        ).to(self.device)
        
        self.cost_critic_target = EnsembleCostCritic(
            state_dim, action_dim, num_quantiles=self.num_quantiles, n_critics=self.n_critics
        ).to(self.device)
        
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), 
            lr=self.lr_schedule(1)
        )        
        
        # 4. 初始化 Lambda 和 u (Lagrangian 参数)
        # 注意：load() 可能会覆盖这些值，但在 _setup_model 中必须先声明结构
        # self.u_param = torch.tensor(float(self.u_init), requires_grad=True, device=self.device)
        # self.u_optimizer = torch.optim.Adam([self.u_param], lr=self.u_lr)
        
        self.log_lambda = torch.tensor(np.log(self.lambda_init), requires_grad=True, device=self.device)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=self.lambda_lr)
        
        # 5. 构建分位点
        if self.tau_hat == None:
            self.tau_hat = torch.linspace(
                0.5 / self.num_quantiles, 
                1 - 0.5 / self.num_quantiles, 
                self.num_quantiles
            ).to(self.device)
        else:
            self.tau_hat = torch.tensor(self.tau_hat).to(self.device)
        

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self.cost_critic.train(True) #ANCHOR - 切换整个 Ensemble
        
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.cost_critic_optimizer, self.lambda_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses,cost_critic_losses =[], [], []
        dual_losses,nu_losses = [],[]
        update_lambdas,update_nus,update_cvars = [],[],[]


        # SB3 的标准训练循环
        for _ in range(gradient_steps):
            # 1. 采样数据
            # replay_data 包括: observations, actions, next_observations, rewards, dones
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env) # type: ignore

            # 将数据转为 Tensor (SB3 已经做好了，在 replay_data 中)
            # 注意: obs 里的最后一维是 e_t
            
        #  -  from sb3
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # -------------------------------------------------
            # ent_coef_optimizer
#!SECTION - temperature update
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

#!SECTION reweard-critic update
            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
#!SECTION cost-critic update
#ANCHOR - 使用 Ensemble 和 TQC (Cost: Keep Largest)
            current_cost = replay_data.extra_field.view(-1, 1) # type: ignore
            dones = replay_data.dones.view(-1, 1)

            with torch.no_grad():
                # [Batch, n_critics, N]
                next_z_ensemble = self.cost_critic_target(replay_data.next_observations, next_actions)
                
                # Flatten -> [Batch, n_critics * N]
                all_next_z = next_z_ensemble.flatten(1, 2)
                
                # Sort: 从小到大
                sorted_next_z, _ = torch.sort(all_next_z, dim=1)
                
                # TQC for Cost: 保留大的 (Risky/Conservative) 从n_drop 往后 
                total_quantiles = sorted_next_z.shape[1]
                n_drop = int(total_quantiles * self.drop_rate)
                # n_keep = total_quantiles - n_drop
                
                # [Batch, K] (Truncated)
                target_next_z_truncated = sorted_next_z[:,n_drop:] 

                # [Batch, K]
                target_cost_quantiles = current_cost + (1 - dones) * self.gamma * target_next_z_truncated

            # 获取当前预测 [Batch, n_critics, N]
            current_z_ensemble = self.cost_critic(replay_data.observations, replay_data.actions)
            target_z_3d = target_cost_quantiles.unsqueeze(1)
            tau_hat_3d = self.tau_hat.view(1, 1, -1, 1) # type: ignore
            cost_critic_loss = quantile_huber_loss(
                current_z_ensemble,    # [Batch, n_critics, N]
                target_z_3d,           # [Batch, 1, K]
                cum_prob=tau_hat_3d,   # [1, 1, N, 1]
                sum_over_quantiles=True
            )
   
            cost_critic_losses.append(cost_critic_loss.item()) 
            
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()
            
            
#!SECTION - policy- net update (Use Average CVaR)
            # current_var_estimation = self.u_param.detach()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            
            # 1. 想要钱 (-Q)
            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_q_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            
            # 2. 怕投诉 (lambda * CVaR)
            #ANCHOR - 使用 Ensemble 计算保守 CVaR
            # [Batch, n_critics, N]
            cost_z_ensemble = self.cost_critic(replay_data.observations, actions_pi)
            
            # 对每个 Critic 内部排序
            sorted_z, _ = torch.sort(cost_z_ensemble, dim=2)
            
            tail_size = int(self.num_quantiles * (1 - self.cvar_beta))
            
            # 计算每个 Critic 的 CVaR -> [Batch, n_critics]-尾部均值
            cvar_per_critic = sorted_z[:, :, -tail_size:].mean(dim=2)
            
            # Conservative Estimate: 取所有 Critic 中最大的 CVaR (最坏情况) -> [Batch]
            cvar_value = cvar_per_critic.max(dim=1).values
            
            update_cvars.append(cvar_value.detach().mean().item())
            lam = torch.exp(self.log_lambda).detach()
            constraint_loss = lam * F.relu(cvar_value - self.cost_limit)
            actor_loss = (-min_q_pi + ent_coef * log_prob + constraint_loss).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
           
#!SECTION - Lagrangian lambda update
 
            # -------------------------------------------------
            # Step 5: 更新参数 Lambda
            # -------------------------------------------------
            # Dual Gradient Ascent: lambda <- lambda + lr * (CVaR - delta)
            # 在 PyTorch 里也就是最小化: -lambda * (CVaR - delta)
            
            # limit_tensor = delta  # 或者 self.cost_limit
            dual_loss = -self.log_lambda * (cvar_value.detach().mean() -self.cost_limit)
            update_lambdas.append(self.log_lambda.detach().exp().item())
            
            self.lambda_optimizer.zero_grad()
            dual_loss.backward()
            self.lambda_optimizer.step()
            dual_losses.append(dual_loss.item())
            
            # Clip lambda 防止过大 (可选)
            with torch.no_grad():
                self.log_lambda.clamp_(max=5.0) # e^5 ~= 148
                
#!SECTION - nu update  
            # Z_cost_1 = cost_quantiles_pi_1.detach()
            # Z_cost_2 = cost_quantiles_pi_2.detach()

            # u_val = self.u_param
            # update_nus.append(u_val.detach().item())
            
            # excess_1 = torch.relu(Z_cost_1 - u_val)
            # coeff = 1.0 / (1.0 - self.cvar_beta)
            # nu_loss_1 = u_val + coeff * excess_1.mean()
            
            # excess_2 = torch.relu(Z_cost_2 - u_val)
            # nu_loss_2 = u_val + coeff * excess_2.mean()  
                      
            # nu_loss = (nu_loss_1 + nu_loss_2) / 2.0
            # nu_losses.append(nu_loss.item())
            
            # self.u_optimizer.zero_grad()
            # nu_loss.backward()
            # self.u_optimizer.step()
            
            # Update Target Networks
            if self.num_timesteps % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                #ANCHOR - 只需要更新 Ensemble 整体
                polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)
                
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))
        self.logger.record("train/updated_lambda", np.mean(update_lambdas))
        # self.logger.record("train/updated_nu", np.mean(update_nus))
        self.logger.record("train/updated_cvar", np.mean(update_cvars))


        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
             
    def _get_torch_save_params(self):
        """
        告诉 SB3 需要保存哪些自定义的 PyTorch 对象 (变量和 state_dicts)。
        """
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        
        #ANCHOR - 更新保存参数名称
        state_dicts.extend([
        "cost_critic", 
        "cost_critic_target", 
        "cost_critic_optimizer", 
        "lambda_optimizer", 
        # "u_optimizer"
    ])
        saved_pytorch_variables.extend([
            "log_lambda", 
            # "u_param"
        ])
        
        return state_dicts, saved_pytorch_variables

    def _get_torch_load_params(self):
        """
        通常不需要重写此方法，除非参数名在保存和加载时不一致。
        SB3 会自动根据 _get_torch_save_params 的返回列表进行加载。
        但为了保险起见，显式声明对应关系是好习惯。
        """
        return self._get_torch_save_params()
 