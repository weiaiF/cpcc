
from builtins import filter, int, list, print, range, zip
import os
import copy
import functools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.consisitency_pollicy.karras_diffusion import KarrasDenoiser
from agents.model import MLPActor, MLPCritic, MLPCriticTwin
# from agents.model_cpql import MLPActor, MLPCritic
from agents.helpers import EMA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HLGuassLoss(nn.Module):
	def __init__(self, v_min, v_max, num_bins, sigma, v_expand, v_expand_mode="both"):
		super().__init__()
		self.v_min = v_min
		self.v_max = v_max
		self.num_bins = num_bins
		self.sigma = sigma
  
		expand = (v_max - v_min) * v_expand
		if v_expand_mode == "both":
			self.v_min -= expand / 2
			self.v_max += expand / 2
		elif v_expand_mode == "min":
			self.v_min -= expand
		elif v_expand_mode == "max":
			self.v_max += expand
  
		self.support = torch.linspace(
			self.v_min, self.v_max, num_bins + 1, dtype=torch.float32
		).to(device)

	def forward(self, logits: torch.Tensor, target:torch.Tensor) -> torch.Tensor:
		return F.cross_entropy(logits, self.transform_to_probs(target))

	def transform_to_probs(self, target):
		# 根据 Stop regressing 文章的简易代码 HLGaussLoss 很需要使用这里的 clip 来维持数值稳定性
		# 而另外两种方案 Two-hot 和 Categorical Distributional RL 也进行了类似的裁剪但是他们允许查询支持外的值
		target = torch.clip(target, self.v_min, self.v_max)  # [bs, 1]
  
		# 这里相当于是使用 erf 误差函数 计算 0-x的高斯积分 
		# 下式进行了对应的换元操作 t = (x - mu) / sigma
  		# 同时这里的 sigma 已经进行了对应的处理 sigma = 0.75*(v_max - v_min) / num_bins 
		cdf_evals = torch.special.erf(
			(self.support - target) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
		) # [bs, num_bins + 1]
  
		# 对于这里是否要使用 torch.maximum 进行裁剪需要实验说明
		# cdf_evals = torch.special.erf(
        # 	torch.maximum((self.support - target.unsqueeze(-1)), torch.tensor(1e-6)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
    	# )

		z = cdf_evals[..., -1] - cdf_evals[..., 0]
		bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1] # [bs, num_bins]
		return bin_probs / (z + torch.tensor(1e-6)).unsqueeze(-1)  # [bs, num_bins]

	def transform_from_probs(self, probs):
		centers = (self.support[:-1] + self.support[1:]) / 2
		return torch.sum(probs* centers, dim = -1)



class CPQL_CL(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space=None,
                 discount=0.99,
                 max_q_backup=False,
                 alpha=1.0,
                 eta=1.0,
                 # ema params
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 # lr
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 q_mode='q',
                 sigma_max=80.0,
                 expectile=0.6,
                 sampler="onestep",
                 # cl loss 相关
                 num_classes=21,
                 sigma_frac=0.75,
                 v_max = 2,
                 v_min = -10,
                 v_expand=0.3,
                 v_expand_mode="both",
                 
                 loss_type = "l2"
                 ):

        # Actor 
        self.actor = MLPActor(state_dim=state_dim, action_dim=action_dim, device=device, max_action=max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.diffusion = KarrasDenoiser(action_dim=action_dim, 
                                        sigma_max=sigma_max,
                                        device=device,
                                        sampler=sampler,)
        
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        # 对 action 进行归一化 但是 metadrive 这里本身的action [-1,1]^2 导致这里不需要归一化
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

        # 训练 step 以及相应 ema 指数平滑        
        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        
        # 注意这里 critic 里面含有两个q
        self.critic = MLPCriticTwin(state_dim, action_dim, num_classes).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)
        
        self.hlgauss_loss = HLGuassLoss(
      							v_min=v_min,
								v_max=v_max,
								num_bins=num_classes,
								sigma= sigma_frac * (v_max-v_min) / num_classes,
								v_expand=v_expand,
        						v_expand_mode=v_expand_mode
        						)

        # 对原本进行cpql的参数进行保存
        self.loss_type = loss_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.alpha = alpha  # bc weight
        self.eta = eta  # q_learning weight
        self.expectile = expectile
        self.device = device
        self.max_q_backup = max_q_backup
        self.q_mode = q_mode
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_classes = num_classes
        

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.actor_target, self.actor)
        self.ema.update_model_average(self.critic_target, self.critic)
       

    def train(self, 
              replay_buffer, 
              batch_size=100,
              reset_flag=False, 
              ):

        if reset_flag == True:
            # Actor 
            self.actor = MLPActor(state_dim=self.state_dim, action_dim=self.action_dim, device=device, max_action=self.max_action).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            
            # Critic
            self.critic = MLPCriticTwin(self.state_dim, self.action_dim, self.num_classes).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        log_dict = {}
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
    
    
        """ Q Training """
        # current_q1, current_q2 = self.critic(state, action)
        if self.q_mode == 'q':
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.diffusion.sample(model=self.actor, state=next_state_rpt)
                target_logit_q1, target_logit_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                
                # hg loss 部分
                target_prob_q1 = torch.softmax(target_logit_q1, dim=-1)
                target_prob_q2 = torch.softmax(target_logit_q2, dim=-1)
                target_q1 = self.hlgauss_loss.transform_from_probs(target_prob_q1) # [bs, 1]
                target_q2 = self.hlgauss_loss.transform_from_probs(target_prob_q2) # [bs, 1]
                
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.diffusion.sample(model=self.actor, state=next_state)
                
                # Compute the target Q value
                target_logit_q1, target_logit_q2 = self.critic(next_state, next_action) # [bs, n_classes]
                target_prob_q1 = torch.softmax(target_logit_q1, dim=-1)
                target_prob_q2 = torch.softmax(target_logit_q2, dim=-1)
                target_q1 = self.hlgauss_loss.transform_from_probs(target_prob_q1) # [bs, 1]
                target_q2 = self.hlgauss_loss.transform_from_probs(target_prob_q2) # [bs, 1]
                target_q = torch.min(target_q1, target_q2).unsqueeze(-1) # [bs, 1]
                
		
            target_q = (reward + not_done * self.discount * target_q).detach() 
            
            target_q = self.hlgauss_loss.transform_to_probs(target_q) # [bs, num_classes]
            
            # Get current Q estimates 
            # Note: critic 产生的结果需要使用 softmax 处理
            logit_q1, logit_q2 = self.critic(state, action) # [256, 101]
            probs1, probs2 = torch.softmax(logit_q1, dim=-1), torch.softmax(logit_q2, dim=-1) # [256, 101]
            current_q1 = self.hlgauss_loss.transform_from_probs(probs1) # [256]
            current_q2 = self.hlgauss_loss.transform_from_probs(probs2)

            # Compute critic loss
            critic_loss = F.cross_entropy(logit_q1, target_q).mean() + F.cross_entropy(logit_q2, target_q).mean()
    
        elif self.q_mode == 'q_v':
            def expectile_loss(diff, expectile=0.8):
                weight = torch.where(diff > 0, expectile, (1 - expectile))
                return weight * (diff**2)
            
            with torch.no_grad():
                q = self.critic.q_min(state, action)
            v = self.critic.v(state)
            value_loss = expectile_loss(q - v, self.expectile).mean()

            current_q1, current_q2 = self.critic(state, action)
            with torch.no_grad():
                next_v = self.critic.v(next_state)
            target_q = (reward + not_done * self.discount * next_v).detach()
        
            critic_loss = value_loss + F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) 
                
                
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=float('inf'), norm_type=2)
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            
        self.critic_optimizer.step()
        

        """ Policy Training """
        compute_bc_losses = functools.partial(self.diffusion.consistency_losses,
                                              model=self.actor,
                                              x_start=action,
                                              num_scales=40,
                                              target_model=self.actor_target,
                                              state=state,
                                              loss_type=self.loss_type)

        bc_losses = compute_bc_losses()
        bc_loss = bc_losses["loss"].mean()
        consistency_loss = bc_losses["consistency_loss"].mean()
        recon_loss = bc_losses["recon_loss"].mean()

        new_action = self.diffusion.sample(model=self.actor, state=state)

        # 将分类loss 进行相应的转化
        q1_new_logit, q2_new_logit = self.critic(state, new_action)
        probs1, probs2 =  torch.softmax(q1_new_logit, dim=-1),  torch.softmax(q2_new_logit, dim=-1)
        q1_new_action = self.hlgauss_loss.transform_from_probs(probs1)
        q2_new_action = self.hlgauss_loss.transform_from_probs(probs2)
                
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        
        # 本来期望在这里使用 td3-bc 里的更新范式后续进行实验发现效果很一般
        # q_loss = -q1_new_action.mean() / q1_new_action.abs().mean().detach() 
        actor_loss = self.alpha * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
    
        actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=float('inf'), norm_type=2)
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
           
        self.actor_optimizer.step()

        """ Step Target network """
        if self.step % self.update_ema_every == 0:
            self.step_ema()
            

        self.step += 1
        
        log_dict["actor_loss"] = actor_loss.item()
        log_dict["bc_loss"] = bc_loss.item()
        log_dict["ql_loss"] = q_loss.item()
        log_dict["critic_loss"] = critic_loss.item()
        log_dict["critic_grad_norms"] = critic_grad_norms
        log_dict["actor_grad_norms"] = actor_grad_norms
        log_dict["Q_value"] = torch.mean(current_q1).item()

        if self.lr_decay: 
            self.actor_lr_scheduler.step()

        return log_dict

    def select_action(self, state, num=10):
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.diffusion.sample(model=self.actor, state=state_rpt)
            logit_q_value = self.critic_target.q_min(state_rpt, action)
            q_probs = torch.softmax(logit_q_value, dim=-1) # [256, 101]
            q_value = self.hlgauss_loss.transform_from_probs(q_probs).flatten() # [50]
        
        # print("q_vaslue", q_value.shape)
        idx = torch.multinomial(F.softmax(q_value), 1)
        action = action[idx].cpu().data.numpy().flatten()

        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def save(self, dir):
        
        torch.save(self.actor.state_dict(), os.path.join(dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(dir, f'critic.pth'))
       

    def load(self, dir):
        self.actor.load_state_dict(
            torch.load(os.path.join(dir, 'actor.pth'))
            )
      
        self.critic.load_state_dict(
                torch.load(os.path.join(dir, f'critic.pth'))
                )
       


