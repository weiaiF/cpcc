
import math
from builtins import int, range, super
import torch
import torch.nn as nn

from agents.helpers import SinusoidalPosEmb

# 这里的模型使用的是常规的TD3相应模型架构 两层MLP + ReLU激活函数

class MLPActor(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 max_action,
                 t_dim=16):

        super(MLPActor, self).__init__()
        self.device = device
        self.max_action = max_action

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                    #    nn.Mish(),
                                       nn.ReLU(),
                                    #    nn.Linear(256, 256),
                                    #    nn.Mish(),
                                       nn.Linear(256, 256),
                                    #    nn.Mish()
                                       nn.ReLU()
                                       )

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        if state is not None:
            x = torch.cat([x, t, state], dim=1)
        else:
            x = torch.cat([x, t], dim=1)
        
        x = self.mid_layer(x)

        # return self.final_layer(x)
        return self.max_action * torch.tanh(self.final_layer(x))

class MLPCriticTwin(nn.Module):
    def __init__(self, state_dim, action_dim, num_classes, hidden_dim=256):
        super(MLPCriticTwin, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                    #   nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, num_classes))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                    #   nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, num_classes))
        
        self.v_model  = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
    
    def v(self, state):
        return self.v_model(state)

class MLPCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MLPCritic, self).__init__()
        self.q_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                    #   nn.Linear(hidden_dim, hidden_dim),
                                    #   nn.LayerNorm(hidden_dim),
                                    #   nn.Mish(),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

        
        self.v_model  = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q_model(x)
    
    def q_min(self, state, action):
        q = self.forward(state, action)
        return q
    
    def v(self, state):
        return self.v_model(state)


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias



class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

