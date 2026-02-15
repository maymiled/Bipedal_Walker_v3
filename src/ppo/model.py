import math
import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        nn.init.orthogonal_(self.trunk[0].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.trunk[2].weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)

        nn.init.constant_(self.trunk[0].bias, 0.0)
        nn.init.constant_(self.trunk[2].bias, 0.0)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.constant_(self.v_head.bias, 0.0)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        v = self.v_head(h).squeeze(-1)
        return mu, v

    def mu_std_value(self, obs):
        mu, v = self.forward(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std, v


def gaussian_log_prob(u, mu, std):
    return (
        -0.5 * (((u - mu) / (std + 1e-8)) ** 2)
        - torch.log(std + 1e-8)
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)


def sample_action_and_logp(mu, std):
    u = mu + std * torch.randn_like(mu)
    a = torch.tanh(u)
    logp_u = gaussian_log_prob(u, mu, std)
    logp = logp_u - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    return a, logp


def logp_from_action(a, mu, std):
    a = torch.clamp(a, -0.999, 0.999)
    u = 0.5 * torch.log((1 + a) / (1 - a))
    logp_u = gaussian_log_prob(u, mu, std)
    logp = logp_u - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    return logp


def gaussian_entropy(std):
    return (
        0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std + 1e-8)
    ).sum(-1)
