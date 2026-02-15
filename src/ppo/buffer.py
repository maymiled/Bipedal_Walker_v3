import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.size = size
        self.device = device

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.logp = np.zeros((size,), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.val = np.zeros((size,), dtype=np.float32)

        self.adv = np.zeros((size,), dtype=np.float32)
        self.ret = np.zeros((size,), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, act, logp, rew, done, val):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.logp[self.ptr] = logp
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.val[self.ptr] = val
        self.ptr += 1

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        adv = 0.0
        for t in reversed(range(self.size)):
            next_non_terminal = 1.0 - self.done[t]
            next_val = last_val if t == self.size - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_val * next_non_terminal - self.val[t]
            adv = delta + gamma * lam * next_non_terminal * adv
            self.adv[t] = adv

        self.ret = self.adv + self.val

        a = self.adv
        self.adv = (a - a.mean()) / (a.std() + 1e-8)

    def get(self):
        return dict(
            obs=torch.tensor(self.obs, device=self.device),
            act=torch.tensor(self.act, device=self.device),
            logp=torch.tensor(self.logp, device=self.device),
            adv=torch.tensor(self.adv, device=self.device),
            ret=torch.tensor(self.ret, device=self.device),
            val=torch.tensor(self.val, device=self.device),
        )
