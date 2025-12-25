import torch
import torch.nn as nn
from torch.distributions import Categorical
from rl.networks import MLP

class PPOAgent:
    def __init__(self, obs_dim, act_dim, hp):
        self.pi = MLP(obs_dim, act_dim)
        self.v  = MLP(obs_dim, 1)
        self.opt = torch.optim.Adam(
            list(self.pi.parameters()) + list(self.v.parameters()),
            lr=hp[0]
        )
        self.gamma, self.beta = hp[1], hp[2]

    def act(self, s):
        dist = Categorical(logits=self.pi(s))
        a = dist.sample()
        return a, dist.log_prob(a)

    def update(self, s, a, r):
        ret = torch.tensor(r, dtype=torch.float32)
        v = self.v(s).squeeze()
        adv = ret - v.detach()

        dist = Categorical(logits=self.pi(s))
        lp = dist.log_prob(a)
        loss = -(lp * adv) + (ret - v).pow(2) - self.beta * dist.entropy()

        self.opt.zero_grad()
        loss.mean().backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt.step()
        return loss.item(), gn.item()
