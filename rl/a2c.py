import torch
import torch.nn as nn
from torch.distributions import Categorical
from rl.networks import MLP

class A2CAgent:
    def __init__(self, obs_dim, act_dim, hp):
        self.actor = MLP(obs_dim, act_dim)
        self.critic = MLP(obs_dim, 1)

        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=hp[0]
        )

        self.gamma = hp[1]
        self.beta = hp[2]

    def act(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def update(self, s, a, r, s2, done):
        value = self.critic(s)
        next_value = self.critic(s2).detach()

        td_target = r + self.gamma * next_value * (1 - done)
        advantage = td_target - value

        dist = Categorical(logits=self.actor(s))
        logp = dist.log_prob(a)

        policy_loss = -(logp * advantage.detach())
        value_loss = advantage.pow(2)
        entropy = dist.entropy()

        loss = policy_loss + value_loss - self.beta * entropy

        self.opt.zero_grad()
        loss.mean().backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), 1.0
        )
        self.opt.step()

        return policy_loss.item(), value_loss.item(), grad_norm.item()
