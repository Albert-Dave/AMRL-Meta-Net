import torch
import torch.nn as nn
import random
from collections import deque
from rl.networks import MLP

class DQNAgent:
    def __init__(self, obs_dim, act_dim, hp):
        self.q = MLP(obs_dim, act_dim)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp[0])

        self.gamma = hp[1]
        self.epsilon = hp[2]

        self.buffer = deque(maxlen=50000)
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q.net[-1].out_features)
        with torch.no_grad():
            return self.q(state).argmax().item()

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.stack(s)
        s2 = torch.stack(s2)
        a = torch.tensor(a)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        qsa = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        qnext = self.q(s2).max(1)[0].detach()

        target = r + self.gamma * qnext * (1 - d)
        loss = nn.MSELoss()(qsa, target)

        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        return loss.item(), grad_norm.item()
