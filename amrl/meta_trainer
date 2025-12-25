class MetaPPO:
    def __init__(self, policy):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        self.clip = 0.2

    def update(self, states, actions, rewards, old_logps):
        returns = compute_returns(rewards)
        new_logps = self.policy.log_prob(states, actions)

        ratio = torch.exp(new_logps - old_logps)
        clipped = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        loss = -torch.min(
            ratio * returns,
            clipped * returns
        ).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
