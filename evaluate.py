import gym, torch, numpy as np
from rl.ppo import PPOAgent
from utils.seed import set_seed

scores = []

for seed in [0,1,2]:
    set_seed(seed)
    env = gym.make("CartPole-v1")
    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        (3e-4, 0.99, 0.01)
    )

    s = env.reset()
    total = 0
    for _ in range(500):
        s_t = torch.tensor(s, dtype=torch.float32)
        a, _ = agent.act(s_t)
        s, r, d, _ = env.step(a.item())
        total += r
        if d:
            break
    scores.append(total)

print(f"Mean: {np.mean(scores):.2f}")
print(f"Std:  {np.std(scores):.2f}")
