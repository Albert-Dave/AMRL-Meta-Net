import torch
from envs.make_env import make_env
from amrl.meta_state import MetaState
from amrl.meta_reward import MetaReward
from amrl.meta_learner import MetaLearner
from amrl.hyperparameter_controller import HPController
from rl.ppo import PPOAgent
from utils.seed import set_seed

set_seed(42)

envs = ["CartPole-v1", "MountainCar-v0", "bsuite/memory_len"]
meta = MetaLearner()

for env_name in envs:
    env = make_env(env_name)
    ms, mr = MetaState(), MetaReward()
    hp = HPController()

    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hp.get()
    )

    s = env.reset()
    h = None

    for t in range(30000):
        s_t = torch.tensor(s, dtype=torch.float32)
        a, _ = agent.act(s_t)
        s2, r, d, _ = env.step(a.item())

        loss, gn = agent.update(s_t, a, r)
        ms.update(r, loss, loss, gn)
        mr.update(r)

        if t > 5000 and t % 5000 == 0:
            delta, h = meta(ms.get().view(1,1,-1), h)
            hp.update(delta.detach().numpy()[0])

        s = s2 if not d else env.reset()

print("Meta-training completed.")
