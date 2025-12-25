"""
train_meta.py
Multi-task AMRL meta-training (Algorithm 1)

Runs sequential meta-training across environments
and updates hyperparameters online.
"""

import torch
import argparse
from envs.make_env import make_env
from amrl.meta_state import MetaState
from amrl.meta_reward import MetaReward
from amrl.meta_learner import MetaLearner
from amrl.hyperparameter_controller import HPController
from rl.ppo import PPOAgent
from utils.seed import set_seed
from utils.logger import Logger

# --------------------------------------------------
# Core training loop for a single environment
# --------------------------------------------------
def train_on_env(env_name, meta_learner, logger, steps=30000):
    env = make_env(env_name)

    meta_state = MetaState()
    meta_reward = MetaReward()
    hp_ctrl = HPController()

    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hp_ctrl.get()
    )

    state = env.reset()
    hidden = None

    for t in range(steps):
        s = torch.tensor(state, dtype=torch.float32)

        action, _ = agent.act(s)
        next_state, reward, done, _ = env.step(action.item())

        loss, grad_norm = agent.update(s, action, reward)

        meta_state.update(reward, loss, loss, grad_norm)
        meta_reward.update(reward)

        # Meta-update
        if t > 5000 and t % 5000 == 0:
            meta_input = meta_state.get().view(1, 1, -1)
            delta, hidden = meta_learner(meta_input, hidden)
            hp_ctrl.update(delta.detach().numpy()[0])

        logger.write({
            "env": env_name,
            "step": t,
            "reward": f"{reward:.2f}",
            "loss": f"{loss:.4f}",
            "alpha": f"{hp_ctrl.a:.6f}",
            "gamma": f"{hp_ctrl.g:.2f}",
            "beta": f"{hp_ctrl.b:.2f}",
        })

        state = next_state if not done else env.reset()

    return meta_learner


# --------------------------------------------------
# Main entry point (required for colab_run.py)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["CartPole-v1", "MountainCar-v0", "bsuite/memory_len"],
        help="List of environments for meta-training"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Training steps per environment"
    )
    args = parser.parse_args()

    set_seed(42)

    logger = Logger(filename="meta_training_log.csv")
    meta_learner = MetaLearner()

    for env_name in args.envs:
        print(f"\n=== Meta-training on {env_name} ===")
        meta_learner = train_on_env(
            env_name,
            meta_learner,
            logger,
            steps=args.steps
        )

    logger.close()
    torch.save(meta_learner.state_dict(), "meta_learner.pt")
    print("\nMeta-training completed and model saved.")


if __name__ == "__main__":
    main()
