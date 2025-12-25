"""
evaluate.py
Evaluation script for AMRL experiments.
Computes mean ± std and logs results.
"""

import torch
import argparse
import gym
import numpy as np
from rl.ppo import PPOAgent
from utils.seed import set_seed
from utils.logger import Logger
from utils.stats import mean_std

# --------------------------------------------------
# Run evaluation episodes
# --------------------------------------------------
def evaluate_env(env_name, episodes=5):
    env = gym.make(env_name)

    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        (3e-4, 0.99, 0.01)
    )

    scores = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            s = torch.tensor(state, dtype=torch.float32)
            action, _ = agent.act(s)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
            if done:
                break

        scores.append(total_reward)

    return scores


# --------------------------------------------------
# Main entry point (required for colab_run.py)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Environment to evaluate"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5
    )
    args = parser.parse_args()

    set_seed(42)
    logger = Logger(filename="evaluation_log.csv")

    scores = evaluate_env(args.env, args.episodes)
    mean, std = mean_std(scores)

    logger.write({
        "environment": args.env,
        "mean_reward": f"{mean:.2f}",
        "std_reward": f"{std:.2f}",
    })

    logger.close()

    print(f"\nEvaluation results for {args.env}")
    print(f"Mean reward: {mean:.2f}")
    print(f"Std reward:  {std:.2f}")


if __name__ == "__main__":
    main()
