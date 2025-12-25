"""
Optuna hyperparameter search for AMRL initialization.
Used to select initial (alpha, gamma, beta) before meta-learning.
"""

import optuna
import gym
import torch
from rl.ppo import PPOAgent
from utils.seed import set_seed

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    beta  = trial.suggest_float("beta", 0.0, 0.05)

    env = gym.make("CartPole-v1")
    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        (alpha, gamma, beta)
    )

    total_reward = 0.0
    state = env.reset()

    for _ in range(1000):  # short, Colab-safe
        s = torch.tensor(state, dtype=torch.float32)
        action, _ = agent.act(s)
        state, reward, done, _ = env.step(action.item())
        total_reward += reward
        if done:
            break

    return total_reward


def main():
    set_seed(42)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
