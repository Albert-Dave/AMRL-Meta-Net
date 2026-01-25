# AMRL-Meta-Net
Hyperparameter configuration remains a critical yet challenging issue in reinforcement learning, as effective settings depend on evolving training dynamics and task-specific characteristics. Most existing approaches rely on static, offline optimization or population-based strategies, which either fail to adapt during training or require substantial computational resources. From an expert systems perspective, hyperparameter control can be viewed as a sequential decision-making problem under uncertainty, where decisions should be continuously revised based on observed system feedback.
Adaptive Meta-Reinforcement Learning (AMRL), a resource-efficient adaptive expert decision framework for online hyperparameter control. AMRL adopts a dual-loop architecture in which a meta-level expert module, implemented as an LSTM-based meta-learner. By explicitly modeling hyperparameter adaptation as a sequential decision process, the proposed framework enables continuous, experience-driven adjustment using a single learning agent, without relying on population-based training or extensive parallel computation.

# AMRL — Adaptive Meta-Reinforcement Learning

This repository contains the official implementation of the paper:

Adaptive Meta-Reinforcement Learning for Online Hyperparameter Optimization

The codebase provides a fully reproducible, Google-Colab-ready implementation of the proposed AMRL framework, along with standard reinforcement learning baselines and benchmark environments.


# Overview

The core idea of AMRL is to adapt reinforcement learning hyperparameters online, during training, using a learned meta-controller.
Unlike static or grid-based tuning, AMRL continuously updates learning rate, discount factor, and entropy regularization based on observed learning dynamics.

Key features:
	•	Online meta-learning of hyperparameters
	•	Compatible with PPO, A2C, and DQN
	•	Supports Gym and bsuite benchmarks
	•	One-click Google Colab reproduction
	•	Deterministic, logged, and reviewer-friendly


# Algorithms Implemented

Meta-Learning
	•	AMRL (proposed)
	•	LSTM-based meta-learner
	•	PPO-style meta-optimization

Baselines
	•	PPO
	•	A2C
	•	DQN

# Hyperparameter Optimization with Optuna

Following the paper, Optuna is used for offline initialization of base hyperparameters prior to meta-learning.
	•	Optuna performs a coarse search over (α, γ, β)
	•	AMRL then adapts these parameters online
	•	Optuna is not used inside the training loop

The Optuna implementation is located at: tuning/optuna_search.py

This design mirrors standard practice in meta-RL and avoids conflating offline tuning with online adaptation.

One-Click Reproducibility (Google Colab)

To reproduce the full pipeline:
	1.	Open colab_run.ipynb in Google Colab
	2.	Click Runtime → Run all

This will:
	•	Install dependencies
	•	Run AMRL meta-training
	•	Evaluate performance
	•	Log results to CSV files

No manual configuration is required.

 # Results & Logging
	•	Training and evaluation metrics are logged as CSV files
	•	Mean and standard deviation are computed using NumPy
	•	Logs are stored in the logs/ directory
	•	All numerical values are reported with two decimal places, matching the paper

# Reproducibility
	•	Fixed random seeds
	•	Deterministic execution (where supported)
	•	CPU-only compatible
	•	No hidden state or notebook-only logic
