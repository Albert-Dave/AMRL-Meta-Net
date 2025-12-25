ENVIRONMENTS = [
    "CartPole-v1",
    "MountainCar-v0",
    "bsuite/memory_len",
    "bsuite/catch"
]

for env_name in ENVIRONMENTS:
    env = make_env(env_name)

    run_single_env_training(
        env,
        shared_meta_learner,
        meta_optimizer,
        steps=50000   
    )
