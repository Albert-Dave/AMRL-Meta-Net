import gym, bsuite

def make_env(name):
    if name.startswith("bsuite"):
        return bsuite.load(name.split("/")[1])
    return gym.make(name)
