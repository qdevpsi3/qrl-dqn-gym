import gym
import gym.spaces
import numpy as np


class BinaryWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(BinaryWrapper, self).__init__(env)
        self.bits = int(np.ceil(np.log2(env.observation_space.n)))
        self.observation_space = gym.spaces.MultiBinary(self.bits)

    def observation(self, obs):
        binary = map(float, "{0:b}".format(int(obs)).zfill(self.bits))
        return np.array(list(binary))
