"""Default class for agents.
"""
import numpy as np


__all__ = ['Agent']


class Agent(object):
    def __init__(self):
        pass

    def act(self, observation, reward, done):
        raise NotImplementedError


class LienarAgent(Agent):
    def __init__(self):
        pass

    def update_parameters(self):
        pass

    def choose_action(self, observation):
        result = np.sign(np.dot(self.parameters, np.array(observation)))
        if result == -1:
            action = 0
        else:
            action = 1
        return action
