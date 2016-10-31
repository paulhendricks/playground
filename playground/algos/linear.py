"""Linear Agents.
"""
import numpy as np


__all__ = ['LinearAgent']


class LinearAgent(object):
    def __init__(self):
        self.name = 'LinearAgent'    # Name to be submitted to OpenAI
        self.parameter = [0.5, 0.5, 0.5, 0.5]

    def act(self, observation, reward, done):
        return np.sign(np.dot(self.parameter, observation))
