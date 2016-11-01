"""Default class for agents.
"""
import numpy as np


__all__ = ['Agent']


class Agent(object):
    def __init__(self):
        pass

    def learn(self):
        raise NotImplementedError

    def act(self, observation, reward):
        raise NotImplementedError
