"""Simple Agents

Complete!
"""
import numpy as np


__all__ = ['SimpleAgent']


class SimpleAgent(object):
    def __init__(self):
        self.name = 'SimpleAgent'    # Name to be submitted to OpenAI
        self.parameter = [0.5, 0.5, 0.5, 0.5]

    def act(self, observation):
        return np.sign(np.dot(self.parameter, observation))

class RandomAgent(object):
    def __init__(self):
        self.name = 'SimpleAgent'    # Name to be submitted to OpenAI
        self.parameter = [0.5, 0.5, 0.5, 0.5]

    def act(self, observation):
        return np.sign(np.dot(self.parameter, observation))
