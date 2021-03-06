"""Random Agents.
"""
from .agent import Agent

__all__ = ['RandomAgent']


class RandomAgent(Agent):
    def __init__(self, observation_space, action_space):
        self.name = 'RandomAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space

    def learn(self):
        pass

    def act(self, observation, reward):
        return self.action_space.sample()
