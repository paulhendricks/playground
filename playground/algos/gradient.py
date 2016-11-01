"""Policy Gradient Search Agents.
"""
from .agent import Agent

__all__ = ['PolicyGradientAgent']


class PolicyGradientAgent(Agent):
    def __init__(self, observation_space, action_space):
        self.name = 'PolicyGradientgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space

    def learn(self):
        pass

    def act(self, observation, reward):
        return self.action_space.sample()
