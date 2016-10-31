"""Blind Search Agents.
"""

__all__ = ['GridSearchAgent', 'MonteCarloAgent']


class GridSearchAgent(object):
    def __init__(self, observation_space, action_space):
        self.name = 'GridSearchAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class MonteCarloAgent(object):
    def __init__(self, observation_space, action_space):
        self.name = 'MonteCarloAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
