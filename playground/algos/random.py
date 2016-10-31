"""Random Agents.
"""

__all__ = ['RandomAgent']


class RandomAgent(object):
    def __init__(self, observation_space, action_space):
        self.name = 'RandomAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
