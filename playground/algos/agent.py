"""Default class for agents.
"""


__all__ = ['Agent']


class Agent(object):
    def __init__(self):
        pass

    def act(self):
        raise NotImplementedError
