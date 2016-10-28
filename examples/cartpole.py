"""Cartpole example

Complete!
"""
import gym

from playground.local import SimulatedAnnealingAgent
from playground.run import run


def main():
    env = gym.make('CartPole-v0')
    agent = SimulatedAnnealingAgent(env.action_space)
    run(env, agent)


if __name__ == '__main__':
    main()
