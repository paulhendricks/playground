"""Cartpole example
"""
import gym

from playground.algos.local import SimulatedAnnealingAgent
from playground.utils import run_experiment


def main():
    env = gym.make('CartPole-v0')
    agent = SimulatedAnnealingAgent(env.action_space)
    run_experiment(env, agent)


if __name__ == '__main__':
    main()
