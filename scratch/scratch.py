import gym
from playground.algos.random import RandomAgent
from playground.experiment import run_experiment


# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = RandomAgent(env.observation_space, env.action_space)
run_experiment(env, agent)
