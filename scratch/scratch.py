import gym
from playground.algos.blind import MonteCarloAgent
from playground.experiment import run_experiment


# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
bourne = run_experiment(env, agent, episode_count=1000)
