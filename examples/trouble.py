import gym
from playground.algos.local import SimulatedAnnealingAgent


# Make environment
env = gym.make('CartPole-v0')

# Run monte carlo agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)

episode_count = 10
max_steps = 200
reward = 0
done = False
