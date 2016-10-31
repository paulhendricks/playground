import gym
from playground.algos.blind import GridSearchAgent, MonteCarloAgent
from playground.algos.local import HillClimbingAgent, SimulatedAnnealingAgent
from playground.algos.random import RandomAgent
from playground.experiment import run_experiment

# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = RandomAgent(env.observation_space, env.action_space)
run_experiment(env, agent)

# Run grid search agent
agent = GridSearchAgent(env.observation_space, env.action_space)
run_experiment(env, agent)

# Run monte carlo agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
run_experiment(env, agent)

# Run hill climbing agent
agent = HillClimbingAgent(env.observation_space, env.action_space)
run_experiment(env, agent)

# Run simulated annealing agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)
run_experiment(env, agent)
