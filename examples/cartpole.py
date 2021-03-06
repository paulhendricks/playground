import gym
from playground.algos.blind import MonteCarloAgent
from playground.algos.gradient import PolicyGradientAgent
from playground.algos.local import HillClimbingAgent, SimulatedAnnealingAgent
from playground.algos.random import RandomAgent
from playground.experiment import run_experiment

# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = RandomAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=1000)

# Run monte carlo agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result, episode_count=1000)

# Run hill climbing agent
agent = HillClimbingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result, episode_count=1000)

# Run simulated annealing agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result, episode_count=1000)

# Run policy gradient agent
agent = PolicyGradientAgent(env.observation_space, env.action_space)
outdir = '/tmp/' + agent.name + '-results'
env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)
for _ in range(2000):
    agent.run_episode(env)
