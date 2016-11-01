import gym
from playground.algos.blind import MonteCarloAgent
from playground.algos.local import HillClimbingAgent, SimulatedAnnealingAgent
from playground.algos.random import RandomAgent
from playground.experiment import run_experiment

# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = RandomAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent)

# Run monte carlo agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=1000, watch=False)
run_experiment(env, result)

# Run hill climbing agent
agent = HillClimbingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=1000, watch=False)
run_experiment(env, result)

# Run simulated annealing agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
print(result.best_parameters)
result.best_episode_reward
run_experiment(env, result)

outdir = '/tmp/' + agent.name + '-results'
env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)

episode_count=1
max_steps=200
reward = 0
done = False
for _ in range(episode_count):
    ob = env.reset()
    for _ in range(max_steps):
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break

# Dump result info to disk
env.monitor.close()