import gym
from playground.algos.blind import MonteCarloAgent
from playground.algos.gradient import PolicyGradientAgent
from playground.algos.local import HillClimbingAgent, SimulatedAnnealingAgent
from playground.algos.random import RandomAgent
from playground.experiment import run_experiment
import tensorflow as tf

# Make environment
env = gym.make('CartPole-v0')

# Run random agent
agent = RandomAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent)

# Run monte carlo agent
agent = MonteCarloAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result)

# Run hill climbing agent
agent = HillClimbingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result)

# Run simulated annealing agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)
result = run_experiment(env, agent, episode_count=5000, watch=False)
run_experiment(env, result)

# Run policy gradient agent
agent = PolicyGradientAgent(env.observation_space, env.action_space)
policy_grad = agent.policy_gradient()
value_grad = agent.value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
outdir = '/tmp/' + agent.name + '-results'
env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)
for _ in range(2000):
    reward = agent.run_episode(env, policy_grad, value_grad, sess)
env.monitor.close()
