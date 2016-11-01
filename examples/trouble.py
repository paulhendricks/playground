import gym
from playground.algos.local import SimulatedAnnealingAgent


# Make environment
env = gym.make('CartPole-v0')

# Run monte carlo agent
agent = SimulatedAnnealingAgent(env.observation_space, env.action_space)

outdir = '/tmp/' + agent.name + '-results'
env.monitor.start(outdir, force=True, video_callable=lambda count: count % 10 == 0)

episode_count=100
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
