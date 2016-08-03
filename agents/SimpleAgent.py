"""Simple Agent for Cartpole Simulation

Not Complete!
"""
import logging
import numpy as np

import gym


__all__ = ['SimpleAgent']


class SimpleAgent(object):
    def __init__(self):
        self.name = 'Simple'    # Name to be submitted to OpenAI
        self.parameter = [0.5, 0.5, 0.5, 0.5]

    def act(self, observation):
        return np.sign(np.dot(self.parameter, observation))


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    env = gym.make('CartPole-v0')
    agent = SimpleAgent()   # Initialise agent

    outdir = '/tmp/' + agent.name + '-results'
    env.monitor.start(outdir, force=True)
    env.monitor.start(outdir, force=True, video_callable=lambda count: count % 1 == 0)
    # env.monitor.configure(video_callable=lambda count: False)

    episode_count = 2000
    max_steps = 200
    # reward = 0
    # done = False

    temp = list()
    for i in xrange(episode_count):
        observation = env.reset()

        for j in xrange(max_steps):
            # print(ob)
            import time
            time.sleep(0.1)  # delays for 5 seconds
            action = agent.act(observation)
            # print(action)
            # print(agent.best, agent.alpha, agent.best_score, agent.best_count)
            observation, reward, done, _ = env.step(action)
            if done:
                temp.append(agent)
                break
    # print(agent.best)
    print(len(temp))
    # Dump result info to disk
    env.monitor.close()

if __name__ == '__main__':
    main()
