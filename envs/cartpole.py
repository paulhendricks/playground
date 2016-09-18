"""Runner

Not Complete!
"""
import logging
import os
import sys

import gym

if os.getcwd() == '/Users/paulhendricks/Projects/playground':
    sys.path.extend([os.path.abspath('./agents')])
else:
    sys.path.extend([os.path.abspath('../../agents')])

from local import FullBlindAgent
from local import GridAgent
from local import MonteCarloAgent
from blind import HillClimbingAgent
from blind import SimulatedAnnealingAgent
from blind import TabuAgent


def runner(Agent):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    env = gym.make('CartPole-v0')
    agent = Agent(env.action_space, repeats=1, decay=0.99, spread=0.1)   # Initialise agent

    outdir = '/tmp/' + agent.name + '-results'
    env.monitor.start(outdir, force=True)
    env.monitor.start(outdir, force=True, video_callable=lambda count: count % 50 == 0)
    # env.monitor.configure(video_callable=lambda count: False)

    episode_count = 200
    max_steps = 200
    reward = 0
    done = False


    temp = list()
    for i in xrange(episode_count):
        ob = env.reset()

        for j in xrange(max_steps):
            # print(ob)
            # import time
            # time.sleep(0.1)  # delays for 5 seconds
            action = agent.act(ob, reward, done)
            # print(action)
            # print(agent.best, agent.alpha, agent.best_score, agent.best_count)
            ob, reward, done, _ = env.step(action)
            if done:
                temp.append((agent))
                break
    print(agent.best)
    print(len(temp))
    # Dump result info to disk
    env.monitor.close()


if __name__ == '__main__':
    runner(HillClimbingAgent)
    runner(MonteCarloAgent)
    runner(SimulatedAnnealingAgent)
