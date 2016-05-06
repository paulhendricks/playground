"""TO BE EDITED

Not Complete!
"""
import logging
import random

import gym


random.seed(0)   # So scores are (slightly) more consistent. Randomness in pole counteracts this a bit


class SimulatedAnnealingAgent(object):
    def __init__(self, action_space, repeats=10, alpha=1, decay=0.9, spread=0.5):
        self.name = 'SimAnn'    # Name to be submitted to OpenAI
        self.action_space = action_space  # Just for consistency with other agents, not used in this case

        self.alpha = alpha  # Learning rate
        self.decay = decay  # Decay in impact of alpha
        self.spread = spread  # Spread of randomness when selecting new values to test
        self.repeats = repeats  # Number of times to repeat testing a value

        self.obs_count = 0  # Number of observation returned (can probably get from the environment somehow)
        self.best = []  # Holds best values (set on first run of action)
        self.test = []  # Holds test values

        self.best_score = 0  # Current max score found
        self.best_count = 0  # Times hit max score (used for bounded problems)
        self.ep_score = 0    # Total score for episode
        self.repeat_count = 0  # Times repeated running test

    # Set the new test values at the start of the episode
    def set_test(self):
        # If less than required repeats than just run again
        if self.repeat_count < self.repeats:
            self.repeat_count += 1
            return self.test

        # Else reset repeat count and set new values based on current best, spread and alpha
        self.repeat_count = 0
        return [self.best[i] + (random.random() - self.spread) * self.alpha for i in range(self.obs_count)]

    # Choose action based on observed values
    def choose_action(self, observation):
        if sum(observation[i] * self.test[i] for i in range(self.obs_count)) > 0:
            return 1

        return 0

    # If get the same ep score then update best to average of all values that have reached the best score
    def update_best(self):
        self.best = [(self.best[i] * self.best_count + self.test[i]) / (self.best_count + 1) for i in range(self.obs_count)]
        self.best_count += 1

    # What gets called
    def act(self, observation, reward, done):

        # Set initial values if first time agent is seeing observations
        if self.obs_count == 0:
            self.obs_count = len(observation)
            self.best = [0] * self.obs_count
            self.test = self.best

        # Set new test values for new episode
        if self.ep_score == 0:
            self.test = self.set_test()

        # Select action
        action = self.choose_action(observation)

        # Update episode score
        self.ep_score += reward

        if done:
            # If score is the same as best then the amount of variance in future choices goes down
            # Set the new best to be the average of all the best scores so far (using incremental mean)
            if self.ep_score == self.best_score:
                self.alpha *= self.decay
                self.update_best()

            # If new score is greater then set everything to that
            elif self.ep_score > self.best_score:
                self.best_score = self.ep_score
                self.best = self.test
                self.best_count = 0
                self.alpha *= self.decay

            # If new score isn't >= then increase the spread when selecting values
            # This helps get around issues making incorrect starting decisions but can probably be improved
            else:
                self.alpha /= self.decay

            self.ep_score = 0

        return action


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    env = gym.make('CartPole-v0')
    agent = SimulatedAnnealingAgent(env.action_space, repeats=1, decay=0.9, spread=0.3)   # Initialise agent

    outdir = '/tmp/' + agent.name + '-results'
    env.monitor.start(outdir, force=True)

    episode_count = 200
    max_steps = 200
    reward = 0
    done = False

    for i in xrange(episode_count):
        ob = env.reset()

        for j in xrange(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # Dump result info to disk
    env.monitor.close()

if __name__ == '__main__':
    main()