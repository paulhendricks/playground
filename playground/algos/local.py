"""Local Search Agents.
"""
import numpy as np
import random

from .agent import Agent


__all__ = ['HillClimbingAgent', 'SimulatedAnnealingAgent']


class HillClimbingAgent(Agent):
    def __init__(self, observation_space, action_space, spread=0.1):
        self.name = 'HillClimbingAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space
        self.n = observation_space.shape[0]
        self.parameters = np.random.rand(self.n) * 2 - 1
        self.episode_reward = 0
        self.episode_number = 1
        self.spread = spread

        # Set best values
        self.best_parameters = np.zeros(self.n)
        self.best_episode_reward = 0


    def choose_action(self, observation):
        result = np.sign(np.dot(self.parameters, np.array(observation)))
        if result == -1:
            action = 0
        else:
            action = 1
        return action

    def update_parameters(self):
        noise = np.random.rand(self.n) * 2 - 1
        delta = noise * self.spread
        self.parameters = self.best_parameters + delta

    def reset(self):
        # Check if episode_reward > best_episode_reward
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            self.best_parameters = self.parameters


        # Reset episode reward
        self.episode_reward = 0
        self.episode_number += 1

    def act(self, observation, reward, done):
        # If new episode, choose new parameters
        if self.episode_reward == 0:
            self.update_parameters()

        # Increment reward
        self.episode_reward += reward

        # Choose an action
        action = self.choose_action(observation)

        # Check if previous episode terminated and clean up for new episode
        if done:
            self.reset()

        # Return action
        return action


class SimulatedAnnealingAgent(Agent):
    def __init__(self, observation_space, action_space, repeats=10, alpha=1, decay=0.9, spread=0.1):
        self.name = 'SimulatedAnnealingAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space
        self.n = observation_space.shape[0]

        self.alpha = alpha  # Learning rate
        self.decay = decay  # Decay in impact of alpha
        self.spread = spread  # Spread of randomness when selecting new values to test
        self.repeats = repeats  # Number of times to repeat testing a value

        self.obs_count = 0  # Number of observation returned (can probably get from the environment somehow)
        self.best = np.zeros(self.n)  # Holds best values (set on first run of action)
        self.test = np.zeros(self.n)  # Holds test values

        self.best_score = 0  # Current max score found
        self.best_count = 0  # Times hit max score (used for bounded problems)
        self.episode_reward = 0    # Total score for episode
        self.repeat_count = 0  # Times repeated running test

    # Set the new test values at the start of the episode
    def update_parameters(self):
        # If less than required repeats than just run again
        if self.repeat_count < self.repeats:
            self.repeat_count += 1
            return self.test

        # Else reset repeat count and set new values based on current best, spread and alpha
        self.repeat_count = 0
        # (random.random() - self.spread)
        # random.gauss(0, 0.1)
        self.test = [self.best[i] + (random.random() - self.spread) * self.alpha for i in range(self.obs_count)]

    # Choose action based on observed values
    def choose_action(self, observation):
        if sum(observation[i] * self.test[i] for i in range(self.obs_count)) > 0:
            return 1
        return 0

    # If get the same ep score then update best to average of all values that have reached the best score
    def update_best(self):
        self.best = [(self.best[i] * self.best_count + self.test[i]) / (self.best_count + 1) for i in range(self.obs_count)]
        self.best_count += 1

    def reset(self):
        # If score is the same as best then the amount of variance in future choices goes down
        # Set the new best to be the average of all the best scores so far (using incremental mean)
        if self.episode_reward == self.best_score:
            self.alpha *= self.decay
            self.update_best()

        # If new score is greater then set everything to that
        elif self.episode_reward > self.best_score:
            self.best_score = self.episode_reward
            self.best = self.test
            self.best_count = 0
            self.alpha *= self.decay

        # If new score isn't >= then increase the spread when selecting values
        # This helps get around issues making incorrect starting decisions but can probably be improved
        else:
            self.alpha /= self.decay

        self.episode_reward = 0

    def act(self, observation, reward, done):
        # If new episode, choose new parameters
        if self.episode_reward == 0:
            self.episode_reward += 1
            self.update_parameters()

        # Increment reward
        self.episode_reward += reward

        # Choose an action
        action = self.choose_action(observation)

        # Check if previous episode terminated and clean up for new episode
        if done:
            self.reset()

        # Return action
        return action
