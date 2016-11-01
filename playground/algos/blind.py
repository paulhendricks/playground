"""Blind Search Agents.
"""
import numpy as np

from .agent import Agent


__all__ = ['MonteCarloAgent']


class MonteCarloAgent(Agent):
    def __init__(self, observation_space, action_space):
        self.name = 'MonteCarloAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space
        self.n = observation_space.shape[0]

        # Set parameter values
        self.parameters = np.zeros(self.n)
        self.best_parameters = np.zeros(self.n)

        # Set episode values
        self.episode_number = 1
        self.episode_reward = 0
        self.best_episode_reward = 0

    def choose_action(self, observation):
        result = np.sign(np.dot(self.parameters, np.array(observation)))
        if result == -1:
            action = 0
        else:
            action = 1
        return action

    def update_parameters(self):
        self.parameters = np.random.rand(self.n) * 2 - 1

    def learn(self):
        # Check if episode_reward > best_episode_reward
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            self.best_parameters = self.parameters

        # Update parameters
        self.update_parameters()

        # Reset episode reward
        self.episode_reward = 0
        self.episode_number += 1

    def act(self, observation, reward, done):
        # Check if previous episode terminated and clean up for new episode
        if done:
            self.learn()

        # Increment reward
        self.episode_reward += reward

        # Choose an action
        action = self.choose_action(observation)

        # Return action
        return action
