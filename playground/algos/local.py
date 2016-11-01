"""Local Search Agents.
"""
import numpy as np

from .agent import Agent


__all__ = ['HillClimbingAgent', 'SimulatedAnnealingAgent']


class HillClimbingAgent(Agent):
    def __init__(self, observation_space, action_space, spread=0.1):
        # Initialize agent
        self.name = 'HillClimbingAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space
        self.n = observation_space.shape[0]

        # Set agent-specific values
        self.spread = spread

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
    def __init__(self, observation_space, action_space, spread=0.1, alpha=1, decay=0.9):
        # Initialize agent
        self.name = 'SimulatedAnnealingAgent'    # Name to be submitted to OpenAI
        self.observation_space = observation_space
        self.action_space = action_space
        self.n = observation_space.shape[0]

        # Set agent-specific values
        self.spread = spread  # Spread of randomness when selecting new values to test
        self.alpha = alpha  # Learning rate
        self.decay = decay  # Decay in impact of alpha

        # Set parameter values
        self.parameters = np.zeros(self.n)
        self.best_parameters = np.zeros(self.n)

        # Set episode values
        self.episode_number = 1
        self.episode_reward = 0
        self.best_episode_reward = 0

    def update_parameters(self):
        noise = np.random.rand(self.n) * 2 - 1
        delta = noise * self.spread * self.alpha
        self.parameters = self.best_parameters + delta

    def choose_action(self, observation):
        result = np.sign(np.dot(self.parameters, np.array(observation)))
        if result == -1:
            action = 0
        else:
            action = 1
        return action

    def reset(self):
        # If score is the same as best then the amount of variance in future choices goes down
        # Set the new best to be the average of all the best scores so far (using incremental mean)
        if self.episode_reward >= self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            self.best_parameters = self.parameters
            self.alpha *= self.decay
        else:
            self.alpha /= self.decay
        self.episode_reward = 0
        self.episode_number += 1

    def act(self, observation, reward, done):
        # If new episode, choose new parameters
        if self.episode_reward == 0:
            self.episode_reward += reward
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
