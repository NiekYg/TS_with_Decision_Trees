import numpy as np
import torch
import pandas as pd
import logging
import json

from base_agent import Agent
from sklearn.ensemble import RandomForestRegressor

class ThompsonSamplingNoContext:
    def __init__(self, nchoices, beta_prior=(1,1)):
        """
        Initialize the ThompsonSampling model without context.
        :param nchoices: int, the number of actions
        :param beta_prior: tuple, the prior beta distribution
        """
        self.nchoices = nchoices
        self.beta_prior = beta_prior
        self.successes = np.full(nchoices, beta_prior[0])
        self.failures = np.full(nchoices, beta_prior[1])

    def fit(self, _, actions, rewards):
        """
        Update the beta distributions based on the observed rewards.
        """
        for i in range(self.nchoices):
            self.successes[i] = np.sum(rewards[actions == i] == 1) + self.beta_prior[0]
            self.failures[i] = np.sum(rewards[actions == i] == 0) + self.beta_prior[1]

    def pick_action(self, X=None):
        """
        Sample from the beta distributions to choose the actions.
        X is not used in this case, as this is context-free TS.
        """
        samples = np.zeros(self.nchoices)
        for i in range(self.nchoices):
            samples[i] = np.random.beta(self.successes[i], self.failures[i])
        return np.argmax(samples)

    def update(self, action, reward):
        """
        Update the success and failure counts for the selected action.
        :param action: int, the action (arm) that was chosen.
        :param reward: int, the observed reward (1 for success, 0 for failure).
        """
        if reward == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1
