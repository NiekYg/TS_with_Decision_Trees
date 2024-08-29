import numpy as np
import torch
import pandas as pd
import logging
import json

class ExploreThenExploit:
    def __init__(self, nchoices, explore_rounds=2500, random_state=None):
        """
        Initialize the ExploreThenExploit model.
        
        :param nchoices: int, the number of actions (arms)
        :param explore_rounds: int, the number of rounds to explore before exploiting
        :param random_state: int or None, random seed for reproducibility
        """
        self.nchoices = nchoices
        self.explore_rounds = explore_rounds
        self.explore_cnt = 0
        self.random_state = np.random.RandomState(random_state)
        self.action_rewards = np.zeros(nchoices)
        self.action_counts = np.zeros(nchoices)
    
    def predict(self, X=None):
        """
        Predict the action (arm) to take.
        
        :param X: Not used in this context-free policy
        :return: int, the selected action (arm)
        """
        if self.explore_cnt < self.explore_rounds:
            self.explore_cnt += 1
            return self.random_state.randint(self.nchoices)
        else:
            return np.argmax(self.action_rewards / (self.action_counts + 1e-6))  # Avoid division by zero
    
    def update(self, action, reward):
        """
        Update the model with the observed reward for the taken action.
        
        :param action: int, the selected action (arm)
        :param reward: int, the observed reward (1 for success, 0 for failure)
        """
        self.action_rewards[action] += reward
        self.action_counts[action] += 1
