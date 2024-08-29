import numpy as np
import torch
import pandas as pd
import logging
import json

from base_agent import Agent
from sklearn.ensemble import RandomForestRegressor

class BernoulliRandomForestTSAgent(Agent):
    """
    An agent for Bernoulli rewarded bandits.
    
    Random Forest with TS exploration based on number of samples in leaves.
    """

    def __init__(self, env_constructor, context_size, 
                use_cuda=False, exploration_variance=1.0, max_depth=10,
                n_estimators=100, initial_random_selections=10):
        """
        An agent for Bernoulli bandits.
        """
        self.t = 1
        self.context_size = context_size

        # Set up the internal environment
        self.internal_env = env_constructor()

        ############### RF parameters ###############
        self.exploration_variance = exploration_variance
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        self.initial_random_selections = initial_random_selections

        print('Random Forest parameters - exploration_variance:', self.exploration_variance, 'n_estimators:', self.n_estimators)
        
        self.model_parameters = {'min_child_weight': 2}

        Agent.__init__(self)


    def train_network(self, observation, action, reward):
        """Fit Random Forest approximator."""
        # Here, action is an integer, so we directly use it to access the observation
        x = observation[action].reshape((self.context_size, 1))
        y = np.array(reward).reshape((1, 1))

        if self.x_train is not None:
            self.x_train = np.append(self.x_train, x, axis=1)
        else:
            self.x_train = x

        if self.reward_theta is not None:
            self.reward_theta = np.append(self.reward_theta, y, axis=0)
        else:
            self.reward_theta = y

        print(f"Appended x_train shape: {self.x_train.shape}")
        print(f"Appended reward_theta shape: {self.reward_theta.shape}")
        
        # Re-build trees after an increasingly number of time steps
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > np.ceil(8*np.log(self.t - 1)):
            X = self.x_train.T
            y = self.reward_theta.flatten()
            print(f"Fitting model with X shape: {X.shape} and y shape: {y.shape}")

            # Train RF model
            self.rf_model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=self.model_parameters['min_child_weight'])
            self.rf_model.fit(X, y)

            self.model_parameters['min_child_weight'] = np.ceil(8*np.log(self.t))

            logging.info('t = ' + str(self.t) + ' - re-trained model')
            print(f"Model retrained at time step {self.t}")
            leaf_of_data = self.rf_model.apply(X)

            y_weighted = y / self.n_estimators
            self.leaves_per_tree = []

            for i in range(self.n_estimators):
                self.leaves_per_tree.append({})
                df = pd.DataFrame({'leaf':leaf_of_data[:,i], 'weighted_target':y_weighted})

                group = df.groupby(by='leaf').agg(['count', 'mean', 'var'])

                for leaf_idx, row in group.iterrows():
                    self.leaves_per_tree[i][str(leaf_idx)] = dict()
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_count"] = row['weighted_target']['count']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_mean"] = row['weighted_target']['mean']
                    self.leaves_per_tree[i][str(leaf_idx)]["leaf_variance"] = row['weighted_target']['var']

        elif self.t >= self.initial_random_selections:
            X = x.T
            y_weighted = y.flatten() / self.n_estimators

            leaf_preds = self.rf_model.apply(X)

            for i in range(self.n_estimators):
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] = (
                    (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"] * 
                    self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 
                    y_weighted) / (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] + 1))
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] = (
                    (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_variance"] *
                    (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] - 1) + 
                    (y_weighted - self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_mean"])**2) / 
                    (self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"]))
                self.leaves_per_tree[i][str(leaf_preds[0][i])]["leaf_count"] += 1
        
        return

    def predict(self, x):
        """Predict reward."""
        # Check if the RF model is trained
        if not hasattr(self, 'rf_model'):
            raise ValueError("The Random Forest model is not trained yet.")

        if self.t > self.initial_random_selections:
            # Ensure that the input data x is correctly shaped as (1, context_size)
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            
            print(f"Shape of X before apply: {x.shape}")
            
            # Apply the Random Forest model directly without transposing
            leaf_assignments = self.rf_model.apply(x)
            
            # Compute mean and variance from the assigned leaves
            mean_per_assigned_leaf = np.array([
                self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_mean"] 
                for i in range(len(self.leaves_per_tree))
            ])

            mu_hat = np.sum(mean_per_assigned_leaf)
            variance_of_mean_per_assigned_leaf = np.array([
                self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_variance"] / 
                self.leaves_per_tree[i][str(leaf_assignments[0][i])]["leaf_count"] 
                for i in range(len(self.leaves_per_tree))
            ])
            total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

        else:
            mu_hat = 0.0
            total_variance_of_mean = self.exploration_variance  # Dummy value
        print(f"Predicted mean (mu_hat): {mu_hat}, Total variance: {total_variance_of_mean}")
        return mu_hat + np.sqrt(total_variance_of_mean) * np.random.randn()


    def get_samples(self, observation, t):
        samples = dict()
        for arm in self.internal_env.arm_set:
            x = observation[arm]  # Directly use the context as a NumPy array
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            pred = self.predict(x)
            samples[arm] = pred
        return samples


    def update_observation(self, observation, action, reward):
        """Adds observation to training data and updates the model."""
        self.train_network(observation, action, reward)
        return

    def pick_action(self, observation):
        """Selects the next arm to pull"""

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
            # Extract the label (arm index) from the SimpleNamespace
            arm_label = arm_selection.label
            # Train the model after accumulating data points
            self.train_network(observation, arm_label, np.random.rand())  # Random reward as a placeholder
        else:
            # If the model is not trained, force it to train
            if not hasattr(self, 'rf_model'):
                arm_label = list(observation.keys())[0]  # Use the first key (arm) as the dummy action
                self.train_network(observation, arm_label, np.random.rand())  # Train with dummy action and reward

            # Proceed with normal sample collection and action selection
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()

        self.t += 1
        return arm_selection  # return the SimpleNamespace object directly

