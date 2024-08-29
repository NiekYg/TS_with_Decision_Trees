import numpy as np
import torch
import pandas as pd
import logging
import json
import xgboost as xgb

from base_agent import Agent
from sklearn.ensemble import RandomForestRegressor

class BernoulliXGBoostTSAgent(Agent):
    """
    An agent for Bernoulli rewarded bandits.

    XGBoost with TS exploration based on number of samples in leaves.
    """

    def __init__(self, env_constructor, context_size,
                use_cuda=False, exploration_variance=1.0,
                max_depth=10, n_estimators=100,
                eta=0.3, gamma=100.0, xgb_lambda=1.0,
                initial_random_selections=10, base_score=1):
        self.t = 1
        self.context_size = context_size

        # Set up the internal environment
        self.internal_env = env_constructor()

        ############### XGBoost parameters ###############
        self.exploration_variance = exploration_variance
        self.use_cuda = use_cuda
        self.x_train = None
        self.reward_theta = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.lr = eta

        self.initial_random_selections = initial_random_selections

        self.base_score = base_score

        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 'objective': 'reg:squarederror', 'base_score': base_score,
                                 'max_depth': max_depth, 'gamma': gamma, 'learning_rate': eta, 'reg_lambda': xgb_lambda, 'min_child_weight': 2}

        Agent.__init__(self)

    def train_model(self, observation, action, reward):
        """Train XGBoost approximator."""
        # Here, observation[action] is a NumPy array (the context)
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

        # Convert the training data to DMatrix format for XGBoost
        Xy = xgb.DMatrix(self.x_train.T, label=self.reward_theta)

        # Rebuild trees after an increasingly number of time steps
        if self.t >= self.initial_random_selections and np.ceil(8*np.log(self.t)) > np.ceil(8*np.log(self.t - 1)):
            self.model = xgb.train(params=self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)
            print(f't = {self.t} - re-trained model')

            leaf_scores = self.model.get_dump(with_stats=True, dump_format='json')
            X = xgb.DMatrix(self.x_train.T)
            residuals_array = np.array(self.reward_theta - self.base_score, dtype="float64")

            json_trees = [json.loads(leaf_scores[i]) for i in range(self.n_estimators)]
            self.leaves_per_tree = []

            for idx, tree in enumerate(json_trees):
                self.leaves_per_tree.append(self.get_leaves(tree))
                pred = self.model.predict(X, iteration_range=(0, idx + 1)).reshape((len(self.reward_theta), 1))
                residuals_array = np.append(residuals_array, self.reward_theta - pred, axis=1)

            individual_preds = self.model.predict(X, pred_leaf=True)
            leaf_of_data = np.array(individual_preds)

            for i in range(len(self.leaves_per_tree)):
                df2 = pd.DataFrame({'leaf': leaf_of_data[:, i], 'residual': residuals_array[:, i]})
                group2 = df2.groupby(by='leaf').agg(['count', 'mean', 'var'])
                for row_idx, row in group2.iterrows():
                    self.leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                    self.leaves_per_tree[i][row_idx]["leaf_count"] = row['residual']['count']
                    self.leaves_per_tree[i][row_idx]["leaf_mean"] = row['residual']['mean'] * self.lr
                    self.leaves_per_tree[i][row_idx]["leaf_variance"] = row['residual']['var'] * self.lr**2
        elif self.t >= self.initial_random_selections:
            X = xgb.DMatrix(x.T)
            leaf_preds = np.array(self.model.predict(X, pred_leaf=True))

            residual = y - self.base_score
            for i in range(len(self.leaves_per_tree)):
                pred = self.model.predict(X, iteration_range=(0, i + 1))
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] = (
                    (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_mean"] *
                    self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] +
                    residual * self.lr) / (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] + 1))[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] = (
                    (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_variance"] *
                    (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] - 1) +
                    (pred - residual) ** 2 * self.lr ** 2) /
                    (self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"]))[0][0]
                self.leaves_per_tree[i][leaf_preds[0][i]]["leaf_count"] += 1

        return

    def predict(self, x):
        """Predict reward."""

        if not hasattr(self, 'model'):
            raise ValueError("The XGBoost model is not trained yet.")

        X = xgb.DMatrix(x.T)
        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_assignments = np.array(individual_preds)

        mean_per_assigned_leaf = np.array(
            [self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_mean"] for i in range(len(self.leaves_per_tree))])

        mu_hat = self.base_score + np.sum(mean_per_assigned_leaf)

        variance_of_mean_per_assigned_leaf = np.array(
            [self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_variance"] /
             self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_count"] for i in range(len(self.leaves_per_tree))])
        total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

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
        """Adds observation to training data and updates model"""
        self.train_model(observation, action, reward)


    def pick_action(self, observation):
        """Selects the next arm to pull"""

        if self.t < self.initial_random_selections:
            arm_selection = self.internal_env.pick_random_arm()
            # Extract the label (arm index) from the SimpleNamespace
            arm_label = arm_selection.label
            # Train the model after accumulating data points
            self.train_model(observation, arm_label, np.random.rand())  # Random reward as a placeholder
        else:
            # If the model is not trained, force it to train
            if not hasattr(self, 'model'):
                arm_label = list(observation.keys())[0]  # Use the first key (arm) as the dummy action
                self.train_model(observation, arm_label, np.random.rand())  # Train with dummy action and reward

            # Proceed with normal sample collection and action selection
            samples = self.get_samples(observation, self.t)
            self.internal_env.overwrite_arm_weight(samples)
            arm_selection = self.internal_env.get_optimal_action()

        self.t += 1
        return arm_selection  # return the SimpleNamespace object directly


    def get_leaves(self, tree: dict):
        """Get all leaves of a tree.

        Parameters
        ----------
        tree : dict
            Tree.

        Returns
        -------
        leaves : dict
            Dist of leaves.
        """

        leaves = {}
        stack = [tree]
        while stack:
            node = stack.pop()
            try:
                stack.append(node['children'][0])
                stack.append(node['children'][1])
            except:
                leaves[node['nodeid']] = node

        return leaves
