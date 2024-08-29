import pandas as pd, numpy as np, re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from types import SimpleNamespace
import numpy as np, warnings, ctypes
from sklearn.utils import resample, shuffle


class SimpleBanditEnv:
    def __init__(self, contexts, rewards, n_arms):
        self.contexts = contexts
        self.rewards = rewards
        self.n_arms = n_arms
        self.current_step = 0
        self.arm_set = list(range(n_arms))  # Define arms
    
    def step(self, action):
        """Returns the reward for the selected action."""
        reward = self.rewards[self.current_step % self.rewards.shape[0], action]
        self.current_step += 1
        return action, reward
    
    def pick_random_arm(self):
        """Selects a random arm (action)."""
        return SimpleNamespace(label=np.random.choice(self.arm_set))
    
    def overwrite_arm_weight(self, samples):
        """Dummy implementation to satisfy the agent's requirements."""
        pass
    
    def get_optimal_action(self):
        """Selects the arm with the highest weight."""
        return SimpleNamespace(label=np.argmax([np.random.random() for _ in self.arm_set]))
    
def parse_data(filename):
    with open(filename, "rb") as f:
        infoline = f.readline()
        infoline = re.sub(r"^b'", "", str(infoline))
        n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
        features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    features = np.array(features.todense())
    features = np.ascontiguousarray(features)
    return features, labels

def select_top_k(X, y, k=15):
    label_counts = y.sum(axis=0)
    top_k_labels = np.argsort(label_counts)[::-1][:k]
    y = y[:, top_k_labels]
    nonzero_indices = ~np.all(y == 0, axis=1)
    return X[nonzero_indices], y[nonzero_indices]

def reduce_success_rate(X, y, rate=0.1):
    X_zero = resample(X, n_samples=int(len(X) * (1 - rate)), random_state=0)
    y_zero = np.zeros((X_zero.shape[0], y.shape[1]))

    return shuffle(
        np.vstack([X, X_zero]),
        np.vstack([y, y_zero])
    )
