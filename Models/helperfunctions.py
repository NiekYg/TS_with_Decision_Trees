import pandas as pd, numpy as np, re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
import numpy as np, warnings, ctypes
from sklearn.utils import resample, shuffle
import torch
import logging
import json
import xgboost as xgb
import openml
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from contextualbandits.online import PartitionedTS
from contextualbandits.utils import _check_beta_prior, _TreeUCB_n_TS_single
from sklearn.tree import DecisionTreeClassifier


class BetaTree(DecisionTreeClassifier):
    def __init__(self, beta_prior=(1, 1), **kwargs):
        """
        Initialize the BetaTree model.

        :param beta_prior: the prior parameters of the beta distribution
        :param hdi_prob: the probability of the highest density interval (HDI)
        :param rope: the region of practical equivalence (ROPE)
        :param max_rope_overlap: the maximum overlap between the HDI and the ROPE
        :param random_state: the random state
        """
        self.beta_prior = beta_prior

        # add default parameters
        kwargs = {"criterion": "entropy", "min_samples_leaf": 10} | kwargs
        super().__init__(**kwargs)

    def _get_param_names(self):
        return DecisionTreeClassifier._get_param_names() + super()._get_param_names()

    def _get_leaf_pairs(self):
        """
        Get the beta distribution parameters for the two leaves of each split node.
        """
        stack = [(0, None)]
        leaf_pairs = defaultdict(list)

        while len(stack) > 0:
            node_id, parent_id = stack.pop()

            if self.tree_.children_left[node_id] != self.tree_.children_right[node_id]:
                # we're at a split node
                stack.append((self.tree_.children_left[node_id], node_id))
                stack.append((self.tree_.children_right[node_id], node_id))
            else:
                # we're at a leaf node
                leaf_pairs[parent_id].append(
                    tuple(
                        self.tree_.value[node_id][0][::-1]
                        * self.tree_.weighted_n_node_samples[node_id]
                    )
                )

        # remove all splits that don't lead to 2 leaf nodes
        return {
            parent_id: leaf_pairs[parent_id]
            for parent_id in leaf_pairs
            if len(leaf_pairs[parent_id]) == 2
        }

    def _hdi_overlaps_rope(
        self, beta1, beta2, rope, max_rope_overlap, n_samples=10_000
    ):
        """
        Check if the HDI of the difference between two beta distributions overlaps the ROPE.
        :param beta1: the parameters of the first beta distribution
        :param beta2: the parameters of the second beta distribution
        :param n_samples: the number of samples to draw from the beta distributions
        """
        samples1 = np.random.beta(
            self.beta_prior[0] + beta1[0], self.beta_prior[1] + beta1[1], n_samples
        )
        samples2 = np.random.beta(
            self.beta_prior[0] + beta2[0], self.beta_prior[1] + beta2[1], n_samples
        )
        

        difference = samples1 - samples2
        samples_in_rope = difference[np.abs(difference) <= rope]
        return len(samples_in_rope) / len(difference) > max_rope_overlap
    

    
    def _prune(self, rope, max_rope_overlap):
        """
        Prune the tree by removing nodes where the HDI of the difference between the two leaves overlaps the ROPE.
        """
        #print(f"Pruning with ROPE: {rope}, Max Overlap: {max_rope_overlap}")
        if self.tree_.max_depth <= 2:
            return 0

        nodes_pruned = 0
        leaf_pairs = self._get_leaf_pairs()
        for parent_id in leaf_pairs:
            #print(f"Checking pruning for parent node {parent_id}...")
            if self._hdi_overlaps_rope(
                *leaf_pairs[parent_id], rope=rope, max_rope_overlap=max_rope_overlap
            ):
                # mark the parent node as a leaf
                self.tree_.children_left[parent_id] = -1
                self.tree_.children_right[parent_id] = -1
                nodes_pruned += 1
        if nodes_pruned > 0:
            self.tree_.max_depth -= 1
            # recursively prune the tree
            return nodes_pruned + self._prune(rope, max_rope_overlap)
        return 0
    

    def _calculate_rope(self, y):
        """
        Calculate the region of practical equivalence (ROPE) based on the beta distribution of the data.
        """
        alpha = self.beta_prior[0] + np.sum(y)
        beta = self.beta_prior[1] + (len(y) - np.sum(y))

        # set the max ROPE overlap to the beta variance
        max_rope_overlap = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # set the ROPE to 10% of the mean of the beta distribution
        beta_mean = alpha / (alpha + beta)
        rope = 0.1 * beta_mean

        return rope, max_rope_overlap

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the model to the data and prune the tree.
        """

        # first fit the model
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        rope, max_rope_overlap = self._calculate_rope(y)
        nr_pruned = self._prune(rope, max_rope_overlap)
        #print(f"Total nodes pruned: {nr_pruned}")


class _TreeUCB_n_TS_single_with_pruning(_TreeUCB_n_TS_single):
    def __init__(
        self, beta_prior=(1, 1), ts=True, alpha=0.8, random_state=None, *args, **kwargs
    ):
        self.beta_prior = beta_prior  ## will be changed later in _OneVsRest
        self.random_state = random_state
        self.conf_coef = alpha
        self.ts = bool(ts)
        self.model = BetaTree(*args, **kwargs)
        self.is_fitted = False
        self.aux_beta = (beta_prior[0], beta_prior[1])  ## changed later


class PartitionedTSWithPruning(PartitionedTS):
    def __init__(
        self,
        nchoices,
        beta_prior=((1, 1), 1),
        smoothing=None,
        noise_to_smooth=True,
        assume_unique_reward=False,
        random_state=None,
        njobs=-1,
        *args,
        **kwargs,
    ):
        #print("Initialized PartitionedTSWithPruning with new logic.")

        if beta_prior is None:
            raise ValueError("Must pass a valid 'beta_prior'.")
        beta_prior = _check_beta_prior(beta_prior, nchoices)

        ## prior and random_state will be changed later inside '_OneVsRest'
        base = _TreeUCB_n_TS_single_with_pruning(
            (1, 1), ts=True, random_state=None, *args, **kwargs
        )
        self._add_common_params(
            base,
            beta_prior,
            smoothing,
            noise_to_smooth,
            njobs,
            nchoices,
            False,
            None,
            False,
            assume_unique_reward,
            random_state,
        )

def get_imbalanced_multiclass_datasets():
    """
    Get datasets with at least 3 classes that are imbalanced
    """
    for dataset in openml.datasets.list_datasets(status="active").values():
        if "NumberOfClasses" not in dataset or dataset["NumberOfClasses"] < 3:
            continue
        if dataset["MinorityClassSize"] / dataset["MajorityClassSize"] < 0.8:
            yield dataset["did"]
    
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
