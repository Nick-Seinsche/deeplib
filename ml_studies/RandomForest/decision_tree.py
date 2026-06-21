"""
Scratch implementation of a Decision Tree from Memory. Performance isn't optimal here.

Performance analysis:
=====================

Fit()
-----
n rows of data
d features

C(do_best_split) = d * n * C(information_gain)
C(information_gain) = O(num_labels)

"""

from __future__ import annotations
from abc import ABC, abstractmethod

import math
import numpy as np
import pandas as pd

class Model(ABC):
    @abstractmethod
    def fit(self, X ,y) -> None:
        pass

    def predict(self, X) -> list | type(np.array):
        pass


class Node:
    def __init__(self, index_set: list, parent_node = None, child_nodes = None):
        self.index_set = index_set
        self.predicted_class = None
        self.parent_node = parent_node
        self.child_nodes = child_nodes
        self._split_feature = None
        self._split_threshold = None

    @staticmethod
    def _H(pi: list) -> float:
        return - sum(x * math.log(x, 2) for x in pi if x > 0)

    def information_gain(self, X: pd.DataFrame, y: pd.Series, feature: str, threshold: int):
        node_items = X.loc[self.index_set, :]
        node_labels = y[self.index_set]
        classes = node_labels.unique()
        n = len(node_labels)

        def _proportions(labels):
            return [(labels == c).sum() / len(labels) for c in classes] if len(labels) else [0]

        pi_total = _proportions(node_labels)

        mask_left = node_items[feature] <= threshold

        left_items = node_items.loc[mask_left, :]
        left_labels = node_labels[mask_left]
        right_items = node_items.loc[~mask_left, :]
        right_labels = node_labels[~mask_left]

        n_left = left_labels.shape[0]
        n_right = right_labels.shape[0]

        pi_left = _proportions(left_labels)
        pi_right = _proportions(right_labels)

        return self._H(pi_total) - (n_left / n) * self._H(pi_left) - (n_right / n) * self._H(pi_right)

    def do_best_split(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> tuple[Node, Node] | None:
        features = X.columns
        X_node = X.loc[self.index_set, :]

        if X_node.shape[0] < kwargs["min_sample_split"]:
            return None

        if y.loc[self.index_set].unique().shape[0] == 1:
            return None

        possible_splits = []
        for feat in features:
            possible_feat_splits = X.loc[self.index_set, feat].unique()
            ig_feat_split = [(split, self.information_gain(X.loc[self.index_set, : ], y[self.index_set], feat, split)) for split in possible_feat_splits]
            best_feat_split = max(ig_feat_split, key=lambda x: x[1])
            possible_splits.append((feat, best_feat_split))
        best_split = max(possible_splits, key=lambda x:x[1][1])
        best_feat = best_split[0]
        best_threshold = best_split[1][0]

        self._split_feature = best_feat
        self._split_threshold = best_threshold

        X_left_index = X_node.loc[X_node[best_feat] <= best_threshold].index
        X_right_index = X_node.loc[X_node[best_feat] > best_threshold].index

        return Node(X_left_index, self), Node(X_right_index, self)


class DecisionTreeClassifier(Model):
    def __init__(self, min_sample_split: int = 2):
        self.min_sample_split = min_sample_split

    def get_depth(self):
        def get_depth_node(node):
            if node.child_nodes is None:
                return 1
            else:
                left, right = node.child_nodes
                return 1 + max(get_depth_node(left), get_depth_node(right))

        return get_depth_node(self.root)

    def fit(self, X, y):
        root = Node(list(X.index), parent_node=None)
        leafs = [root]


        while True:
            results = []
            has_split = False
            for leaf in leafs:
                result = leaf.do_best_split(X, y, min_sample_split=self.min_sample_split)
                if result:
                    has_split = True
                results.append((result, leaf))

            if not has_split:
                break

            for result in results:
                leaf = result[1]

                if result[0] is None:
                    continue

                node_left, node_right = result[0][0], result[0][1]
                leaf.child_nodes = [node_left, node_right]
                leafs.remove(leaf)
                leafs.append(node_left)
                leafs.append(node_right)

        for leaf in leafs:
            leaf.predicted_class = y[leaf.index_set].mode()[0]

        self.root = root
        self.leafs = leafs

    def predict(self, X):
        return np.array([self._predict_one(self.root, row) for _, row in X.iterrows()])

    def _predict_one(self, node, row):
        if node.child_nodes is None:
            return node.predicted_class
        left, right = node.child_nodes
        if row[node._split_feature] <= node._split_threshold:
            return self._predict_one(left, row)
        else:
            return self._predict_one(right, row)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC
    from sklearn.metrics import accuracy_score

    # --- 1. Tiny, deterministic dataset -------------------------------
    # Small n and few features on purpose: easy to print/inspect the
    # whole tree by hand and confirm the splits make sense, before
    # trusting accuracy numbers on anything bigger.
    X_arr, y_arr = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"feat_{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # --- 2. Fit your implementation -----------------------------------
    clf = DecisionTreeClassifier(min_sample_split=5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Your DecisionTreeClassifier accuracy: {acc:.3f}")
    print(f"My tree depth: {clf.get_depth()}, leaves: {len(clf.leafs)}")


    # --- 3. Sanity check against sklearn's tree ------------------------
    # Not expected to match exactly (different tie-breaking, no max_depth
    # on your side, etc.) but should be in the same ballpark. If sklearn
    # gets ~0.85+ and yours gets ~0.50 (coin flip), the bug is in your
    # logic, not the data.
    ref = SklearnDTC(min_samples_split=5, random_state=42)
    ref.fit(X_train, y_train)
    ref_preds = ref.predict(X_test)
    ref_acc = accuracy_score(y_test, ref_preds)
    print(f"sklearn DecisionTreeClassifier accuracy: {ref_acc:.3f}")
    print(f"sklearn tree depth: {ref.get_depth()}, leaves: {ref.get_n_leaves()}")

    # --- 4. Degenerate / edge cases --------------------------------
    # Cheap smoke tests that catch the classic off-by-one and
    # empty-split bugs without needing the full dataset.

    # 4a. Single feature is perfectly separable -> should hit ~1.0 acc
    X_easy = pd.DataFrame({"x": [0, 1, 2, 3, 10, 11, 12, 13]})
    y_easy = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    clf_easy = DecisionTreeClassifier(min_sample_split=2)
    clf_easy.fit(X_easy, y_easy)
    easy_preds = clf_easy.predict(X_easy)
    print(f"Perfectly separable toy case accuracy: {accuracy_score(y_easy, easy_preds):.3f}")

    # 4b. All labels identical -> root should become a leaf immediately,
    # no split should be attempted (tests the purity-check fix).
    X_pure = pd.DataFrame({"x": [1, 2, 3, 4]})
    y_pure = pd.Series([1, 1, 1, 1])
    clf_pure = DecisionTreeClassifier(min_sample_split=2)
    clf_pure.fit(X_pure, y_pure)
    print(f"Single-class root is leaf: {clf_pure.root.child_nodes is None}")
    print(f"Single-class root predicted_class: {clf_pure.root.predicted_class}")

    # 4c. Fewer rows than min_sample_split -> root should stay a leaf too
    X_tiny = pd.DataFrame({"x": [1, 2]})
    y_tiny = pd.Series([0, 1])
    clf_tiny = DecisionTreeClassifier(min_sample_split=5)
    clf_tiny.fit(X_tiny, y_tiny)
    print(f"Below-min-split root is leaf: {clf_tiny.root.child_nodes is None}")



