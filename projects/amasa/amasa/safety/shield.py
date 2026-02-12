"""Decision-tree safety shield with explanations.

Three CART classifiers:
1) RiskClassifier: labels state as safe/critical/unsafe.
2) ActionGate: binary allow/deny for proposed action bucketized to {-1,0,1}^7.
3) ReasonTree: predicts textual reason tag for denial.

Trees are trained on logged rollouts with privileged cost labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

RISK_LABELS = {0: "safe", 1: "critical", 2: "unsafe"}
REASONS = {0: "ok", 1: "force>5N", 2: "off-corridor", 3: "terminal"}


class SafetyShield:
    def __init__(self):
        self.risk_tree = DecisionTreeClassifier(max_depth=4)
        self.action_tree = DecisionTreeClassifier(max_depth=4)
        self.reason_tree = DecisionTreeClassifier(max_depth=4)
        self.trained = False

    def fit(self, states, actions, costs, terminals):
        # risk labels derived from costs and terminals
        risk_labels = np.where(costs > 0.5, 2, 0)
        risk_labels = np.where(terminals > 0.5, 1, risk_labels)
        self.risk_tree.fit(states, risk_labels)

        # bucketize actions
        bucket = np.sign(actions)
        allow = (costs < 0.5).astype(int)
        self.action_tree.fit(np.hstack([states, bucket]), allow)

        reason_labels = np.where(costs > 0.5, 1, 0)
        reason_labels = np.where(np.linalg.norm(states[:, :2], axis=1) > 0.03, 2, reason_labels)
        reason_labels = np.where(terminals > 0.5, 3, reason_labels)
        self.reason_tree.fit(np.hstack([states, bucket]), reason_labels)
        self.trained = True

    def filter(self, state, action):
        """Return possibly modified action and explanation list."""
        if not self.trained:
            return action, ["shield not trained"]
        bucket = np.sign(action)[None, :]
        allow = self.action_tree.predict(np.hstack([state[None, :], bucket]))[0]
        risk = RISK_LABELS[self.risk_tree.predict(state[None, :])[0]]
        reason = REASONS[self.reason_tree.predict(np.hstack([state[None, :], bucket]))[0]]
        if allow:
            return action, ["risk:" + risk, "reason:" + reason]
        # clamp action toward zero when denied
        safe_action = 0.2 * np.sign(action)
        return safe_action, ["blocked", "risk:" + risk, "reason:" + reason]

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
