import numpy as np

from projects.amasa_clean.amasa.safety import SafetyGuard, GuardConfig


def test_guard_outputs_and_lambda_update():
    guard = SafetyGuard(GuardConfig(state_dim=23, action_dim=7, shield_enabled=False, risk_enabled=True))
    state = np.zeros(23, dtype=np.float32)
    action = np.ones(7, dtype=np.float32) * 0.5

    safe_action, info = guard.process_action(state, action)
    assert safe_action.shape == action.shape
    assert "risk_score" in info

    before = guard.lambda_value
    guard.update_cost(1.0)
    after = guard.lambda_value
    assert after >= before


def test_guard_risk_learning_step():
    guard = SafetyGuard(GuardConfig(state_dim=23, action_dim=7, shield_enabled=False, risk_enabled=True))
    state = np.random.randn(23).astype(np.float32)
    action = np.random.randn(7).astype(np.float32)
    loss = guard.observe_for_risk(state, action, cost=1.0)
    assert loss is not None
