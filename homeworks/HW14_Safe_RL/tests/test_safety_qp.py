import numpy as np
from homeworks.HW14_Safe_RL.src.safety_qp import qp_project_action


def test_qp_project_simple_box():
    # 2D action space with box constraints: -0.5 <= a_i <= 0.5
    G = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h = np.array([0.5, 0.5, 0.5, 0.5])

    action = np.array([1.0, -1.0])
    proj = qp_project_action(action, G, h)

    # Expected projection is clipped to box [-0.5, 0.5]
    expected = np.array([0.5, -0.5])
    assert np.allclose(proj, expected, atol=1e-3)






