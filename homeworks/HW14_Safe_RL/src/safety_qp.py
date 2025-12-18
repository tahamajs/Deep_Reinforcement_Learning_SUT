"""QP-based Safety Layer (action projection) using cvxpy.

This module provides a small helper to project an unsafe continuous action
onto a linear constraint set defined as G a <= h (where h may depend on state).

Uses OSQP/ECOS via CVXPY. Intended for educational experiments; in production
use a vetted solver and numeric checks.
"""
from typing import Optional

import numpy as np

try:
    import cvxpy as cp
except Exception as e:
    cp = None


def qp_project_action(action: np.ndarray, G: np.ndarray, h: np.ndarray, solver: Optional[str] = None) -> np.ndarray:
    """Project `action` onto the feasible set {a | G a <= h} by solving a QP:

    minimize 0.5 * ||a - action||^2
    s.t. G a <= h

    Args:
        action: (d,) raw action vector
        G: (m, d) inequality matrix
        h: (m,) right-hand side vector
        solver: optional cvxpy solver name (e.g., 'OSQP', 'ECOS')

    Returns:
        projected action as numpy array (d,)

    Raises:
        RuntimeError if cvxpy is not installed or solver fails.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for qp_project_action. Install with `pip install cvxpy`.")

    a = cp.Variable(shape=action.shape)
    objective = cp.Minimize(0.5 * cp.sum_squares(a - action))
    constraints = []
    if G is not None and h is not None:
        constraints = [G @ a <= h]

    prob = cp.Problem(objective, constraints)
    try:
        if solver is None:
            # Prefer OSQP if available
            solver_to_try = None
            if 'OSQP' in cp.installed_solvers():
                solver_to_try = cp.OSQP
            elif 'ECOS' in cp.installed_solvers():
                solver_to_try = cp.ECOS
            else:
                solver_to_try = cp.SCS
            prob.solve(solver=solver_to_try, warm_start=True)
        else:
            prob.solve(solver=solver)
    except Exception as exc:
        # Try default solve without specifying solver
        prob.solve()

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"QP solver failed, status={prob.status}")

    sol = np.array(a.value).reshape(action.shape)
    return sol



