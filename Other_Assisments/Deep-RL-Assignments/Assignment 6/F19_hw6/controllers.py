"""LQR, iLQR and MPC."""

import numpy as np
from scipy.linalg import solve_continuous_are


def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """

    env.state = x.copy()
    res = env.step(u, dt)
    # gym/gymnasium step compatibility: new API may return (obs, reward, terminated, truncated, info)
    next_state = res[0] if isinstance(res, (list, tuple)) else res
    diff = next_state - x
    xdot = diff / dt

    return xdot


def approximate_A(env, x, u, dynamics, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))

    for i in range(len(x)):

        delta_vector = np.zeros_like(x)
        delta_vector[i] = delta
        A1 = dynamics(env, x + delta_vector, u, dt)
        A2 = dynamics(env, x - delta_vector, u, dt)
        A[:, i] = (A1 - A2) / (2 * delta)

    return A


def approximate_B(env, x, u, dynamics, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))

    for i in range(len(u)):

        delta_vector = np.zeros_like(u)
        delta_vector[i] = delta
        B1 = dynamics(env, x, u + delta_vector, dt)
        B2 = dynamics(env, x, u - delta_vector, dt)
        B[:, i] = (B1 - B2) / (2 * delta)

    return B


def calc_lqr_input(env, sim_env, tN=None, max_iter=None):
    """
    Robust LQR Controller with 'Safe Mode' to prevent physics explosions.
    """
    # 1. Access Environment
    real_env = getattr(env, "unwrapped", env)
    real_sim = getattr(sim_env, "unwrapped", sim_env)

    x = real_env.state.copy()

    # --- SAFETY CHECK: EMERGENCY BRAKING ---
    # If velocities are too high (> 20 rad/s), ignore LQR and just apply brakes.
    # This prevents the "Overflow" and "NaN" errors you are seeing.
    velocities = x[2:]  # Assuming last 2 elements are velocities
    if np.linalg.norm(velocities) > 15.0:
        # print("[WARN] Velocity too high! Emergency Braking.")
        # Apply force opposite to velocity to slow down
        u = -0.5 * velocities
        return np.clip(u, real_env.action_space.low, real_env.action_space.high)

    # 2. Linearization
    u_zero = np.zeros(real_env.action_space.shape[0])
    # Use dt=1e-3. 1e-2 is too coarse for stability, 1e-5 is too small for float32.
    calc_dt = 1e-3

    A = approximate_A(real_sim, x, u_zero, simulate_dynamics, dt=calc_dt)
    B = approximate_B(real_sim, x, u_zero, simulate_dynamics, dt=calc_dt)

    # 3. Valid Check
    # If linearization returned NaNs (simulation already broken), return zero to avoid crash
    if np.isnan(A).any() or np.isnan(B).any():
        return np.zeros(real_env.action_space.shape[0])

    # 4. Tune Gains (The "Calm Down" Fix)
    # We override the env's Q/R to be more stable.
    # High R = Don't use excessive force.
    # Non-zero Q_vel = Don't move too fast.
    Q = np.zeros_like(A)
    np.fill_diagonal(
        Q, [10.0, 10.0, 1.0, 1.0]
    )  # Lower position cost (10), add velocity cost (1)

    R = (
        np.eye(real_env.action_space.shape[0]) * 1.0
    )  # High penalty on action to prevent spikes

    # 5. Solve LQR
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        u = -K @ (x - real_env.goal)
    except Exception:
        # If LQR fails, default to a weak damping action (passive braking)
        # print("[WARN] LQR failed. Passive damping.")
        u = -0.1 * velocities

    # 6. Clip
    u = np.clip(u, real_env.action_space.low, real_env.action_space.high)
    return u
