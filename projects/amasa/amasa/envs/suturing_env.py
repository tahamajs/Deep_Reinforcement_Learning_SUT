import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SuturingEnv(gym.Env):
    """Lightweight surgical suturing task.

    State (float32 vector):
      [q(7), dq(7), needle_xyz(3), tissue_stress, progress, phase_one_hot(4)]
    Action: 7 joint torques in [-1, 1].

    Reward:
      +1000 when all 4 sutures complete (sparse)
      shaped: -dist_to_target - 0.1*force - 0.01*|action|^2 + phase bonus
    Cost:
      1 if force > 5N or needle leaves safe corridor, else 0
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps: int = 500, seed: int | None = None):
        super().__init__()
        self.n_joints = 7
        self.max_steps = max_steps
        self.dt = 0.02
        self.safe_force = 5.0
        self.safe_corridor = 0.03
        self.n_sutures = 4
        self.rng = np.random.default_rng(seed)

        high_obs = np.ones(7 + 7 + 3 + 1 + 1 + 4, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        self.q = np.zeros(self.n_joints, dtype=np.float32)
        self.dq = np.zeros(self.n_joints, dtype=np.float32)
        self.needle = np.zeros(3, dtype=np.float32)
        self.stress = 0.0
        self.step_count = 0
        self.suture_idx = 0
        self.phase = 0
        self.targets = self._sample_targets()

    def _sample_targets(self):
        # Four target entry points along a gentle arc; random offset per episode
        base = np.array([
            [0.02, 0.00, -0.01],
            [0.025, 0.005, -0.01],
            [0.03, -0.005, -0.01],
            [0.035, 0.0, -0.01],
        ], dtype=np.float32)
        jitter = self.rng.normal(scale=0.002, size=base.shape)
        return base + jitter

    def _obs(self):
        phase_one_hot = np.zeros(4, dtype=np.float32)
        phase_one_hot[self.phase] = 1.0
        progress = float(self.suture_idx) / self.n_sutures
        return np.concatenate([
            self.q,
            self.dq,
            self.needle,
            np.array([self.stress], dtype=np.float32),
            np.array([progress], dtype=np.float32),
            phase_one_hot,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q = self.rng.normal(scale=0.05, size=self.n_joints).astype(np.float32)
        self.dq = np.zeros_like(self.q)
        self.needle = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.stress = 0.0
        self.step_count = 0
        self.suture_idx = 0
        self.phase = 0
        self.targets = self._sample_targets()
        return self._obs(), {}

    # ---- dynamics helpers -------------------------------------------------
    def _compute_force(self, action, dist):
        # crude proxy: larger torques and deeper penetration produce force
        base = 6.0 * np.tanh(np.linalg.norm(action))
        depth_penalty = 80.0 * max(0.0, -dist[2])  # pushing into tissue (z<target)
        noise = self.rng.normal(scale=0.1)
        return float(max(0.0, base + depth_penalty + noise))

    def _update_needle(self, action):
        # forward kinematics surrogate: needle moves with first 3 joint velocities
        delta = 0.005 * action[:3] + self.rng.normal(scale=0.0005, size=3)
        self.needle += delta.astype(np.float32)

    def _update_phase(self, dist):
        # phase 0: approach, 1: pierce, 2: pull-through, 3: knot
        if self.phase == 0 and np.linalg.norm(dist) < 0.01:
            self.phase = 1
        elif self.phase == 1 and dist[2] < -0.003:
            self.phase = 2
        elif self.phase == 2 and np.linalg.norm(dist) < 0.007 and self.stress < self.safe_force:
            self.phase = 3
        elif self.phase == 3 and np.linalg.norm(dist) < 0.005:
            # completed this suture
            self.suture_idx += 1
            self.phase = 0
            if self.suture_idx < self.n_sutures:
                # move target slightly deeper for next suture
                self.targets[self.suture_idx] += np.array([0.0, 0.0, -0.001], dtype=np.float32)

    # ---- step -------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # joint dynamics (damped integrator)
        self.dq = 0.92 * self.dq + 0.08 * action + self.rng.normal(scale=0.01, size=self.n_joints)
        self.q = self.q + self.dq * self.dt

        target = self.targets[min(self.suture_idx, self.n_sutures - 1)]
        self._update_needle(action)
        dist = self.needle - target
        self.stress = self._compute_force(action, dist)
        self._update_phase(dist)

        reward = -np.linalg.norm(dist) - 0.1 * self.stress - 0.01 * float(np.square(action).mean())
        reward += 2.0 * (self.phase == 1) + 4.0 * (self.phase == 2) + 6.0 * (self.phase == 3)
        terminated = False
        success = self.suture_idx >= self.n_sutures
        if success:
            reward += 1000.0
            terminated = True
        cost = float(self.stress > self.safe_force or np.linalg.norm(dist[:2]) > self.safe_corridor)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        obs = self._obs()
        info = {
            "cost": cost,
            "force": self.stress,
            "dist": np.linalg.norm(dist),
            "phase": self.phase,
            "suture_idx": self.suture_idx,
            "success": success,
        }
        return obs, reward, terminated, truncated, info

    # ---- vectorized rollout helper ---------------------------------------
    def rollout(self, policy_fn, episodes=1):
        trajectories = []
        for _ in range(episodes):
            obs, _ = self.reset()
            done = False
            ep = []
            while not done:
                action = policy_fn(obs)
                next_obs, reward, terminated, truncated, info = self.step(action)
                ep.append((obs, action, reward, next_obs, float(terminated), float(info.get("cost", 0.0))))
                obs = next_obs
                done = terminated or truncated
            trajectories.append(ep)
        return trajectories


def make_env(**kwargs):
    return SuturingEnv(**kwargs)


if __name__ == "__main__":
    env = SuturingEnv()
    o, _ = env.reset()
    total_r = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        o, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            break
    print("rollout reward", total_r, "cost", info["cost"], "success", info["success"])
