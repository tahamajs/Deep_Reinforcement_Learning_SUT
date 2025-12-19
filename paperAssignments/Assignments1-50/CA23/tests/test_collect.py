import pytest
pytest.importorskip("torch")
import torch

from src.data import collect_episodes, Transition


class DummySpace:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class DummyEnv:
    def __init__(self):
        self.observation_space = DummySpace(shape=(4,))
        self.action_space = DummySpace(n=2)
        self._step = 0

    def reset(self):
        # return an observation; gym may return array-like
        return [0.0, 0.0, 0.0, 0.0]

    def step(self, action):
        self._step += 1
        obs = [0.0, 0.0, 0.0, float(self._step)]
        reward = 1.0
        done = self._step >= 2
        info = {}
        return obs, reward, done, info

    def close(self):
        pass


def test_collect_episodes_monkeypatch(monkeypatch):
    def make(env_name):
        return DummyEnv()

    monkeypatch.setattr("gym.make", make)

    class DummyPolicy:
        def get_action(self, x):
            # return an action (tensor or int) and a log prob
            return torch.tensor(0), torch.tensor(0.0)

    episodes = collect_episodes("AnyEnv", DummyPolicy(), num_episodes=2, max_steps=10)
    assert len(episodes) == 2
    assert all(isinstance(ep, list) for ep in episodes)
    assert all(isinstance(t, Transition) for ep in episodes for t in ep)


def test_train_run_smoke(monkeypatch, tmp_path):
    # Smoke test: import scripts.train and run a very small config using DummyEnv
    pytest.importorskip("torch")
    from importlib import import_module
    module = import_module("scripts.train")

    # define DummyEnv locally
    class DummyEnvLocal(DummyEnv):
        pass

    # monkeypatch gym.make to return dummy env
    monkeypatch.setattr("gym.make", lambda name: DummyEnvLocal())

    # ensure the script writes to a temp dir by patching ensure_dir in module
    monkeypatch.setattr(module, "ensure_dir", lambda p: tmp_path)

    # prepare a small config
    from src.config import ExperimentConfig
    cfg = ExperimentConfig()
    cfg.env_name = "Dummy"
    cfg.max_episodes = 1
    cfg.hidden_sizes = [8]
    cfg.batch_size = 1
    cfg.entropy_coef = 0.0

    # Run the training loop (should not raise)
    module.run(cfg)

    # Expect that a CSV training log was attempted to be written to tmp_path
    # (module.run writes to ensure_dir(...)/training_log.csv)
    # since we returned tmp_path from ensure_dir, check that a file exists in that dir
    # (it may be named 'training_log.csv')
    found = any(p.name.endswith("training_log.csv") for p in tmp_path.iterdir())
    assert found
