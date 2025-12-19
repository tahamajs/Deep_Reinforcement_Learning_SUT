"""Small, import-safe smoke runner that executes a single forward pass using the `debug` config."""
from pathlib import Path
from src.config import Config
from src.model import ActorCritic
from src.data import RandomMDPDataset
from src.utils import set_seed, get_device


def run_once(config_path: Path | str = "configs/debug.yaml") -> None:
    cfg = Config.load_yaml(Path(config_path))
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    net = ActorCritic(obs_dim=4, action_dim=2, hidden_sizes=cfg.hidden_sizes)
    ds = RandomMDPDataset(num_transitions=32, obs_dim=4, action_dim=2)

    batch = ds[0]["obs"].unsqueeze(0)
    action, logp = net.act(batch)
    v = net.get_value(batch)
    print("action", action, "logp", logp, "value", v)


if __name__ == "__main__":
    run_once()
