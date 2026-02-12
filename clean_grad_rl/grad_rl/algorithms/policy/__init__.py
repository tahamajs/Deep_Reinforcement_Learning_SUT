from .reinforce import train_reinforce
from .ppo import train_ppo
from .trpo_lite import train_trpo_lite
from .cpo_lite import train_cpo_lite

ALGORITHMS = {
    "reinforce": train_reinforce,
    "ppo": train_ppo,
    "trpo_lite": train_trpo_lite,
    "cpo_lite": train_cpo_lite,
}
