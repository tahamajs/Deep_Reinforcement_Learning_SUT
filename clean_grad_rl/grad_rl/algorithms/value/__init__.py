from .dqn import train_dqn, train_rainbow_lite

ALGORITHMS = {
    "dqn": train_dqn,
    "rainbow_lite": train_rainbow_lite,
}
