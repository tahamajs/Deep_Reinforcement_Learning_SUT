from .a2c import train_a2c
from .sac import train_sac

ALGORITHMS = {
    "a2c": train_a2c,
    "sac": train_sac,
}
