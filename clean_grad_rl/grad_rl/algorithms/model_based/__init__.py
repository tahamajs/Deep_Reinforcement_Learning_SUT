from .dyna_q import train_dyna_q
from .mbpo_lite import train_mbpo_lite

ALGORITHMS = {
    "dyna_q": train_dyna_q,
    "mbpo_lite": train_mbpo_lite,
}
