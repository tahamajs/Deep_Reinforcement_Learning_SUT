from .ippo import train_ippo
from .qmix_lite import train_qmix_lite

ALGORITHMS = {
    "ippo": train_ippo,
    "qmix_lite": train_qmix_lite,
}
