from grad_rl.algorithms.value import ALGORITHMS as VALUE_ALGOS
from grad_rl.algorithms.policy import ALGORITHMS as POLICY_ALGOS
from grad_rl.algorithms.actor_critic import ALGORITHMS as AC_ALGOS
from grad_rl.algorithms.model_based import ALGORITHMS as MB_ALGOS
from grad_rl.algorithms.marl import ALGORITHMS as MARL_ALGOS

CHAIN_REGISTRY = {
    "value": VALUE_ALGOS,
    "policy": POLICY_ALGOS,
    "actor_critic": AC_ALGOS,
    "model_based": MB_ALGOS,
    "marl": MARL_ALGOS,
}
