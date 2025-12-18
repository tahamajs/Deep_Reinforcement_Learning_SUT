from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TriggerConfig:
    cooldown: int = 10
    trigger_td: float = 0.7
    trigger_unc: float = 0.2
    trigger_ent_low: float = 0.3
    trigger_ent_high: float = 2.0


def should_trigger(
    td_error: float,
    unc: float,
    entropy: float,
    cfg: TriggerConfig,
    last_trigger: Optional[int],
    step: int,
) -> bool:
    """
    Decide whether to trigger SimGolf planner based on metrics.
    - td_error: scalar absolute TD error
    - unc: uncertainty scalar (ensemble var or similar)
    - entropy: policy entropy scalar
    - cfg: TriggerConfig with thresholds and cooldown
    - last_trigger: last step when planner fired (or None)
    - step: current global step
    """
    if last_trigger is not None and (step - last_trigger) < cfg.cooldown:
        return False
    cond_td = float(td_error) > float(cfg.trigger_td)
    cond_unc = float(unc) > float(cfg.trigger_unc)
    cond_ent = (float(entropy) < float(cfg.trigger_ent_low)) or (
        float(entropy) > float(cfg.trigger_ent_high)
    )
    return bool(cond_td or cond_unc or cond_ent)










