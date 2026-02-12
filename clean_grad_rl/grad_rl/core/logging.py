from __future__ import annotations

import json
import time
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunLogger:
    run_id: str
    chain: str
    algo: str
    env: str
    seed: int
    budget: Dict[str, Any]
    out_dir: Path
    train_curve: List[Dict[str, float]] = field(default_factory=list)
    safety_curve: List[Dict[str, float]] = field(default_factory=list)
    _start: float = field(default_factory=time.time)

    def log_train(self, step_or_ep: int, reward: float, cost: Optional[float] = None):
        point = {"x": float(step_or_ep), "reward": float(reward)}
        self.train_curve.append(point)
        if cost is not None:
            self.safety_curve.append({"x": float(step_or_ep), "cost": float(cost)})

    def finalize(self, eval_stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        elapsed = time.time() - self._start
        fps = 0.0
        if self.train_curve:
            max_x = max(p["x"] for p in self.train_curve)
            fps = max_x / elapsed if elapsed > 0 else 0.0
        payload = {
            "run_id": self.run_id,
            "timestamp": int(time.time()),
            "chain": self.chain,
            "algo": self.algo,
            "env": self.env,
            "seed": self.seed,
            "budget": self.budget,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "train_curve": self.train_curve,
            "eval": eval_stats,
            "timing": {
                "wall_clock_sec": elapsed,
                "fps": fps,
            },
        }
        if self.safety_curve:
            payload["safety"] = {
                "curve": self.safety_curve,
                "mean_cost": eval_stats.get("mean_cost", 0.0),
                "violation_rate": eval_stats.get("violation_rate", 0.0),
            }
        if extra:
            payload.update(extra)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload
