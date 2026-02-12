"""Metrics records and serialization helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import os
from typing import Iterable, Dict, Any


@dataclass
class StepRecord:
    step: int
    reward: float
    cost: float
    lambda_value: float
    risk_score: float
    shield_blocked: int
    force: float
    corridor_violation: int


@dataclass
class EpisodeRecord:
    episode: int
    total_reward: float
    avg_cost: float
    success: int
    length: int
    scenario: str
    algo: str
    seed: int


def save_records_jsonl(path: str, records: Iterable[StepRecord | EpisodeRecord]):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")


def save_summary_csv(path: str, rows: Iterable[Dict[str, Any]]):
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
