from __future__ import annotations


def linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    if duration <= 0:
        return end
    t = min(max(step / duration, 0.0), 1.0)
    return start + (end - start) * t
