import pytest
pytest.importorskip("numpy")

from src.data import discounts


def test_discounts_basic():
    rewards = [1.0, 1.0, 1.0]
    # gamma=1 should give cumulative sums
    r1 = discounts(rewards, 1.0)
    assert r1 == [3.0, 2.0, 1.0]

    r05 = discounts(rewards, 0.5)
    assert pytest.approx(r05[0], rel=1e-6) == 1.0 + 0.5 + 0.25


def test_discounts_invalid_gamma():
    with pytest.raises(ValueError):
        discounts([1.0], -0.1)
    with pytest.raises(ValueError):
        discounts([1.0], 1.1)
