import pytest

from pitchvision.splits import HoldoutError, assert_not_holdout, development_matches, is_holdout


def test_hash_split_is_stable_and_near_twenty_percent():
    ids = [f"match-{i}" for i in range(5000)]
    flags = [is_holdout(m) for m in ids]
    assert flags == [is_holdout(m) for m in ids]
    assert 0.18 < sum(flags) / len(flags) < 0.22


def test_holdout_is_locked():
    ids = [f"match-{i}" for i in range(200)]
    locked = next(m for m in ids if is_holdout(m))
    with pytest.raises(HoldoutError):
        assert_not_holdout(locked)
    assert_not_holdout(locked, allow_holdout=True)
    assert_not_holdout(development_matches(ids))


def test_int_and_str_ids_agree():
    assert is_holdout(3812) == is_holdout("3812")
