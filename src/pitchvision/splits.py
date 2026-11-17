"""Match-level splits and the holdout lock (see docs/specs/00-charter.md, section 4)."""

from __future__ import annotations

import hashlib

HOLDOUT_FRACTION = 0.20
_SALT = "pitchvision-spec00"


class HoldoutError(RuntimeError):
    """A development step tried to use a locked holdout match."""


def is_holdout(match_id: str | int) -> bool:
    """Return True if a fixed hash of the match ID puts the match in the holdout."""
    digest = hashlib.sha256(f"{_SALT}:{match_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < HOLDOUT_FRACTION


def assert_not_holdout(match_ids, *, allow_holdout: bool = False) -> None:
    """Raise HoldoutError if a match is in the holdout. Only the final spec sets allow_holdout."""
    items = [match_ids] if isinstance(match_ids, (str, int)) else list(match_ids)
    bad = sorted(str(m) for m in items if is_holdout(m))
    if bad and not allow_holdout:
        raise HoldoutError(f"these matches are in the locked holdout: {bad}")


def development_matches(match_ids) -> list:
    """Return the matches that are not in the holdout, in the same order."""
    return [m for m in match_ids if not is_holdout(m)]
