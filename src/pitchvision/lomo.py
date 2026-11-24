"""Leave-one-match-out folds of spec 15 (docs/specs/15-leave-one-match-out.md, section 2)."""

from __future__ import annotations

MATCHES = (
    "idsse-J03WMX", "idsse-J03WN1", "idsse-J03WOH", "idsse-J03WOY", "idsse-J03WPY",
    "idsse-J03WQQ", "idsse-J03WR9", "metrica-1", "metrica-2", "metrica-3",
)  # fmt: skip
# Spec 15 amendment A2: 5 folds; every other match of MATCHES, starting with the first (4 IDSSE, 1 Metrica)
TEST_MATCHES = MATCHES[::2]
NO_TUNE = "idsse-J03WN1"  # 208 windows: too few to tune on
SPEC07_DONORS = {"idsse-J03WN1": "1996435", "idsse-J03WOH": "1996436"}
HOLDOUT = tuple(SPEC07_DONORS)  # the spec 07 matches, used here on the owner's decision (spec 15)
GAP_BUCKETS_S = (0.0, 1.0, 3.0, 6.0, 9.6, float("inf"))
GAP_LABELS = ("<1s", "1-3s", "3-6s", "6-9.6s", ">=9.6s")


def fold(test: str) -> tuple[list[str], str]:
    """(training matches, tuning match) for the fold that tests on `test`."""
    i = MATCHES.index(test)
    order = [MATCHES[(i + j) % len(MATCHES)] for j in range(1, len(MATCHES))]
    tune = next(m for m in order if m != NO_TUNE)
    return [m for m in MATCHES if m not in (test, tune)], tune


def all_donors() -> dict[str, str]:
    """Camera-mask donor for each of the 10 matches."""
    from .dataset import donors

    return {**donors(), **SPEC07_DONORS}
