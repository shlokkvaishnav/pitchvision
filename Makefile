.PHONY: install test lint format check fix-pth weights reproduce
install:
	python3 -m venv .venv && .venv/bin/pip install -q -c requirements.lock -e ".[dev,ml,data,export]" --extra-index-url https://download.pytorch.org/whl/cpu && $(MAKE) fix-pth
test:
	.venv/bin/pytest -q
	node --test tests/js/*.test.mjs
lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/mypy src
format:
	.venv/bin/ruff format src tests
# Do this before each push. A failed CI run sends an email.
check: lint test
fix-pth:
	@# macOS can hide .pth files. Then Python does not read them, and the editable install stops.
	@chflags nohidden .venv/lib/*/site-packages/*.pth 2>/dev/null || true

PY = PYTHONPATH=src .venv/bin/python

# The trained weights of spec 05 (GitHub release, pinned SHA-256).
weights:
	$(PY) scripts/fetch_weights.py

# Rebuild every evaluation result and the demo from public data and the released weights (CPU, no training),
# then compare them with the committed files. Training: scripts/train_more.py on a GPU (see kaggle/).
reproduce: weights
	$(PY) scripts/baseline_audit.py --out results
	$(PY) scripts/rescore_05_spline.py
	$(PY) scripts/holdout.py --dry-run
	$(PY) scripts/holdout.py
	$(PY) scripts/intervals.py
	$(PY) scripts/seeds_summary.py
	$(PY) scripts/export_demo.py --weights data/models/05-more-data.pt --tag spec05 --std-scale 1.1747
	$(PY) scripts/compare_results.py --tol 1e-6
