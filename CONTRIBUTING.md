# Contributing

- Install Python 3.11 or 3.12 with `python -m pip install -e '.[dev]'`. Run `python -m pytest tests -q` and `python -m pipeline.build_group_summaries --check`.
- Run notebooks with `python scripts/verify_notebooks.py`; it copies inputs to a temporary workspace and exports separately. Preserve observations and reports; use synthetic regression cases.
- Describe the changed method, command and expected result. Keep units, clocks, [rights](NOTICE.md) and [consent](DATA_ETHICS.md) explicit.

## Hashed environment

The lock targets **CPython 3.11, x86_64 GNU/Linux**. It pins distributions, without promising identical runtime behavior across platforms.

```sh
python -m pip install --require-hashes -r requirements.lock
python -m pip install -e . --no-deps
```

## Notebooks

`MMS_DATA_ROOT` selects data; `MMS_OUTPUT_ROOT` selects a separate scratch directory. Without an output root, helpers do not write. `MMS_SENSOR_TIMEZONE` requires documented acquisition metadata before joining naive clocks to UTC questions.
