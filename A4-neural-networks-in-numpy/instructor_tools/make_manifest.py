#!/usr/bin/env python3
"""Generate assignment4.yaml with real expected hashes.

Computes hashes by injecting solution.py as the `answers` module in
memory (no on-disk edits), then running every grader runner against the
full working-tree src/ (solutions active — NOT reference mode).

Usage:
    uv run python instructor_tools/make_manifest.py
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure src/ and root are on sys.path (same setup as submit.py)
for p in (str(ROOT / "src"), str(ROOT), str(ROOT / "grader")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Task → category metadata
# ---------------------------------------------------------------------------

_TASK_CATEGORIES = [
    ("b", "b-linear", "The Linear Layer"),
    ("c", "c-activation", "Activation Functions"),
    ("d", "d-loss", "Loss Functions"),
    ("e", "e-regularization", "Dropout & Regularization"),
    ("f", "f-models", "Sequential & MLP"),
    ("g", "g-optim", "Optimizers: SGD & Adam"),
    ("h", "h-data-handling", "Datasets & DataLoader"),
    ("i", "i-metrics", "Evaluation Metrics"),
    ("j", "j-training", "Training Loop & Learning Rate"),
    ("k", "k-decision-boundaries", "Decision Boundaries"),
    ("l", "l-training-diagnostics", "Training Diagnostics"),
    ("m", "m-regression", "Regression"),
    ("n", "n-input-scaling", "Input Scaling"),
    (
        "o",
        "o-hyperparameter-experiments",
        "Optimizer & Regularization Experiments",
    ),
    ("p", "p-weight-visualization", "Weight Visualization"),
]

# Map runner key → category id
_RUNNER_TO_CAT: dict[str, str] = {
    runner_key: cat_id for runner_key, cat_id, _ in _TASK_CATEGORIES
}


def _title_from_id(test_id: str) -> str:
    """Derive a human-readable title from a test ID."""
    return test_id.replace("_", " ")


def _quote(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'


def main() -> None:
    # ------------------------------------------------------------------
    # Inject solution.py as `answers` before importing grader_impl so
    # grader runners pick up the correct answer-key values.
    # ------------------------------------------------------------------
    solution = importlib.import_module("solution")
    sys.modules["answers"] = solution  # type: ignore[assignment]

    import grader_impl  # noqa: E402 (after sys.path setup above)

    # ------------------------------------------------------------------
    # Run each task individually to learn which test IDs belong to it.
    # ------------------------------------------------------------------
    task_tests: dict[str, dict[str, str]] = {}
    # Full combined submission for reference
    all_tests: dict[str, str] = {}

    for runner_key, cat_id, _ in _TASK_CATEGORIES:
        runner = grader_impl._RUNNERS[runner_key]
        raw = runner()  # {test_id: hash_hex}
        task_tests[runner_key] = raw
        all_tests.update(raw)

    # ------------------------------------------------------------------
    # Build YAML text
    # ------------------------------------------------------------------
    lines: list[str] = [
        "schema_version: 1",
        "id: a4-neural-networks-in-numpy",
        'title: "A4: Neural Networks in NumPy"',
        "description_md: |",
        "  Run `uv run python submit.py`, then upload submission.json.",
        "order: 3",
        "categories:",
    ]

    for _, cat_id, cat_label in _TASK_CATEGORIES:
        lines.append(f"  - {{id: {cat_id}, label: {_quote(cat_label)}}}")

    lines.append("tests:")

    for runner_key, cat_id, _ in _TASK_CATEGORIES:
        for test_id, hash_hex in task_tests[runner_key].items():
            # Mirror run_puzzles: hyphenate ids to match homework.yaml.
            manifest_id = test_id.replace("_", "-")
            title = _title_from_id(test_id)
            expected = f"sha256:{hash_hex}"
            lines += [
                f"  - id: {manifest_id}",
                f"    title: {_quote(title)}",
                f"    category: {cat_id}",
                "    points: 1",
                f"    expected_hash: {_quote(expected)}",
            ]

    yaml_text = "\n".join(lines) + "\n"

    out = ROOT / "assignment4.yaml"
    out.write_text(yaml_text)
    print(f"Written → {out}")

    # Summary
    total = sum(len(v) for v in task_tests.values())
    print("\nTest IDs by category:")
    for runner_key, cat_id, cat_label in _TASK_CATEGORIES:
        n = len(task_tests[runner_key])
        print(f"  {cat_id:<38} {n:3d} tests")
    print(f"\nTotal: {total} tests")

    # Also dump a submission.json for inspection
    submission = {
        "schema_version": 1,
        "homework_id": "a4-neural-networks-in-numpy",
        "tests": {
            tid.replace("_", "-"): f"sha256:{h}"
            for tid, h in all_tests.items()
        },
    }
    sub_path = ROOT / "submission_instructor.json"
    sub_path.write_text(json.dumps(submission, indent=2))
    print(f"\nInstructor submission → {sub_path}")


if __name__ == "__main__":
    main()
