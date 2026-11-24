#!/usr/bin/env python3
"""Apply the A4 patch to an existing student installation.

Run from your A4 assignment directory:

    python apply_patch.py

or pass the path explicitly:

    python apply_patch.py --dir /path/to/A4-neural-networks-in-numpy
"""

import argparse
import shutil
import sys
from pathlib import Path


_PAYLOAD = Path(__file__).resolve().parent / "payload"

# String that appears only inside _run_p() — the last observe-only
# runner — and survives as a literal in the compiled .pyc.
_SENTINEL = b"p_observations"


def _validate_root(root: Path) -> None:
    """Exit with a clear message if root is not an A4 install."""
    missing = [
        name
        for name in (
            "grader/grader.pyc",
            "answers.py",
            "task_prompts",
        )
        if not (root / name).exists()
    ]
    if missing:
        sys.exit(
            f"Error: {root} does not look like an A4 assignment "
            "directory.\n"
            f"  Missing: {', '.join(missing)}\n"
            "  cd into your A4 directory or use --dir PATH."
        )


def _replace(src: Path, dst: Path, label: str) -> None:
    """Overwrite dst with src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  replaced  {label}")


def _skip(label: str, reason: str) -> None:
    print(f"  skipped   {label}  ({reason})")


def _has_observe_runners(grader_pyc: Path) -> bool:
    """Return True if grader.pyc has the task-p observe-only runner."""
    return _SENTINEL in grader_pyc.read_bytes()


def apply_patch(root: Path) -> None:
    """Patch the A4 install at root."""
    _validate_root(root)

    # Always: replace the training loop skeleton.
    _replace(
        _PAYLOAD / "src" / "training" / "train_ffnn.py",
        root / "src" / "training" / "train_ffnn.py",
        "src/training/train_ffnn.py",
    )

    # Always: replace all task prompts.
    prompts_dst = root / "task_prompts"
    for src_prompt in sorted((_PAYLOAD / "task_prompts").glob("*.md")):
        _replace(
            src_prompt,
            prompts_dst / src_prompt.name,
            f"task_prompts/{src_prompt.name}",
        )

    # Detect old grader BEFORE overwriting it (the check reads its bytes).
    grader_pyc = root / "grader" / "grader.pyc"
    grader_was_current = _has_observe_runners(grader_pyc)

    # Always: replace the grader.
    _replace(
        _PAYLOAD / "grader" / "grader.pyc",
        grader_pyc,
        "grader/grader.pyc",
    )

    # Conditional: reset answers.py only when the old grader was detected.
    if grader_was_current:
        _skip(
            "answers.py",
            "grader was current — your answers are preserved",
        )
    else:
        print("  [old grader detected — resetting answers.py]")
        _replace(
            _PAYLOAD / "answers.py",
            root / "answers.py",
            "answers.py",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the A4 patch to an existing student install."
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=".",
        help=(
            "path to your A4 assignment directory "
            "(default: current working directory)"
        ),
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    print(f"Patching {root} ...")
    apply_patch(root)
    print("Done.")


if __name__ == "__main__":
    main()
