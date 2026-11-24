#!/usr/bin/env python3
"""Build the student patch zip: dist/assignment4-patch.zip.

Stages a minimal payload (stripped train_ffnn.py, corrected task
prompts, blanked answers.py, current grader.pyc) together with
apply_patch.py into a zip students can download and run.

Usage:
    uv run python instructor_tools/build_patch.py
"""

import py_compile
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
STAGING = DIST / "assignment4-patch"

# Reuse solution-stripping helpers from the main packaging script.
sys.path.insert(0, str(ROOT))
from package import (  # noqa: E402
    _blank_task_j_answers,
    strip_solutions,
)


def _build_payload() -> None:
    """Assemble payload/ under STAGING."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    payload = STAGING / "payload"
    payload.mkdir()

    # src/training/train_ffnn.py — strip ### BEGIN/END SOLUTION blocks.
    train_src = ROOT / "src" / "training" / "train_ffnn.py"
    train_dst = payload / "src" / "training" / "train_ffnn.py"
    train_dst.parent.mkdir(parents=True)
    train_dst.write_text(strip_solutions(train_src.read_text()))

    # task_prompts — all *.md files, verbatim.
    prompts_dst = payload / "task_prompts"
    prompts_dst.mkdir()
    for f in sorted((ROOT / "task_prompts").glob("*.md")):
        shutil.copy2(f, prompts_dst / f.name)

    # answers.py — blank task-j scalars defensively.
    answers_text = (ROOT / "answers.py").read_text()
    (payload / "answers.py").write_text(_blank_task_j_answers(answers_text))

    # grader/grader.pyc — current correct grader.
    grader_dst = payload / "grader"
    grader_dst.mkdir()
    shutil.copy2(
        ROOT / "grader" / "grader.pyc",
        grader_dst / "grader.pyc",
    )


def _compile_apply_script() -> None:
    """Compile apply_patch.py → apply_patch.pyc in the staging root."""
    py_compile.compile(
        str(ROOT / "instructor_tools" / "apply_patch.py"),
        cfile=str(STAGING / "apply_patch.pyc"),
        optimize=0,
        doraise=True,
    )


def _copy_readme() -> None:
    """Copy README.md to the staging root."""
    shutil.copy2(
        ROOT / "instructor_tools" / "README.md",
        STAGING / "README.md",
    )


def _build_zip() -> Path:
    """Compress STAGING → dist/assignment4-patch.zip."""
    zip_path = DIST / "assignment4-patch.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(STAGING.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(STAGING))
    return zip_path


def main() -> None:
    _build_payload()
    _compile_apply_script()
    _copy_readme()
    zip_path = _build_zip()
    print(f"Staging dir  → {STAGING}")
    print(f"Zip archive  → {zip_path}")


if __name__ == "__main__":
    main()
