#!/usr/bin/env python3
"""Package A4 for student release.

Builds a staging directory ./A4-neural-networks-in-numpy/ (the nested
same-name subfolder) then zips it to ../assignment4.zip.

All destructive transforms (strip solutions, stub tests, blank answers,
rewrite submit.py/Makefile) happen only on staging copies.  The working
instructor tree is never modified.
"""

import ast
import py_compile
import re
import shutil
import zipfile
from pathlib import Path

# Top-level items to copy into the staging dir (explicit allowlist;
# excludes the staging dir itself, solution.py, instructor_tools/, etc.)
_TOP_LEVEL_DIRS = [
    "scripts",
    "src",
    "task_prompts",
    "tests",
]

_TOP_LEVEL_FILES = [
    "answers.py",
    "conftest.py",
    "pyproject.toml",
    "README.md",
    "submit.py",
    "Makefile",
    ".python-version",
    ".gitignore",
]

# Test files that exercise ONLY provided infrastructure — ship in full
# (no stubbing, they have no smoke tests and test the Module base class
# and diagnostic Metric subclasses which students don't implement).
_FULL_TEST_FILES = {
    "test_module_contract.py",
    "test_parameter_metrics.py",
}

# Build artifacts that must never ship: stale bytecode under __pycache__
# would leak reference solutions for to-implement files, and the mypy
# cache is instructor-only noise.
_CACHE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".mypy_cache"
)

# Same, but preserving deliberate top-level .pyc modules (used for the
# src_reference tree, which ships compiled reference modules on purpose).
_CACHE_IGNORE_KEEP_PYC = shutil.ignore_patterns("__pycache__", ".mypy_cache")

# Instructor-only Makefile targets to strip from the shipped copy.
_INSTRUCTOR_TARGETS = {
    "package",
    "compile-grader",
    "compile-reference",
    "release",
}


# ---------------------------------------------------------------------------
# Source transforms
# ---------------------------------------------------------------------------


def strip_solutions(source: str) -> str:
    """Replace ### BEGIN SOLUTION … ### END SOLUTION blocks with a stub."""
    return re.sub(
        r"([ \t]*)### BEGIN SOLUTION.*?### END SOLUTION",
        r"\1# ------ WRITE YOUR CODE HERE ------\n\1pass",
        source,
        flags=re.DOTALL,
    )


def stub_test_functions(source: str) -> str:
    """Replace non-smoke test bodies with ``pass``.

    Functions whose name ends in ``_smoke`` are kept verbatim.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    replacements = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_")
            and not node.name.endswith("_smoke")
        ):
            def_line = lines[node.lineno - 1]
            fn_indent = len(def_line) - len(def_line.lstrip())
            body_indent = " " * (fn_indent + 4)
            replacements.append(
                (node.body[0].lineno, node.end_lineno, body_indent)
            )

    for start, end, body_indent in sorted(
        replacements, key=lambda x: x[0], reverse=True
    ):
        lines[start - 1 : end] = [body_indent + "pass\n"]

    return "".join(lines)


def _strip_makefile_instructor_targets(source: str) -> str:
    """Remove instructor-only Makefile targets and their .PHONY entries.

    Target definition blocks are removed entirely.  Target names are also
    removed from the .PHONY declaration.  Recipe body lines (tab-indented)
    that happen to mention these names are left untouched.
    """
    # Pass 1: remove target definition blocks line-by-line.
    # A block starts at a "target:..." line and ends before the next
    # non-blank, non-tab-indented line.
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    skip_block = False

    for line in lines:
        if skip_block:
            if line.startswith("\t") or line.strip() == "":
                continue  # still inside the block
            skip_block = False  # block ended; process line normally

        # Check if this line opens an instructor-only target block
        is_instr = any(
            re.match(rf"^{re.escape(t)}:", line) for t in _INSTRUCTOR_TARGETS
        )
        if is_instr:
            skip_block = True
            continue

        out.append(line)

    # Pass 2: remove instructor target names from .PHONY lines.
    # We process the .PHONY block (a single declaration spanning backslash-
    # continued lines).  We only modify lines that are part of .PHONY, not
    # recipe bodies.  A .PHONY block consists of the ".PHONY: ..." line
    # followed by whitespace-indented continuation lines that end with "\".
    result_lines: list[str] = []
    in_phony = False

    for line in out:
        if line.startswith(".PHONY:"):
            in_phony = True

        if in_phony:
            new_line = line
            for target in _INSTRUCTOR_TARGETS:
                # Remove "target" as a whole word (surrounded by spaces/tabs,
                # backslash, or line boundaries).
                new_line = re.sub(
                    rf"(?<=[ \t]){re.escape(target)}(?=[ \t\\]|$)",
                    "",
                    new_line,
                )
            # Collapse multiple spaces to one
            new_line = re.sub(r"  +", " ", new_line)
            # If this continuation line is now empty (only whitespace + \),
            # skip it entirely.
            tail = new_line.rstrip()
            if tail.rstrip("\\").strip() == "" and not tail.startswith(
                ".PHONY"
            ):
                # Check if the previous line ended with \ — if so, this
                # would break continuation; just drop this empty line.
                if not line.rstrip().endswith("\\"):
                    in_phony = False
                continue
            result_lines.append(new_line)
            # .PHONY block ends when a line does NOT end with backslash
            if not new_line.rstrip().endswith("\\"):
                in_phony = False
        else:
            result_lines.append(line)

    result = "".join(result_lines)
    # Clean up runs of 3+ blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _blank_task_j_answers(source: str) -> str:
    """Reset the three filled task-j scalars back to None."""
    for var in ("j_best_lr", "j_lr_001_underfits", "j_lr_10_unstable"):
        # Match "var: <type annotation> = <value>" — annotation may contain
        # spaces and pipe chars (e.g. "float | None"), so consume up to " = ".
        source = re.sub(
            rf"^({re.escape(var)}\s*:.*?=\s*).*$",
            r"\1None",
            source,
            flags=re.MULTILINE,
        )
    return source


def _strip_portfolio_sections(source: str) -> str:
    """Remove README blocks between BEGIN/END PORTFOLIO comment markers."""
    result = re.sub(
        r"<!-- BEGIN PORTFOLIO -->.*?<!-- END PORTFOLIO -->\n?",
        "",
        source,
        flags=re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", result).lstrip("\n")


def _rewrite_submit_import(source: str) -> str:
    """Change ``from grader_impl import`` to ``from grader import``."""
    return source.replace("from grader_impl import", "from grader import")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def main() -> None:
    assignment_dir = Path(__file__).resolve().parent
    dir_name = assignment_dir.name  # "A4-neural-networks-in-numpy"

    staging_dir = assignment_dir / dir_name
    zip_path = assignment_dir.parent / "assignment4.zip"

    # ------------------------------------------------------------------ clean
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()

    # ---------------------------------------------------------------- src/ (stripped)
    src_dst = staging_dir / "src"
    shutil.copytree(assignment_dir / "src", src_dst, ignore=_CACHE_IGNORE)
    for py_file in src_dst.rglob("*.py"):
        py_file.write_text(strip_solutions(py_file.read_text()))

    # ---------------------------------------------------------------- src_reference/ (verbatim)
    shutil.copytree(
        assignment_dir / "src_reference",
        staging_dir / "src_reference",
        ignore=_CACHE_IGNORE_KEEP_PYC,
    )

    # ---------------------------------------------------------------- tests/ (smoke skeleton)
    tests_dst = staging_dir / "tests"
    tests_dst.mkdir()
    for src_test in sorted((assignment_dir / "tests").glob("test_*.py")):
        text = src_test.read_text()
        if src_test.name not in _FULL_TEST_FILES:
            text = stub_test_functions(text)
        (tests_dst / src_test.name).write_text(text)

    # ---------------------------------------------------------------- grader/ (pyc only)
    # Always recompile from grader_impl.py so the shipped grader.pyc can
    # never be stale relative to the current grader source.
    grader_dst = staging_dir / "grader"
    grader_dst.mkdir()
    py_compile.compile(
        str(assignment_dir / "grader" / "grader_impl.py"),
        cfile=str(grader_dst / "grader.pyc"),
        optimize=2,
        doraise=True,
    )

    # ---------------------------------------------------------------- task_prompts/ scripts/ (verbatim)
    for dir_name_item in ("task_prompts", "scripts"):
        shutil.copytree(
            assignment_dir / dir_name_item,
            staging_dir / dir_name_item,
            ignore=_CACHE_IGNORE,
        )

    # ---------------------------------------------------------------- top-level files
    for name in _TOP_LEVEL_FILES:
        src = assignment_dir / name
        if not src.exists():
            continue
        text = src.read_text()
        if name == "submit.py":
            text = _rewrite_submit_import(text)
        elif name == "Makefile":
            text = _strip_makefile_instructor_targets(text)
        elif name == "answers.py":
            text = _blank_task_j_answers(text)
        elif name == "README.md":
            text = _strip_portfolio_sections(text)
        (staging_dir / name).write_text(text)

    # ---------------------------------------------------------------- empty data/ results/figures/
    for subpath in ("data", "results/figures"):
        p = staging_dir / subpath
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    # ---------------------------------------------------------------- zip
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging_dir.rglob("*")):
            if file.is_file():
                # Internal path: A4-neural-networks-in-numpy/<rel>
                arcname = assignment_dir.name / file.relative_to(staging_dir)
                zf.write(file, arcname)

    print(f"Staging dir  → {staging_dir}")
    print(f"Zip archive  → {zip_path}")


if __name__ == "__main__":
    main()
