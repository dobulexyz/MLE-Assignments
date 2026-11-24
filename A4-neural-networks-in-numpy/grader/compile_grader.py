"""Compile grader_impl.py -> grader.pyc for student distribution."""

import py_compile
import sys
from pathlib import Path

_DIR = Path(__file__).parent
src = _DIR / "grader_impl.py"
dst = _DIR / "grader.pyc"

py_compile.compile(str(src), cfile=str(dst), optimize=2, doraise=True)
print(f"Compiled  {src.name} -> {dst.name}")
print(f"Python    {sys.version}")
