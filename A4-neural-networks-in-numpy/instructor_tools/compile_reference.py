"""
Instructor script: compiles src/ into src_reference/ for student distribution.

For each implementation .py file in src/:
  - Compiles to src_reference/pkg/module.cpython-XY.pyc (same package dir,
    not __pycache__) so Python's cache mechanism never overwrites it.
  - Generates a stub src_reference/pkg/module.py that loads from the .pyc
    via SourcelessFileLoader at import time.

Files in KEEP_AS_SOURCE are copied as plain source (no implementation
secrets in those files): __init__.py and the four provided modules
(module.py, mnist.py, training.py).

Python 3.12+ no longer imports source-less bytecode from __pycache__
automatically, so the explicit loader in each stub is necessary.

Run with: make compile-reference
"""

import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DST = ROOT / "src_reference"

KEEP_AS_SOURCE = {
    "__init__.py",
    "module.py",
    "mnist.py",
    "training.py",
}
SENTINEL_MODULE = Path("nn") / "linear.py"
SENTINEL_LINE = b"_c4f7b2a1 = True\n"

_VT = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

# Stub template written to each src_reference/pkg/module.py.
# At import time the stub locates its companion .pyc (same directory, not
# __pycache__), loads it with SourcelessFileLoader, and replaces itself in
# sys.modules so callers see the compiled module directly.
_STUB = (
    "# Reference implementation. Do not edit.\n"
    "from importlib.machinery import SourcelessFileLoader as _L\n"
    "from importlib.util import module_from_spec as _ms,"
    " spec_from_file_location as _sf\n"
    "from pathlib import Path as _P\n"
    "from sys import modules as _m, version_info as _v\n"
    "_f = str(\n"
    "    _P(__file__).parent\n"
    '    / f"{_P(__file__).stem}.cpython-{_v.major}{_v.minor}.pyc"\n'
    ")\n"
    "_s = _sf(__name__, _f, loader=_L(__name__, _f))\n"
    "_x = _ms(_s)\n"
    "_s.loader.exec_module(_x)\n"
    "_m[__name__] = _x\n"
    "del _L, _ms, _sf, _P, _m, _v, _f, _s, _x\n"
)


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir()

    compiled = 0
    copied = 0

    for src_file in sorted(SRC.rglob("*.py")):
        rel = src_file.relative_to(SRC)
        dst_dir = DST / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        if src_file.name in KEEP_AS_SOURCE:
            shutil.copy2(src_file, dst_dir / src_file.name)
            copied += 1
            continue

        pyc_dest = dst_dir / f"{src_file.stem}.{_VT}.pyc"

        if rel == SENTINEL_MODULE:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(SENTINEL_LINE)
                tmp.write(src_file.read_bytes())
            try:
                py_compile.compile(
                    str(tmp_path),
                    cfile=str(pyc_dest),
                    dfile=str(src_file),
                    doraise=True,
                    optimize=0,
                )
            finally:
                tmp_path.unlink()
        else:
            py_compile.compile(
                str(src_file),
                cfile=str(pyc_dest),
                dfile=str(src_file),
                doraise=True,
                optimize=0,
            )

        (dst_dir / src_file.name).write_text(_STUB)
        compiled += 1

    print(f"Compiled {compiled} modules, copied {copied} source files.")
    print(f"Reference written to {DST}")


if __name__ == "__main__":
    main()
