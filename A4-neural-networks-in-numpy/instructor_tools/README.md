# Assignment 4 — Patch Instructions

This patch updates your existing A4 installation with corrected task prompts,
an updated training-loop skeleton, and (if your grader is outdated) an updated
grader and a reset `answers.py`.

## What gets updated

| File | Always? |
|------|---------|
| `task_prompts/*.md` | Yes |
| `src/training/train_ffnn.py` | Yes |
| `grader/grader.pyc` | Yes |

Your implementations in `src/` are never touched.

## How to apply

**Step 1 — Unzip the archive** somewhere convenient:

```
unzip assignment4-patch.zip
```

**Step 2 — Run the patch script** from inside your A4 directory
(requires Python 3.14):

```
cd /path/to/A4-neural-networks-in-numpy
uv run python /path/to/assignment4-patch/apply_patch.pyc
```

Alternatively, pass the path explicitly instead of `cd`-ing first:

```
python /path/to/assignment4-patch/apply_patch.pyc --dir /path/to/A4-neural-networks-in-numpy
```
