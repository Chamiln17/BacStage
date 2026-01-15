# Filtering Guide (Pipeline-first, Minimal Manual Work)

This guide explains how to filter your dataset down to **Bac (3AS)** videos using the **existing project pipeline**:

- Collect data: `run_pipeline.py collect`
- Engineer baseline features: `run_pipeline.py engineer`
- Filter to Bac videos: (choose **Option A** or **Option B** below)

The goal is to avoid heavy keyword maintenance. Start with **grade markers (cheap + high-signal)**, then optionally add a small amount of subject logic.

---

## What you already have in this repo

### Data inputs/outputs

- **Raw collected data (snapshots)**: `data/raw/videos_metadata.csv`
- **Engineered features**: `data/processed/videos_engineered.csv`

### Filtering scripts (already in project)

- **Apply Bac filter + export datasets**: `scripts/apply_bac_filter.py`
  - Produces:
    - `data/processed/videos_with_bac_filter.csv`
    - `data/processed/videos_bac_3as_only.csv`
    - `data/processed/videos_non_bac.csv`
- **Create validation samples**: `scripts/validate_filter.py`
  - Produces:
    - `data/validation/sample_for_manual_review.csv`
    - plus per-slice samples

### Existing “lightweight” subject + exam signals

Your feature engineering already computes:
- `subject` using a small keyword map in `src/features/engineer.py` (see `_extract_subject`)
- `is_exam_focused` based on quick exam keywords (Bac/exam/revision/etc.)

That means you can filter without maintaining a massive subject dictionary.

---

## Step 0 — Environment

From repo root:

```bash
uv sync --all-extras
```

---

## Step 1 — Collect (raw snapshots)

```bash
uv run python run_pipeline.py collect --channels data/raw/channels.csv
```

Notes:
- This writes/updates `data/raw/videos_metadata.csv`.
- If you want only the latest snapshot per video at collection time, you can use `--mode dedupe` in collection. If you keep snapshots (recommended), you’ll dedupe later for filtering.

---

## Step 2 — Engineer baseline features (optional but recommended)

This step is useful even if you filter later, because it gives you `subject` and `is_exam_focused`.

```bash
uv run python run_pipeline.py engineer --input data/raw/videos_metadata.csv --output data/processed/videos_engineered.csv
```

---

## Step 3 — Choose your filtering approach

### Option A (recommended if you want “one command”): use the existing Bac filter script

This is the most automated option. It reads raw metadata and writes filtered datasets.

```bash
uv run python scripts/apply_bac_filter.py --dedupe
```

Outputs (in `data/processed/`):
- `videos_with_bac_filter.csv` (full table + filter columns)
- `videos_bac_3as_only.csv` (Bac-only subset)
- `videos_non_bac.csv` (rejected subset)

Then generate manual-review samples:

```bash
uv run python scripts/validate_filter.py
```

Review file:
- `data/validation/sample_for_manual_review.csv`

If you feel the keyword dictionary is too heavy, you can still use this option but gradually simplify the dictionaries over time (keep only high-signal grade markers + a small set of subject keywords).

---

### Option B (minimal manual work): grade-markers-first filter (fast baseline)

If you want a filter that requires **almost no keyword maintenance**, start with:

1) **grade markers** (Bac vs non-Bac grades)
2) (optional) the already-engineered `subject` / `is_exam_focused` signals

This approach is intentionally simple and debuggable.

#### B1) Decide what you mean by “Bac 3AS”

Recommended rule set:

- **Hard include** if text contains explicit 3AS/Bac markers.
- **Hard exclude** if text contains explicit lower-grade markers (1AS/2AS/middle school).
- Otherwise **unknown/ambiguous** (you can drop, or keep only if `is_exam_focused==1` and `subject != "General"`).

#### B2) Paste-able code (Notebook or one-off script)

Use this on **deduped latest snapshot per video** (so you filter “videos”, not “snapshots”).

```python
import pandas as pd

df = pd.read_csv("data/raw/videos_metadata.csv")

# 1) keep latest snapshot per video_id
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="ISO8601", errors="coerce")
df = df.sort_values("snapshot_date", ascending=False).drop_duplicates("video_id", keep="first")

text = (
    df["title"].fillna("").astype(str) + " "
    + df["description"].fillna("").astype(str) + " "
    + df.get("tags", "").fillna("").astype(str)
).str.lower()

# 2) minimal grade markers (start small; expand only when you see consistent misses)
bac_markers = [
    "بكالوريا", "باك", "bac", "bacalauréat", "baccalauréat",
    "3as", "3 as", "ثالثة ثانوي", "السنة الثالثة ثانوي",
    "terminale",
]

non_bac_markers = [
    "1as", "1 as", "أولى ثانوي", "سنة اولى ثانوي", "سنة أولى ثانوي",
    "2as", "2 as", "ثانية ثانوي", "سنة ثانية ثانوي", "سنة ثانية ثانوي",
    "متوسط", "bem", "1am", "2am", "3am", "4am",
    "ابتدائي",
]

has_bac = text.str.contains("|".join(map(pd.regex.escape, bac_markers)), regex=True)
has_non_bac = text.str.contains("|".join(map(pd.regex.escape, non_bac_markers)), regex=True)

# 3) classification
df["grade_flag"] = "unknown"
df.loc[has_non_bac, "grade_flag"] = "non_bac"
df.loc[has_bac & ~has_non_bac, "grade_flag"] = "bac"

# 4) strict Bac-only dataset
df_bac_strict = df[df["grade_flag"] == "bac"].copy()
df_bac_strict.to_csv("data/processed/videos_bac_3as_only_strict.csv", index=False)

# 5) optional “soft” expansion using engineered signals
# If you ran feature engineering, you can merge subject/is_exam_focused and accept some unknowns:
try:
    feat = pd.read_csv("data/processed/videos_engineered.csv")[["video_id", "subject", "is_exam_focused"]]
    df2 = df.merge(feat, on="video_id", how="left")
    soft_keep = (df2["grade_flag"] == "bac") | (
        (df2["grade_flag"] == "unknown")
        & (df2["is_exam_focused"] == 1)
        & (df2["subject"].fillna("General") != "General")
    )
    df_bac_soft = df2[soft_keep].copy()
    df_bac_soft.to_csv("data/processed/videos_bac_3as_only_soft.csv", index=False)
except FileNotFoundError:
    pass
```

Why this works:
- Grade markers are the strongest signal and require the least maintenance.
- `subject` and `is_exam_focused` are already produced by your existing pipeline and are “cheap” features.
- You only add new markers when you repeatedly see missed cases during manual review.

---

## Step 4 — Validate (mandatory)

Even for the minimal filter, validate on a sample:

```bash
uv run python scripts/validate_filter.py
```

Open:
- `data/validation/sample_for_manual_review.csv`

Recommended review workflow:
- Mark `manual_is_bac` (TRUE/FALSE)
- Mark `manual_subject_correct` (TRUE/FALSE)
- Add `notes` for failure patterns

Then adjust **only what’s proven necessary** (usually a few new grade markers or a few exclusions).

---

## Common pitfalls (based on your current data)

- **Snapshots vs videos**: `data/raw/videos_metadata.csv` contains multiple rows per `video_id` across `snapshot_date`. Filter on **latest snapshot** (or define a consistent baseline snapshot) unless you explicitly want snapshot-level filtering.
- **Arabic text variance**: titles may include variations like `سنة اولى` vs `سنة أولى`. Start with the most common spellings; add variants only when you observe misses.
- **“No subject detected”**: your strict keyword approach can reject lots of videos if subject terms are missing; using `subject` from `VideoFeatureEngineer` can reduce this without adding large dictionaries.

---

## Recommended default (low manual work, good coverage)

If you want a practical default today:

1) Run the pipeline:
```bash
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv
```

2) Run filtering on latest per-video snapshot:
```bash
uv run python scripts/apply_bac_filter.py --dedupe
```

3) Validate:
```bash
uv run python scripts/validate_filter.py
```

4) Only then decide whether to:
- simplify the keyword dictionaries (keep grade markers + a small subject set), or
- switch to Option B and rely mostly on grade markers + engineered signals.

