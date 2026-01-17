# Targets & Filtering Guide (YouTube Educational Advisor)

This document summarizes the agreed approach for:

- Defining **target variables** for engagement using **snapshot (panel) data**
- Preventing bias/leakage with **irregular re-scrape intervals**
- Defining **Bac (3AS) relevance** from metadata (title/description/tags)
- Filtering the **raw video table** into a modeling-ready dataset

---

## Why keep snapshots?

You should keep multiple rows per video with different `snapshot_date` because it enables:

- **True velocity / growth**: you can compute changes over time (e.g., views/day), not just total counts.
- **Leakage-safe labeling**: you can build targets from *future deltas* (e.g., next 7 days) rather than using totals that encode video age.
- **Auditability**: snapshots preserve historical states (you can re-run feature engineering and reproduce labels).

**Important**: irregular snapshot spacing is fine as long as you always use actual time gaps (days) when computing rates.

---

## Target variables: what “engagement” means with current data

Today you have:

- **Exposure**: `view_count`
- **Active interactions**: `like_count`, `comment_count`
- **Video metadata**: title, description, tags, publish time, duration, etc.

You do *not yet* have watch time/retention (often the best proxy for learning impact). Therefore, engagement must be defined using **growth + interaction intensity**, not retention.

### Recommended: two-target mindset (best for an “advisor”)

Instead of forcing one number to mean everything, model two signals:

1) **Growth (reach / attention)**
- Captures whether the video is gaining views.

2) **Depth (impact proxy)**
- Captures whether viewers are interacting (especially comments).

You can combine them later for ranking, but separating them makes the system more interpretable:

- “This change helps growth.”
- “This change helps discussion/engagement depth.”

---

## Leakage-safe labeling with snapshots

### Define a baseline time and horizon

Pick a baseline snapshot time $t$ and a horizon $H$ (days). Then compute **future deltas** from snapshots:

- $\Delta V_H = V(t+H) - V(t)$
- $\Delta L_H = L(t+H) - L(t)$
- $\Delta C_H = C(t+H) - C(t)$

This makes your labels causal in time:

- Features should come from information available at/at-below $t$
- Labels come from what happens after $t$

### Recommended horizons (practical, given your constraints)

Because you can re-scrape daily for ~7 days:

- **MVP label**: use **$H = 7$** days (next-7-day growth).

For older videos, a short horizon can be near-zero and noisy. When you later have more history, use an **adaptive horizon**:

- Age < 14 days: $H = 3$–7 days
- 14–90 days: $H = 14$–30 days
- > 90 days: $H = 30$–90 days

---

## Concrete engagement targets (formulas)

### 1) Growth target (reach / attention)

Use a log-scaled rate to reduce outlier dominance:

$$
y_{\text{growth}} = \log\left(1 + \frac{\Delta V_H}{H}\right)
$$

### 2) Depth/impact proxy target (interaction intensity)

Normalize interactions by exposure over the same horizon:

$$
r_{\text{interaction}} = \frac{\Delta L_H + w\cdot \Delta C_H}{\max(\Delta V_H, 1)}
$$

- Use $w > 1$ because comments are usually “deeper” than likes.
- A reasonable starting range is $w \in [3, 5]$. Tune later using validation.

Optionally apply a log transform for stability:

$$
y_{\text{depth}} = \log(\epsilon + r_{\text{interaction}})
$$

### 3) Single composite `engagement_score` (optional)

If you want one scalar target for ranking, combine standardized components:

$$
\text{engagement\_score} = z(y_{\text{growth}}) + \alpha \cdot z(y_{\text{depth}})
$$

- $z(\cdot)$ is standardization (mean 0, std 1) across training examples
- $\alpha$ controls the growth vs depth trade-off (start at $\alpha = 1$)

---

## Handling irregular snapshots (without bias)

### Do not use “last two rows only”

When spacing is irregular, “difference between last two snapshots” can be extremely noisy.

Prefer one of:

- **Fixed future horizon deltas**: $\Delta$ between $t$ and $t+H$
- **Slope over a window** (more robust): fit a simple linear regression to `(snapshot_date -> view_count)` over the last W days and use the slope as velocity.

### Always normalize by time

If you compute a delta, always compute the time gap:

- `days = (snapshot_date_2 - snapshot_date_1).total_seconds() / 86400`
- rate = delta / max(days, small_eps)

---

## Bac (3AS) relevance: why you need video-level filtering

A channel-level `subject` is useful for discovery, but it is not enough to guarantee that each **video** targets Bac (3AS).

Therefore, the recommended pipeline is:

1) **Collect broadly** (raw)
2) **Filter/tag at video level** (feature engineering)
3) Train the engagement/advisor model only on relevant videos

---

## Bac relevance filter (metadata-only, Arabic/French-friendly)

Use `text = (title + " " + description + " " + tags).lower()` and apply keyword rules.

### Positive triggers (include)

Flag as Bac candidate if any of these appear:

- Arabic: `بكالوريا`, `باك`, `شهادة البكالوريا`
- Grade: `3as`, `ثالثة ثانوي`, `السنة الثالثة ثانوي`
- French (if present): `bac`, `terminale`

**Note**: year tokens like `2026` should not be used alone—only as supporting context near Bac terms (e.g., `بكالوريا 2026`).

### Negative grade triggers (exclude)

If the video explicitly targets lower grades, exclude:

- Arabic: `سنة اولى ثانوي`, `سنة ثانية ثانوي`
- Grade: `1as`, `2as`
- Other: `متوسط`, `ابتدائي`, `1am/2am/3am/4am`

### Conflict rule (when both appear)

If both Bac and lower-grade keywords appear:

- Keep it **only** if it clearly says it’s a Bac resource (e.g., “مراجعة بكالوريا”).
- Otherwise exclude (to reduce false positives).

### Extract a `grade_level` label (recommended)

Parse grade mentions and create `grade_level ∈ {1AS, 2AS, 3AS, unknown}`.

This enables:

- Clean filtering (Bac-only modeling)
- Future multi-task learning (grade + engagement)

---

## Filtering the raw video table into a modeling dataset

Assuming your raw data resembles:

`data/raw/videos_metadata.csv`
- `video_id, title, description, publish_date, channel_id, duration_sec, view_count, like_count, comment_count, tags, snapshot_date, ...`

### Step A — Deduplicate snapshots (keep all, but ensure uniqueness)

Enforce a uniqueness key:

- (`video_id`, `snapshot_date`)

### Step B — Choose label anchor(s)

For each `video_id`, choose baseline snapshot $t$ and horizon $H$, then compute deltas using the nearest snapshot at/after $t+H$.

### Step C — Create relevance tags

Compute:

- `is_bac_candidate` (rule-based)
- `grade_level` (parsed)

Then filter for Bac modeling:

- Keep `grade_level == "3AS"` or `is_bac_candidate == True` (depending on how strict you want to be).

### Step D — Build modeling rows

Build a training row keyed by `(video_id, baseline_snapshot_date)` with:

- Features from metadata (title length, duration, publish hour/day, tag count, etc.)
- Optional “early performance” features if available at baseline (e.g., counts at $t$)
- Labels from deltas on $[t, t+H]$: `y_growth`, `y_depth`, optional `engagement_score`

### Step E — Prevent domination by huge channels

When training/evaluating:

- Use a **group split by `channel_id`** for validation.
- Downsample or cap per-channel examples (or use sample weights) so large channels do not dominate.

---

## Notes & pitfalls

- **CSV parsing**: descriptions may contain newlines (quoted multi-line fields). Always load with a proper CSV parser (e.g., pandas), not line-based parsing.
- **Near-zero deltas**: for older videos over short horizons, $\Delta$ can be ~0. Treat this as a “flat” regime; it can still be useful (predicting stagnation), but avoid dividing by tiny values and avoid over-optimizing noise. Adaptive horizons or slope-based velocity help.

