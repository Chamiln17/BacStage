## Overview

This guide documents a **balanced filtering strategy** for Algerian YouTube education data where **most channels are Bac-focused**, but you still want:

- **high quality** (low noise)
- **reasonable dataset size** (good recall)
- **minimal manual keyword maintenance**

The approach is **pipeline-first** and uses your existing artifacts:

- Raw snapshots: `data/raw/videos_metadata.csv`
- Engineered features: `data/processed/videos_engineered.csv`
- Existing filtering outputs (optional): `data/processed/videos_with_bac_filter.csv`

The core idea is:

- **Channel prior (Bac by default)** for Bac-heavy channels
- **Hard exclusions** for strong evidence of non‑Bac grade
- **Soft positives** for ambiguous cases (so we don’t keep everything)
- **Data-driven discovery** of missing title keywords via **TF‑IDF** (to reduce manual work)

---

## Why the previous filter felt “too strict”

In practice, many Bac videos have titles like:
- “حل الموضوع 2”
- “سلسلة رقم 3”
- “منهجية الإجابة”

These often **don’t contain explicit subject terminology**, so a strict “subject keywords required” gate will reject good Bac videos.

Because your channels are largely Bac-specific, your filter should behave more like:

- **keep by default**, unless there’s clear evidence it’s for another grade
- **don’t require subject detection to accept a Bac video**

---

## Goal definition (what counts as “Bac 3AS”)

We want to tag/filter videos into:

- **bac_3as**: clearly Bac / 3AS focused
- **non_bac**: clearly for other grades (1AS/2AS/middle school/primary)
- **unknown**: ambiguous (no clear grade evidence)

Then we choose a final dataset using a balanced rule:

- keep `bac_3as`
- keep a subset of `unknown` only when “soft positives” suggest Bac relevance
- exclude `non_bac`

---

## Inputs: what columns we rely on

### From `data/raw/videos_metadata.csv`

Used for filtering:
- `title`, `description`, `tags` (text)
- `channel_id` (channel prior)
- `duration_sec` (optional quality gate)
- `snapshot_date` (for dedupe)

### From `data/processed/videos_engineered.csv` (optional but recommended)

Used as *soft positives*:
- `subject` (light subject detection already implemented)
- `is_exam_focused` (cheap exam-intent signal already implemented)

---

## Step 0 — Dedupe snapshots (filter “videos”, not “snapshots”)

Your collection stores multiple rows per `video_id` across `snapshot_date`.
Filtering should usually be applied on **latest snapshot per video**:

- Sort by `snapshot_date` desc
- Keep first row per `video_id`

This reduces noise and makes channel/grade logic consistent.

---

## Step 1 — Build a small grade marker set (high signal, low maintenance)

### Positive markers (Bac / 3AS)

Start with a small list and expand only when you observe misses:

- **Arabic**: `بكالوريا`, `باك`, `شهادة البكالوريا`, `ثالثة ثانوي`, `السنة الثالثة ثانوي`
- **Latin**: `bac`, `3as`, `terminale`

### Negative markers (non‑Bac)

Use *specific* phrases where possible (avoid overly broad “سنة أولى” alone):

- **Secondary lower**: `1as`, `2as`, `سنة أولى ثانوي`, `سنة ثانية ثانوي`, `أولى ثانوي`, `ثانية ثانوي`
- **Middle school**: `متوسط`, `bem`, `1am`, `2am`, `3am`, `4am`
- **Primary**: `ابتدائي`

### Strong Bac intent phrases (conflict resolver)

These help when both Bac and non‑Bac markers appear (often in descriptions):

- Arabic examples: `مراجعة بكالوريا`, `تحضير بكالوريا`, `تصحيح بكالوريا`, `موضوع بكالوريا`, `حل موضوع بكالوريا`
- French examples: `bac blanc`, `révision bac`, `corrigé bac`, `sujet bac`

---

## Step 2 — Build a channel prior (data-driven, minimal manual labeling)

Because your channels are mostly Bac-focused, we use channel behavior as a prior:

### 2.1 Compute per-channel grade evidence rates

For each channel, compute grade-evidence counts from titles/descriptions/tags:

- `count_bac_marked`: videos containing Bac markers
- `count_non_bac_marked`: videos containing non‑Bac markers
- `count_any_grade_marked`: videos containing either

Then compute:

\[
p_{bac} = \frac{count\_bac\_marked}{max(count\_any\_grade\_marked, 1)}
\]
\[
p_{non} = \frac{count\_non\_bac\_marked}{max(count\_any\_grade\_marked, 1)}
\]

### 2.2 Define “Bac-heavy channels”

Mark a channel “Bac-heavy” if it satisfies something like:

- `p_bac` is high (e.g., ≥ 0.7)
- and `p_non` is low (e.g., ≤ 0.1)

You don’t have to hardcode thresholds upfront—inspect the distribution and pick a breakpoint.

Why this works:
- It matches your claim (“most educators are Bac specific”) using evidence from your own dataset.
- It avoids manually maintaining a whitelist.

---

## Step 3 — The balanced filtering rules (high quality + non-small dataset)

We apply a decision tree in this order.

### Rule 1: Hard exclude (strong non‑Bac evidence)

If explicit non‑Bac grade markers are present:

- **Exclude** by default (label as `non_bac`)
- **Except** if there is strong Bac intent in the title (conflict override)

This prevents obvious leakage from 1AS/2AS/middle school into your Bac set.

### Rule 2: Hard include (strong Bac evidence)

If explicit Bac markers are present:

- **Include** (label as `bac_3as`)

Important: **Do not require subject detection** for this case.

This directly fixes the “good Bac video but generic title” problem as long as the title/desc includes a Bac marker somewhere.

### Rule 3: Ambiguous cases → channel prior + soft positives

If neither Bac nor non‑Bac markers appear:

- If channel is **not** Bac-heavy → label `unknown` (usually drop)
- If channel **is** Bac-heavy → keep only if the video has **soft positives**

#### Soft positives (quality gate)

For ambiguous videos in Bac-heavy channels, require at least one:

- **Exam intent**: `is_exam_focused == 1` (from engineered features)
- **Subject not “General”**: `subject != "General"` (from engineered features)
- **Duration gate** (optional): `duration_sec >= 300` (5+ minutes)
- **TF‑IDF “Bac-ish” token hit**: title contains one of the high-signal terms discovered from TF‑IDF

This ensures we don’t “keep everything from Bac-heavy channels” and introduce noise.

---

## Step 4 — TF‑IDF keyword discovery (reduce manual work)

Your idea is good, but it only works if we set it up correctly.

### 4.1 Use high-precision pseudo-labels

Do not manually label everything. Instead:

- **Positive set (Bac)**: titles containing Bac markers and NOT containing non‑Bac markers
- **Negative set (non‑Bac)**: titles containing non‑Bac markers and NOT containing Bac markers

These are “clean enough” seeds for discovery.

### 4.2 Vectorization strategy

Because your titles include Arabic + French + mixed spellings:

- Use **word n‑grams (1,2)** to capture phrases like “bac blanc”, “موضوع بكالوريا”
- Use **character n‑grams (3,5)** to capture spelling variants and Arabic forms

### 4.3 What to extract

Avoid “top TF‑IDF overall”—you want *discriminative* terms.

Good ranking options:

- **Difference of means**: average TF‑IDF in positives minus average TF‑IDF in negatives
- **Linear model weights**: train a simple logistic regression and take the highest positive coefficients

Then:

- Take the top ~30–100 terms as your **TF‑IDF-derived soft positive dictionary**
- These terms become a “hit list” for ambiguous titles in Bac-heavy channels

### 4.4 Why TF‑IDF helps here

It automatically discovers common Bac title patterns in your data, such as:

- “تصحيح”, “حل”, “سلسلة”, “منهجية”, “موضوع”, “bac blanc”, “révision”, etc.

You only need to review the extracted list once and remove obvious junk tokens.

---

## Step 5 — Validation (mandatory)

Filtering is only “good” if validated.

Use stratified sampling:

- sample from `bac_3as` predictions (precision check)
- sample from `unknown` kept by soft positives (noise check)
- sample from `non_bac` rejections (false negative check)
- sample edge cases where both Bac and non‑Bac markers exist (conflict rule check)

You already have a sampling script and review template:
- `data/validation/sample_for_manual_review.csv`

---

## Tuning knobs (how to balance size vs quality)

The safest way to tune is to adjust **one knob at a time**:

### Knob A — “Bac-heavy channel” threshold

- Increasing strictness (higher `p_bac` requirement) improves precision but reduces recall.
- Relaxing it increases dataset size but may include mixed channels.

### Knob B — Soft positive requirement

From lenient → strict:

- **Lenient**: keep ambiguous if `subject != "General" OR is_exam_focused == 1`
- **Balanced**: keep ambiguous if `is_exam_focused == 1` OR `tfidf_hit == 1`
- **Strict**: keep ambiguous if `tfidf_hit == 1` AND `duration_sec >= 300`

### Knob C — Conflict override strictness

If both Bac and non‑Bac markers appear:

- **Balanced**: keep only if strong Bac intent phrase appears
- **Lenient**: keep if Bac marker appears in the title (not only description/tags)

---

## Recommended default configuration (balanced)

If your channels are truly Bac-heavy and you want quality without shrinking too much:

- **Hard include**: any Bac marker → keep
- **Hard exclude**: any non‑Bac marker → exclude
- **Conflict override**: keep only if strong Bac intent phrase exists (e.g., “مراجعة بكالوريا” / “bac blanc”)
- **Ambiguous (no markers)**:
  - keep only if channel is Bac-heavy AND (`is_exam_focused==1` OR `tfidf_hit==1`)
  - optionally require `duration_sec >= 300` if you see short-form noise

This typically increases recall vs “subject keywords required”, while still preventing low-signal ambiguous videos from flooding your dataset.

---

## Debugging checklist (what to inspect when results look wrong)

- **Too small dataset**:
  - Check how many videos are `unknown` because they have no markers
  - Relax soft positives: allow `subject != "General"` as a soft positive
  - Add a few missing Bac markers seen in real titles (minimal list expansion)

- **Too noisy dataset**:
  - Tighten ambiguous acceptance: require `tfidf_hit`
  - Add duration gate `duration_sec >= 300` (or 600)
  - Tighten conflict override: require strong Bac intent phrase

- **Many false negatives**:
  - Look at excluded videos from Bac-heavy channels and see why they were excluded
  - If descriptions contain “1AS/2AS/3AS”, apply conflict rule (don’t auto-exclude on non‑Bac markers unless the title strongly targets that grade)

---

## How this integrates with the rest of your ML project

Once you have a filtered dataset:

- Use it as the modeling base instead of all collected videos.
- Keep `channel_id` and `publish_date` for proper splitting and leakage control.
- Continue using snapshot logic from `TARGETS_AND_FILTERING_GUIDE.md` for leakage-safe targets.

Recommended practice:
- Train/validate with **group split by `channel_id`** so one channel doesn’t dominate and leak style patterns into validation.
