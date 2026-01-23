# Capstone Presentation Content
**AI-Driven Analytics and Optimization System for Educational Content**

> **Total Duration:** 15 minutes | **Slides:** ~17 | **Team:** [Your Names]

---

## Section 1: Introduction (3 Slides | ~3 mins)

### Slide 1.1: Hook & Problem Statement

**Slide Title:** The Hidden Challenge of Educational Content Visibility

**Bullet Points:**
- 📚 Algeria's Baccalaureate exam: **636,000+ students** annually competing for university access
- 📺 Algerian Bac YouTube channels: **Millions of views**, but wildly inconsistent engagement
- ❓ **The Problem:** What makes some educational videos thrive while others get buried?

**Speaker Notes:**
> "Picture this: A dedicated teacher spends 8 hours creating a chemistry lesson video, uploads it—and it gets just 500 views. Meanwhile, another video on the same topic from the same channel reaches 100,000 views. Why? This isn't luck. There are patterns—and we set out to discover them."

**Visual Suggestion:**
- Bar chart showing engagement disparity across videos (use engagement distribution from `01_data_exploration.ipynb`)
- Or: Two YouTube thumbnail mockups side-by-side with dramatic view count difference

---

### Slide 1.2: Project Objectives

**Slide Title:** Our Mission: From Guessing to Data-Driven Content

**Bullet Points:**
- 🎯 **Analyze** historical success factors from top-performing Bac videos
- 🔮 **Predict** potential engagement *before* publishing using ML
- 💡 **Recommend** actionable improvements via RAG-based AI agent

**Speaker Notes:**
> "We built a three-phase system. First, we analyze what works. Then, we predict how a new video will perform. Finally, an AI agent synthesizes this knowledge to give creators concrete suggestions—like a virtual data scientist in their pocket."

*Reference: `docs/Project Architecture/Goal of project.md` lines 10-12 for the three pillars*

**Visual Suggestion:**
- Three-phase pipeline diagram (Analysis → Prediction → Recommendation)
- Use the Mermaid diagram from `docs/description of the project.md`

---

### Slide 1.3: Real-World Impact

**Slide Title:** Who Benefits & Why It Matters

**Bullet Points:**
- 👨‍🏫 **Content Creators:** Stop wasting effort on unseen videos
- 📈 **Students:** Better access to quality, engaging educational material
- 🏛️ **Education System:** Data-driven insights for the Algerian Bac ecosystem

**Speaker Notes:**
> "Think of a first-year teacher who just started their YouTube channel. They upload 10 videos—9 fail, 1 succeeds. With our system, they could identify *why* that one video worked and replicate its success intentionally. That's not just efficiency—it's democratizing content creation expertise."

**Concrete Example:**
> "Imagine Yacine, a physics teacher in Oran. Our model might tell him: 'Your intro is 90 seconds—top-performing videos average 45 seconds. Your title lacks the keyword 'درس' (lesson). Fix these, and your predicted engagement jumps from Low to Medium.'"

**Visual Suggestion:**
- Persona card/avatar of a hypothetical teacher
- Impact arrow diagram showing time saved

---

## Section 2: Data Pipeline (3 Slides | ~3 mins)

### Slide 2.1: Data Sources & Collection

**Slide Title:** Building the Knowledge Base

**Bullet Points:**
- 📊 **Source:** YouTube Data API v3 (quota-optimized)
- 🎥 **Volume:** 10,030 videos from 36 Bac-focused channels
- 📝 **Transcripts:** ~8,000 videos with Arabic/French subtitles extracted

**Speaker Notes:**
> "We collected from 36 channels specifically curated for Algerian Bac 3AS content. Our collection script in `src/data/youtube_collector.py` uses the 'uploads playlist' method—1 API unit instead of 100 for search—allowing us to collect more within YouTube's daily 10,000-unit limit."

*Reference: `src/data/youtube_collector.py` and `src/data/transcript_collector.py`*

**Visual Suggestion:**
- Database icon → API arrow → CSV files stack diagram
- Show actual file sizes: `videos_metadata.csv` (41MB), `transcripts_merged.csv` (200MB)

---

### Slide 2.2: Dataset Statistics

**Slide Title:** The Numbers Behind Our Analysis

**Bullet Points:**
| Metric                | Value                                                                      |
| --------------------- | -------------------------------------------------------------------------- |
| Total Videos          | 10,030                                                                     |
| Channels              | 36                                                                         |
| Subjects              | 9 (Math, Physics, French, English, Philosophy, History, Sciences, Islamic, Arabic) |
| Time Period           | 2019-2026                                                                  |
| Transcripts Collected | ~5000                                                                     |
| Valid Transcripts     | ~4,500 (after cleaning)                                                    |

**Speaker Notes:**
> "Our final dataset of 10,030 videos spans 7 years of content. We have 8 academic subjects represented, with Math and Physics being the most common. About 65% of videos have valid transcripts—the others either had no captions or were corrupted YouTube metadata."

*Reference: `data/modeling/split_metadata.json` for exact counts*

**Visual Suggestion:**
- Pie chart: Subject distribution
- Timeline bar: Videos per year/semester

---

### Slide 2.3: Challenges & Solutions

**Slide Title:** Data Engineering Hurdles We Overcame

**Bullet Points:**
- ⚠️ **Challenge 1:** Corrupted transcripts (JS/HTML code instead of text)
  - ✅ **Solution:** Corruption detection in `clean_data.py` (lines 192-209)
- ⚠️ **Challenge 2:** Filtering Bac-3AS from general content
  - ✅ **Solution:** Hybrid classifier with TF-IDF + Channel Priors
- ⚠️ **Challenge 3:** Missing engagement ground truth (no Watch Time access)
  - ✅ **Solution:** Engineered custom Engagement Score formula

**Speaker Notes:**
> "Real-world data is messy. Some transcripts were just YouTube's internal JavaScript code—not actual speech. We built a corruption detector that checks for markers like 'window.', 'ytcfg', and 'javascript' in the text. If 2+ markers are found, we reject that transcript."

*Code Evidence:*
```python
# From src/data/clean_data.py lines 192-201
corruption_markers = [
    'window.', 'ytcfg', 'u003d', 'u0026', 'javascript', 
    'var ', 'function(', '<html', 'innertubeapi'
]
def is_corrupted(text):
    score = sum(1 for m in corruption_markers if m in text.lower())
    return score >= 2
```

**Visual Suggestion:**
- Side-by-side: Raw corrupted text vs. Clean transcript
- Flow diagram: Raw Data → Validation → Clean Data

---

## Section 3: Analytical Approach (4 Slides | ~4 mins)

### Slide 3.1: Data Cleaning & Preprocessing

**Slide Title:** From Raw Data to ML-Ready Features

**Bullet Points:**
- 🔧 **Missing Values:** Text fields → empty string; Numerics → 0/NaN handling
- 🚿 **Corruption Removal:** ~1,500 corrupted transcripts filtered out
- 🔄 **Type Consistency:** Datetime parsing, numeric coercion
- ✅ **Validation:** `has_transcript` flag for downstream features

**Speaker Notes:**
> "The `clean_videos()` function in `clean_data.py` handles all imputation. We fill missing descriptions with empty strings—because 'no description' is itself a signal. For transcripts, we explicitly mark videos without captions using a `has_transcript` flag, which becomes a feature in our model."

*Reference: `src/data/clean_data.py` function `clean_videos()` lines 32-159*

**Visual Suggestion:**
- Before/After table showing null counts
- Code snippet: `df['description'] = df['description'].fillna("")`

---

### Slide 3.2: Feature Engineering (Part 1 - Temporal & Text)

**Slide Title:** Extracting Signal from Metadata

**Bullet Points:**
- ⏰ **Temporal Features:**
  - `publish_hour`, `publish_day_of_week` (cyclic sin/cos encoding)
  - `is_weekday`, `is_evening_upload` (17:00-21:00)
  - `days_since_publish`, `days_since_publish²` (polynomial decay)
- 📝 **Text Features:**
  - `title_length`, `description_length`, `tag_count`
  - `is_exam_focused` (keywords: revision, sujet, corrigé)

**Speaker Notes:**
> "We use cyclic encoding for hours—so 23:00 and 00:00 are mathematically close. This captures the fact that uploading at 11 PM versus midnight isn't that different. We also detect 'exam-focused' videos using keywords like 'bac', 'revision', 'sujet'—these tend to have higher engagement near exam seasons."

*Reference: `src/features/engineer.py` methods `_create_temporal_features()` lines 163-253 and `_create_text_features()` lines 255-326*

**Visual Suggestion:**
- Sin/Cos wave diagram for cyclic hour encoding
- Feature table with example values

---

### Slide 3.3: Feature Engineering (Part 2 - Transcript & Engagement)

**Slide Title:** Deep Linguistic Analysis

**Bullet Points:**
- 📖 **Transcript Features (24 features):**
  - Readability: Flesch-Kincaid, Gunning Fog, ARI scores
  - Pacing: `speech_rate_wpm`, `speech_rate_optimal` (120-180 wpm)
  - Pedagogy: `question_density`, `example_density`, `explanation_count`
- 🎯 **Target Variable:**
  $$\text{Engagement Score} = \ln\left(1 + \frac{(3 \times \text{Comments}) + \text{Likes}}{\sqrt{\text{Views}}}\right)$$

**Speaker Notes:**
> "We extract linguistic features like Flesch Reading Ease—measuring how accessible the language is. We also count pedagogical markers: questions (indicating interactive teaching), examples ('مثال', 'par exemple'), and explanations ('لأن', 'parce que'). Our engagement formula penalizes viral but shallow views by using square root normalization, while log-transforming to handle social media's heavy-tailed distributions."

*Reference: `src/features/transcript_features.py` function `extract_transcript_features()` lines 38-214*

**Visual Suggestion:**
- Feature category breakdown diagram
- Engagement Score formula with annotation arrows

---

### Slide 3.4: Model Selection & Training

**Slide Title:** Choosing the Right Algorithm

**Bullet Points:**
- 🧪 **Models Tested:**
  - Random Forest, XGBoost, LightGBM
- 📐 **Feature Strategy:** 40 numerical + 100 TF-IDF + 100 AraBERT embeddings = **908 total features**
- 📊 **Data Split:** 60% Train (6,018) / 20% Val (2,006) / 20% Test (2,006)
- 🎯 **Target:** `engagement_score` (regression task)

**Speaker Notes:**
> "We treat this as a regression problem—predicting a continuous engagement score. We combine traditional numerical features with TF-IDF vectors of titles and AraBERT embeddings for richer semantic understanding of Arabic content. All three models were trained on identical splits for fair comparison."

*Reference: `data/modeling/split_metadata.json` for feature list and split sizes*

**Visual Suggestion:**
- Model architecture diagram showing feature fusion
- Train/Val/Test split bar

---

## Section 4: Solution and Results (3 Slides | ~3 mins)

### Slide 4.1: Model Comparison

**Slide Title:** Head-to-Head Evaluation

**Bullet Points:**
| Model             | MAE   | RMSE  | R² (Validation) |
| ----------------- | ----- | ----- | --------------- |
| **Random Forest** | 0.367 | 0.482 | 0.580           |
| LightGBM          | 0.375 | 0.486 | 0.572           |
| XGBoost           | 0.374 | 0.491 | 0.564           |

**Winner:** Random Forest (best R² and MAE)

**Speaker Notes:**
> "On the validation set, Random Forest narrowly edges out the competition with 58% of variance explained. All three models perform similarly—suggesting our features are robust and the problem is learnable. The MAE of 0.37 means on average we're off by less than half a point on our engagement scale."

*Reference: `data/modeling/baseline_results.csv`*

**Visual Suggestion:**
- Grouped bar chart comparing MAE/RMSE/R²
- Highlight Random Forest bar in green

---

### Slide 4.2: Final Test Performance

**Slide Title:** Held-Out Test Results

**Bullet Points:**
- 🏆 **Best Model:** Random Forest
- 📈 **Test R²:** **0.625** (62.5% variance explained)
- 📉 **Test MAE:** **0.353**
- 📊 **Test RMSE:** **0.463**

**Speaker Notes:**
> "On the completely unseen test set, our model actually *improves* to 62.5% R²—indicating good generalization. A MAE of 0.35 means for a typical video, our predicted engagement is quite close to reality. This is strong for social media prediction, where random factors play a huge role."

*Reference: `data/modeling/test_results.json`*

**Visual Suggestion:**
- Actual vs. Predicted scatter plot (if available from notebook)
- Big R² = 0.625 stat callout

---

### Slide 4.3: Key Insight Discovered

**Slide Title:** What Drives Engagement?

**Bullet Points:**
- 🔑 **Top Feature:** `speech_rate_optimal` → Videos at 120-180 words/min get higher engagement
- 📝 **Title Sweet Spot:** 50-70 characters outperform very short or very long titles
- ⏰ **Best Upload Time:** Evening (17:00-21:00) on weekdays
- ❓ **Pedagogical Signal:** Higher `question_density` correlates with engagement

**Speaker Notes:**
> "One fascinating insight: speech rate matters. Videos where the teacher speaks at 120-180 words per minute—the optimal learning pace—consistently outperform fast or slow speakers. This is actionable: teachers can literally slow down their delivery to improve engagement."

*Insight from: Feature importance analysis in `02_model_baseline_testing.ipynb`*

**Visual Suggestion:**
- Feature importance bar chart (top 10)
- Callout box for speech rate insight

---

## Section 5: Live Demonstration (2 Slides + Demo | ~3 mins)

### Slide 5.1: System Architecture

**Slide Title:** How It All Connects

**Bullet Points:**
- 🔧 **Backend:** Python CLI pipeline (`run_pipeline.py`)
- 📊 **Data Layer:** CSV artifacts in `data/` folders
- 🤖 **ML Models:** Pickled in `models/` (Random Forest, LightGBM, XGBoost)
- 🧠 **RAG Agent:** LLM + Vector DB (planned GenAI layer)

**Speaker Notes:**
> "Our system is orchestrated by `run_pipeline.py`—a unified CLI that chains: collect → clean → engineer → train → predict. Models are serialized as pickle files. The RAG agent (Phase 3) will query a vector database of best practices to generate personalized recommendations."

*Reference: `run_pipeline.py` outline shows `cmd_collect`, `cmd_engineer`, `cmd_filter_data`, etc.*

**Visual Suggestion:**
- System architecture flowchart
- Terminal screenshot of `uv run python run_pipeline.py --help`

---

### Slide 5.2: Demo Scenarios

**Slide Title:** Live Prediction Demo

**Demo Script (2-3 scenarios):**

1. **Scenario A: High-Engagement Prediction**
   - Input: Title = "شرح درس الاشتقاق - باك 2026 رياضيات"
   - Duration = 25 mins, Tags = bac, math, derivative
   - Expected Output: High engagement score prediction (~2.5+)

2. **Scenario B: Low-Engagement Warning**
   - Input: Title = "video" (generic)
   - Duration = 90 mins (too long), No tags
   - Expected Output: Low engagement score prediction (~0.5)

3. **Scenario C: Optimization Suggestion (RAG Agent)**
   - Input: Same as Scenario B
   - RAG Output: "Add 'bac' keyword to title, reduce duration to <45min, add subject tags"

**Speaker Notes:**
> "Let's see this live. I'll input a well-optimized physics video title with proper tags—watch the predicted score. Then I'll show the same topic with a poor title and no tags—notice how the prediction drops. Finally, our RAG agent will suggest specific fixes."

> [!CAUTION]
> **Backup:** Record a 1-minute video of this exact flow in case of demo failure.

**Visual Suggestion:**
- Split screen: Input form → Prediction output
- Terminal showing actual prediction values

---

## Section 6: Reflection & Future Work (2 Slides | ~2 mins)

### Slide 6.1: Limitations

**Slide Title:** Honest Assessment of Constraints

**Bullet Points:**
1. **Data Limitation:** No access to YouTube's Watch Time or Impressions (private metrics)
   - *Impact:* Engagement Score is a proxy, not ground truth
2. **Technical Limitation:** Transcripts rely on YouTube's auto-generated captions
   - *Impact:* ~35% of videos lack usable transcripts
3. **Scope Limitation:** Model trained on Bac 3AS content only
   - *Impact:* May not generalize to other educational niches (BEM, university)

**Speaker Notes:**
> "We must be honest. YouTube doesn't expose watch time via API—so our engagement metric, while principled, is an approximation. Also, 35% of videos had no captions, limiting our transcript analysis. Finally, this model is specific to Algerian Bac content—it's not a general-purpose predictor."

**Visual Suggestion:**
- Honest limitations table with severity indicators
- Gap icon or warning symbol

---

### Slide 6.2: Future Improvements

**Slide Title:** The Road Ahead

**Bullet Points:**
1. **Technical Enhancement:** Fine-tune AraBERT specifically on educational content
   - *Benefit:* Better Arabic NLP understanding
2. **Feature Addition:** Extract audio features (tone, pace variation) from videos
   - *Benefit:* Capture delivery quality beyond text
3. **Product Evolution:** Build web interface for creators to self-test videos
   - *Benefit:* Democratize access beyond command-line users
4. **Scope Expansion:** Extend to BEM (middle school) and university content

**Speaker Notes:**
> "For the future: we'd love to fine-tune AraBERT on our specific domain—educational Arabic. We could also extract audio features like tone variation and pauses to capture delivery quality. Most importantly, a web app would let teachers who don't code benefit from our predictions directly."

**Visual Suggestion:**
- Roadmap timeline (6-12 months)
- Icon for each improvement

---

## Appendix: Quick Reference

### Key Files Referenced

| Section                | File                                  | Purpose                              |
| ---------------------- | ------------------------------------- | ------------------------------------ |
| Data Collection        | `src/data/youtube_collector.py`       | API-optimized video harvesting       |
| Transcript Collection  | `src/data/transcript_collector.py`    | Subtitle extraction with Tor support |
| Data Cleaning          | `src/data/clean_data.py`              | Null handling, corruption detection  |
| Feature Engineering    | `src/features/engineer.py`            | 40+ numerical features               |
| Transcript Features    | `src/features/transcript_features.py` | Readability, pedagogy markers        |
| Pipeline Orchestration | `run_pipeline.py`                     | Unified CLI                          |
| Model Results          | `data/modeling/baseline_results.csv`  | Validation metrics                   |
| Test Results           | `data/modeling/test_results.json`     | Final test performance               |

### Key Metrics to Memorize

- **Videos:** 10,030
- **Channels:** 36
- **Features:** 908 (40 numerical + 100 TF-IDF + 100 AraBERT + one-hot encodings)
- **Best Model:** Random Forest
- **Test R²:** 0.625
- **Test MAE:** 0.353

---

> *"Success is not about having the best model, but about telling the best story with your data."*
> — Final Presentation Guide
