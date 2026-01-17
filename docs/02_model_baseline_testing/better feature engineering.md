# Content-Based Feature Engineering for Pre-Publication Educational Video Engagement: Comprehensive Recommendations

Given your constraint of **pre-publication prediction** (no post-publication engagement signals), combined with your **10,000+ Algerian Bac video dataset** and access to **transcripts**, I recommend a strategic three-tier feature engineering approach that extracts rich pedagogical signals while avoiding data leakage. Your current 25-feature model provides a solid temporal/metadata baseline; below are actionable recommendations for **28+ new content-based features** that should meaningfully improve prediction accuracy.

***

## 1. Core Problem: Your Current Feature Set is Incomplete

Your existing 25 features capture **when and where** videos are published but miss **what content** they contain. This is critical for educational engagement because:

- **Research shows:** Readability, concept density, and instructional clarity (extracted from transcripts/titles) correlate more strongly with engagement than publishing timing[^1][^2]
- **Pre-publication context:** You can't use view/like/comment counts; transcript and metadata are your only content signals
- **Algerian Bac specificity:** Educational videos succeed when aligned with curriculum + appropriately explained for 16-18 year-old learners

Your `is_exam_focused` feature is a good start, but it's binary. You need a **richer content representation**.

***

## 2. Recommended Feature Architecture (Three Tiers)

### **TIER 1: HIGH-IMPACT TRANSCRIPT FEATURES (Implement First)**

These 12 features are extractable with standard NLP libraries and should drive 40-50% of model improvement:

#### **A. Speech Rate \& Clarity Indicators**

| **Feature** | **Extraction Method** | **Rationale** | **Optimal Range** |
| :-- | :-- | :-- | :-- |
| `speech_rate_wpm` | word_count / (duration_sec - 60) × 60 | Natural pace aids comprehension; too fast overwhelms[^1] | 120–160 WPM |
| `speech_rate_above_optimal` | 1 if speech_rate > 180, else 0 | Flag rushed delivery | N/A (binary flag) |
| `speech_rate_below_optimal` | 1 if speech_rate < 100, else 0 | Flag sluggish pace | N/A (binary flag) |

**Implementation (Python):**

```python
def extract_speech_features(transcript, duration_sec):
    words = transcript.split()
    effective_duration = max(duration_sec - 60, 1)  # Exclude intro/outro
    speech_rate_wpm = (len(words) / effective_duration) * 60
    return {
        'speech_rate_wpm': speech_rate_wpm,
        'above_optimal': 1 if speech_rate_wpm > 180 else 0,
        'below_optimal': 1 if speech_rate_wpm < 100 else 0
    }
```


#### **B. Readability \& Linguistic Complexity**

| **Feature** | **Formula / Library** | **Bac Benchmark** | **Interpretation** |
| :-- | :-- | :-- | :-- |
| `flesch_kincaid_grade` | `0.39(W/S) + 11.8(Sy/W) - 15.59` | 7–10th grade (optimal for 16-18yo) | Higher = harder to read |
| `flesch_reading_ease` | `206.835 - 1.015(W/S) - 84.6(Sy/W)` | 60–70 (easy to moderately easy) | Higher = easier; <30 is hard |
| `type_token_ratio` (lexical diversity) | unique_words / total_words | 0.4–0.6 (healthy diversity) | <0.35 = repetitive; >0.7 = scattered |
| `avg_sentence_length` | total_words / sentence_count | 15–20 words (optimal clarity) | Long sentences (>25) increase load |

**Implementation:**

```python
from textstat import flesch_kincaid_grade, flesch_reading_ease
import nltk
from nltk.tokenize import sent_tokenize

def extract_readability_features(transcript):
    fk_grade = flesch_kincaid_grade(transcript)
    fre_score = flesch_reading_ease(transcript)
    
    words = transcript.split()
    unique_words = len(set(w.lower() for w in words))
    ttr = unique_words / len(words) if len(words) > 0 else 0
    
    sentences = sent_tokenize(transcript)
    avg_sent_len = len(words) / len(sentences) if sentences else 0
    
    return {
        'flesch_kincaid_grade': fk_grade,
        'flesch_reading_ease': fre_score,
        'type_token_ratio': ttr,
        'avg_sentence_length': avg_sent_len
    }
```


#### **C. Content Density \& Keyword Extraction**

| **Feature** | **Method** | **Meaning** |
| :-- | :-- | :-- |
| `keyword_density_per_minute` | (TF-IDF keywords / transcript_words) × (duration / 60) | How conceptually rich is the content? Higher = more dense learning |
| `domain_entity_count` | Named Entity Recognition (NER) on curated Bac lexicon | Count of curriculum-aligned concepts (e.g., "photosynthesis", "Newton's law") |
| `entity_recurrence_ratio` | (entities_appearing_2+_times) / total_entities | Deeper coverage when concepts repeated |

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Keyword extraction
vectorizer = TfidfVectorizer(max_features=100, stop_words=['french', 'arabic'])
X = vectorizer.fit_transform([transcript])
keywords = vectorizer.get_feature_names_out()
keyword_density = (len(keywords) / len(transcript.split())) * (duration_sec / 60)

# Domain NER (using spaCy or custom regex on Bac lexicon)
bac_physics_terms = {'force', 'énergie', 'momentum', 'travail', 'cinétique', …}
domain_entities = [w for w in transcript.split() if w.lower() in bac_physics_terms]
```


#### **D. Engagement Proxies: Questions \& Structure**

| **Feature** | **Extraction** | **Why It Matters** |
| :-- | :-- | :-- |
| `question_count` | Count "?" in transcript | Instructors asking questions = active learning pedagogy[^3] |
| `question_density_per_minute` | question_count / (duration_sec / 60) | Frequency indicates dialogue-based teaching |
| `segment_count` | Number of sentences or topic shifts | More segments = better pacing; too many = fragmented |
| `transition_smoothness` | Semantic similarity between consecutive segments (cosine distance) | Abrupt transitions reduce comprehension |

**Implementation:**

```python
from nltk.tokenize import sent_tokenize

questions = [s for s in sent_tokenize(transcript) if '?' in s]
question_density = len(questions) / (duration_sec / 60)

# Topic transitions via word embeddings (optional, Phase 2)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-mean-tokens')
segments = sent_tokenize(transcript)
embeddings = [model.encode(seg) for seg in segments]
transitions = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
transition_smoothness = np.mean(transitions)  # Higher = smoother
```


#### **E. Sentiment \& Tone (Quality Signal)**

| **Feature** | **Tool** | **Range** |
| :-- | :-- | :-- |
| `sentiment_score` | VADER or DistilBERT (multilingual) | -1 (very negative) to +1 (very positive) |
| `tone_label` | Transformer classification | categorical: optimistic, analytical, cautious, critical |

**Implementation:**

```python
from transformers import pipeline

sentiment_pipe = pipeline("sentiment-analysis", 
                        model="distilbert-base-multilingual-cased")
result = sentiment_pipe(transcript[:512])  # Truncate for efficiency
sentiment = 1 if result[^0]['label'] == 'POSITIVE' else -1 if result[^0]['label'] == 'NEGATIVE' else 0
```


***

### **TIER 2: TITLE \& DESCRIPTION FEATURES (High Priority, Moderate Effort)**

These 8 features extract linguistic patterns predictive of engagement:


| **Feature** | **Extraction Method** | **Intuition** |
| :-- | :-- | :-- |
| `title_keyword_count` | TF-IDF on title text | Focused vs. vague titles |
| `description_keyword_count` | TF-IDF on description | Detailed vs. sparse descriptions |
| `title_description_keyword_overlap` | Jaccard similarity of keyword sets | Alignment between promise (title) and detail (description) |
| `is_title_question` | 1 if title ends with "?" | Questions engage more (e.g., "How to solve...?") |
| `has_explanation_markers` | 1 if description contains {learn, explain, understand, step-by-step, tutorial, guide} | Signals pedagogical intent |
| `emotional_word_ratio_title` | (emotional words / total) in title | Engagement appeal without clickbait |
| `exam_keyword_intensity` | (count of {exam, bac, test, revision, préparation}) / title_length | Enhanced signal vs. binary `is_exam_focused` |
| `curriculum_term_presence` | 1 if title/desc contains mapped Bac learning objectives | Explicit curriculum alignment |

**Implementation:**

```python
# Title-description alignment
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(stop_words='french')
title_vec = vec.fit_transform([title])[^0].toarray()
desc_vec = vec.fit_transform([description])[^0].toarray()
overlap = np.dot(title_vec, desc_vec) / (np.linalg.norm(title_vec) * np.linalg.norm(desc_vec) + 1e-6)

# Question framing
is_question = 1 if title.strip().endswith('?') else 0

# Explanation markers
explanation_markers = {'learn', 'explain', 'understand', 'guide', 'how', 'tutorial'}
has_markers = 1 if any(m in description.lower() for m in explanation_markers) else 0

# Exam intensity
exam_keywords = ['exam', 'baccalauréat', 'test', 'révision', 'préparation', 'bac']
exam_intensity = sum(description.lower().count(kw) for kw in exam_keywords) / len(description.split())
```


***

### **TIER 3: STRUCTURE \& COHERENCE FEATURES (Moderate Effort, Specialized Impact)**

These 8 features require slightly more sophisticated NLP but provide nuanced pedagogical signals:


| **Feature** | **Method** | **Meaning** |
| :-- | :-- | :-- |
| `has_strong_intro` | Semantic match: first 60 sec with keywords {learn, today, will explain, objective} | Clear instructional intent up front |
| `has_recap` | Semantic match: last 60 sec with keywords {summary, recap, conclude, remember, key points} | Consolidation signal |
| `intro_recap_length_ratio` | (intro_words + recap_words) / total_words | Sufficient bookending (10-15% ideal) |
| `topic_coherence_score` | LDA topic modeling coherence metric (0-1) | How well-defined are content topics? |
| `topic_diversity` | Number of distinct topics inferred by LDA | Breadth of coverage (higher = wider scope) |
| `primary_topic_dominance` | Probability mass concentration on top-2 topics | 1.0 = highly focused; 0.5 = scattered |
| `causal_connective_density` | (count: because, therefore, as a result, since) / total_words | Logical flow indicator |
| `definition_frequency` | (count: "is defined as", "means", "refers to") / sentence_count | Explicit concept clarification |

**Implementation (Phase 2):**

```python
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary

# LDA Topic Modeling
tokens = [w.lower() for w in transcript.split() if len(w) > 3]  # Simple tokenization
dictionary = Dictionary([tokens])
corpus = [dictionary.doc2bow(tokens)]
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

# Topic coherence
from gensim.models import CoherenceModel
coherence_model = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary)
topic_coherence = coherence_model.get_coherence()

# Dominant topic
doc_topics = lda_model.get_document_topics(corpus[^0])
primary_dominance = sorted(doc_topics, key=lambda x: x[^1], reverse=True)[^0][^1]
```


***

## 3. Handling Multilingual \& Algerian Bac Context

### **Challenge: Arabic-French Mixing + ASR Variability**

Your transcript data likely has:

- High-quality French transcripts (~90% accuracy)
- Lower-quality Arabic or code-switched segments (~70-80% accuracy)
- Algerian dialect variations (less well-represented in standard models)


### **Solutions:**

#### **A. Transcript Confidence Scoring**

```python
def compute_transcript_confidence(transcript, video_duration):
    """
    Heuristic proxy for ASR confidence.
    Real solution: compute Word Error Rate (WER) if reference transcripts available.
    """
    # Flag suspicious patterns
    inaudible_markers = ['[inaudible]', '[unclear]', '[...]']
    suspicious_words = sum(1 for m in inaudible_markers if m in transcript)
    
    # Language detection
    from textblob import TextBlob
    blob = TextBlob(transcript)
    lang_dist = blob.detect_language()  # e.g., 'en', 'fr', 'ar'
    
    # Heuristic confidence: penalize if >40% is inaudible or mixed language
    confidence = 1.0 - (suspicious_words / len(transcript.split())) * 0.3
    return confidence

feature_dict['transcript_confidence_score'] = compute_transcript_confidence(transcript, duration)
```


#### **B. Use Multilingual NLP Models**

```python
# Instead of English-only spaCy model, use multilingual
# Option 1: Multilingual BERT
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

# Option 2: XLM-RoBERTa (supports 100+ languages, including Arabic)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Option 3: Language-specific pipelines
nlp_fr = spacy.load("fr_core_news_sm")
nlp_ar = spacy.load("ar_core_news_sm")  # If available

def process_multilingual(transcript):
    # Detect language per sentence, apply appropriate model
    sentences = sent_tokenize(transcript)
    entities = []
    for sent in sentences:
        lang = TextBlob(sent).detect_language()
        if lang == 'fr':
            doc = nlp_fr(sent)
        elif lang == 'ar':
            doc = nlp_ar(sent)
        else:
            doc = nlp_en(sent)  # Fallback
        entities.extend([ent.text for ent in doc.ents])
    return entities
```


#### **C. Build Curated Algerian Bac Lexicon**

```python
bac_curriculum_terms = {
    'physics': [
        'force', 'énergie', 'momentum', 'travail', 'puissance',
        'قوة', 'طاقة', 'زخم'  # Arabic equivalents
    ],
    'biology': [
        'photosynthèse', 'respiration', 'adaptation', 'évolution',
        'البناء_الضوئي', 'التنفس', 'التطور'
    ],
    # ... add all subjects
}

def extract_bac_terms(transcript):
    """Count curriculum-aligned entities regardless of language."""
    terms_found = []
    for subject, terms in bac_curriculum_terms.items():
        for term in terms:
            if term.lower() in transcript.lower():
                terms_found.append((term, subject))
    return terms_found
```


***

## 4. Feature Selection \& Validation Strategy

### **Step 1: Filter for Multicollinearity**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for all proposed features
vif_data = pd.DataFrame()
vif_data["feature"] = X_features.columns
vif_data["VIF"] = [variance_inflation_factor(X_features.values, i) for i in range(X_features.shape[^1])]

# Keep only VIF < 5
features_low_vif = vif_data[vif_data["VIF"] < 5]["feature"].tolist()
```


### **Step 2: Correlation with Engagement Label**

```python
correlations = {}
for feature in new_features:
    corr, pvalue = spearmanr(df[feature], df['engagement_score'])
    if abs(corr) > 0.10:  # Threshold
        correlations[feature] = corr

# Retain top 20-25 features by absolute correlation
top_features = sorted(correlations.items(), key=lambda x: abs(x[^1]), reverse=True)[:25]
```


### **Step 3: Incremental Model Improvement**

```python
from sklearn.ensemble import XGBClassifier
from sklearn.metrics import f1_score

# Baseline with 25 existing features
baseline_model = XGBClassifier(n_estimators=100, max_depth=6)
baseline_model.fit(X_baseline, y)
baseline_f1 = cross_val_score(baseline_model, X_baseline, y, cv=5, scoring='f1_macro').mean()

# Add TIER 1 features
X_tier1 = pd.concat([X_baseline, X_tier1_features], axis=1)
tier1_model = XGBClassifier(n_estimators=100, max_depth=6)
tier1_f1 = cross_val_score(tier1_model, X_tier1, y, cv=5, scoring='f1_macro').mean()

improvement = tier1_f1 - baseline_f1
print(f"Baseline F1: {baseline_f1:.3f}")
print(f"Tier 1 F1: {tier1_f1:.3f}, Improvement: +{improvement:.3f}")

# Only proceed to Tier 2 if improvement > 1-2%
```


### **Step 4: Subject-Stratified Validation**

```python
# Verify features generalize across Bac subjects
for subject in df['subject'].unique():
    subject_data = df[df['subject'] == subject]
    subject_corr = subject_data[feature].corr(subject_data['engagement_score'])
    print(f"Feature '{feature}' correlation for {subject}: {subject_corr:.3f}")

# Features should have consistent sign of correlation across subjects
```


***

## 5. Recommended Feature Set for MVP (Tier 1 + Select Tier 2)

### **28 New Features to Extract:**

#### **Transcript Features (12):**

1. `speech_rate_wpm`
2. `speech_rate_above_optimal`
3. `flesch_kincaid_grade`
4. `flesch_reading_ease`
5. `type_token_ratio`
6. `avg_sentence_length`
7. `keyword_density_per_minute`
8. `domain_entity_count`
9. `question_count`
10. `sentiment_score`
11. `topic_coherence`
12. `topic_diversity`

#### **Title \& Description Features (8):**

13. `title_keyword_count`
14. `description_keyword_count`
15. `title_description_keyword_overlap`
16. `is_title_question`
17. `has_explanation_markers`
18. `emotional_word_ratio_title`
19. `exam_keyword_intensity` (replaces `is_exam_focused`)
20. `curriculum_alignment_score`

#### **Structure Features (6):**

21. `has_strong_intro`
22. `has_recap`
23. `intro_recap_coverage_ratio`
24. `segment_count`
25. `causal_connective_density`
26. `definition_frequency`

#### **Robustness \& Metadata (2):**

27. `transcript_confidence_score`
28. `domain_language_mix_ratio` (% French vs. Arabic)

**Total: 25 (existing) + 28 (new) = 53 features**
**Expected Model Improvement: +5-10% F1-score (conservative estimate)**

***

## 6. Implementation Roadmap

| **Phase** | **Duration** | **Deliverable** | **Features Added** |
| :-- | :-- | :-- | :-- |
| **Phase 1 (MVP)** | 2-3 weeks | Extract Tier 1 features; validate on 100 videos | Transcript + Title/Desc (Tier 1+2 basics) = 20 features |
| **Phase 2 (Improvement)** | 2-3 weeks | Add Tier 2 (structure); fine-tune multilingual NLP | +8 features; total 28 |
| **Phase 3 (Optimization)** | 2-3 weeks | LDA topic modeling, deeper curriculum alignment | +6 advanced features; finalize 34 total |
| **Phase 4 (Production)** | 1-2 weeks | Error analysis, educator feedback, deployment | Refinement only |

**Total Timeline: 8-12 weeks to production-ready system**

***

## 7. Actionable Next Steps for Your Team

1. **This Week:**
    - Audit 50 transcripts manually for quality (ASR accuracy)
    - Build Bac curriculum lexicon (physics, math, languages, sciences—20 key terms per subject)
    - Compute baseline Flesch-Kincaid \& keyword density on sample
2. **Next Week:**
    - Implement Tier 1 feature extraction pipeline
    - Compute correlations with engagement_score; select top 20 features
3. **Week 3-4:**
    - Train XGBoost with new features; measure F1-score improvement
    - Collect educator feedback: "Do these features align with instructional quality?"
4. **Week 5+:**
    - Iterate on multilingual handling; address transcript confidence issues
    - Deploy to production with confidence scores and explainability

***

## 8. Expected Output \& Educator Recommendations

Once trained, your model will generate **interpretable pre-publication feedback** like:

```
Video: "Photosynthèse: Les réactions lumineuses" (28 min)

📊 ENGAGEMENT POTENTIAL: 74% (High)

Dimensional Breakdown:
  ✅ Content Clarity: 78% (Flesch-Kincaid: Grade 8.2 ✓ Optimal)
  ✅ Concept Density: 72% (11.3 terms/min, well-balanced)
  ⚠️  Pacing: 68% (Speech rate 165 WPM, slightly fast; consider more pauses)
  ✓ Structure: 81% (Strong intro, clear recap, 12 segments)

Top Recommendations:
  1. Add 2-3 processing pauses after concept clusters (would +5% engagement)
  2. Strengthen curriculum alignment: add "photosystem I" and "photosystem II" 
     (currently missing from top concepts; would +3%)
  3. Consider title revision: "How Light Energy Powers Photosynthesis?" 
     (question framing would +2%)

Confidence: HIGH (Transcript confidence: 92%, all features extracted successfully)
```

This actionable, interpretable output is exactly what pre-publication educators need.

***

## References

Educational video research: Speech rate, pacing, and comprehension[^4][^1]

Readability assessment via Flesch-Kincaid in educational contexts[^5][^6][^2]

Discourse analysis: Questions in pedagogical videos correlate with engagement[^7][^8][^3]

Multilingual NLP: XLM-RoBERTa, DistilBERT-multilingual for Arabic-French mixing[^9][^10][^11]

TF-IDF keyword extraction for domain-specific content[^12][^13][^14]

LDA topic modeling and coherence in educational transcripts[^15][^16][^17][^18]

LIWC linguistic features for engagement prediction[^19][^20]

Sentiment analysis via transformers in educational contexts[^21][^22][^23]

***

**Bottom Line:** Your transcripts are a goldmine. Invest 8-12 weeks in extracting **28 content-based features**, validate them incrementally, and you'll likely achieve **5-10% improvement in engagement prediction accuracy** while providing educators **interpretable, actionable pre-publication feedback**. Start with Tier 1 (2-3 weeks); the rest follows naturally.
<span style="display:none">[^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52]</span>

<div align="center">⁂</div>

[^1]: https://www.edglossary.org/student-engagement/

[^2]: https://www.coursebox.ai/blog/mayers-cognitive-theory-of-multimedia-learning

[^3]: https://accedacris.ulpgc.es/bitstream/10553/55361/2/Measuring_quality_instructional.pdf

[^4]: https://e-journal.usd.ac.id/index.php/LLT/article/download/6414/3592

[^5]: https://scholar.smu.edu/cgi/viewcontent.cgi?article=1202\&context=datasciencereview

[^6]: https://aclanthology.org/W12-2019.pdf

[^7]: https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_38.pdf

[^8]: https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Just_Ask_Learning_To_Answer_Questions_From_Millions_of_Narrated_ICCV_2021_paper.pdf

[^9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12240937/

[^10]: https://arxiv.org/pdf/2309.14084.pdf

[^11]: https://www.sciencedirect.com/topics/psychology/learner-engagement

[^12]: https://www.tidytextmining.com/tfidf

[^13]: https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/

[^14]: https://cornerstone.lib.mnsu.edu/cgi/viewcontent.cgi?article=1140\&context=all

[^15]: https://easychair.org/publications/preprint/ld4g/open

[^16]: https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1244\&context=scschcomdis

[^17]: https://lamethods.org/book2/chapters/ch09-nlp/ch09-nlp.html

[^18]: https://www.temjournal.com/content/141/TEMJournalFebruary2025_759_767.pdf

[^19]: http://www.jatit.org/volumes/Vol103No8/33Vol103No8.pdf

[^20]: https://files.eric.ed.gov/fulltext/EJ1121524.pdf

[^21]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308096

[^22]: https://revistas.unir.net/index.php/ijimai/article/download/261/107

[^23]: https://www.jsu.edu/online/faculty/MULTIMEDIA LEARNING by Richard E. Mayer.pdf

[^24]: https://easychair.org/publications/preprint/Q1GB/open

[^25]: https://ijirt.org/publishedpaper/IJIRT159758_PAPER.pdf

[^26]: https://www.diva-portal.org/smash/get/diva2:1591859/FULLTEXT01.pdf

[^27]: https://arxiv.org/pdf/2307.03200.pdf

[^28]: https://www.sciencedirect.com/science/article/abs/pii/S2542660524000477

[^29]: https://aclanthology.org/2023.emnlp-industry.51.pdf

[^30]: https://www.youtube.com/watch?v=g-sndkf7mCs

[^31]: https://www.sciencedirect.com/org/science/article/pii/S1548109324000196

[^32]: https://dl.acm.org/doi/full/10.1145/3571510

[^33]: http://www.revistaiberica.org/index.php/iberica/article/download/852/549/

[^34]: https://www.iieta.org/download/file/fid/60454

[^35]: https://aclanthology.org/W15-4407.pdf

[^36]: https://www.youtube.com/watch?v=652MgnBZ1dA

[^37]: https://www.emerald.com/gkmc/article/doi/10.1108/GKMC-01-2025-0024/1300953/Evaluating-the-growth-trends-of-MOOCs-content-on

[^38]: https://www.atlantis-press.com/article/126002856.pdf

[^39]: https://www.sciencedirect.com/science/article/abs/pii/S147403462200060X

[^40]: https://openaccess.uoc.edu/server/api/core/bitstreams/6dc21de5-e86c-42de-b9a4-c67c1008f81d/content

[^41]: https://www.sciencedirect.com/science/article/pii/S1877050925000390/pdf?md5=f156ea1354e55c72470a59ff10c6cbe5\&pid=1-s2.0-S1877050925000390-main.pdf

[^42]: https://www.iieta.org/download/file/fid/129754

[^43]: https://arxiv.org/html/2505.02324v1

[^44]: https://dl.acm.org/doi/abs/10.1007/s10639-024-12726-8

[^45]: https://ejournal.uin-malang.ac.id/index.php/jeasp/article/download/26892/12010

[^46]: https://encord.com/blog/named-entity-recognition/

[^47]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7274338/

[^48]: https://d-nb.info/1300421576/34

[^49]: https://www.carmatec.com/blog/comprehensive-guide-to-named-entity-recognition-ner/

[^50]: https://mbrenndoerfer.com/writing/tf-idf-bag-of-words-text-representation-information-retrieval

[^51]: https://www.arxiv.org/pdf/2509.24120.pdf

[^52]: https://arxiv.org/abs/2504.18142

