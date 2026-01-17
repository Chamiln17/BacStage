# In-Depth: TF-IDF for Extracting Features from Text

## 🎯 **What is TF-IDF?**

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects **how important a word is to a document in a collection**. It's perfect for your YouTube video titles and descriptions because it:

1. **Identifies discriminative words** - Words that distinguish high-engagement videos from low-engagement ones
2. **Downweights common words** - "video", "lesson", "2024" appear everywhere and aren't useful
3. **Upweights rare, topic-specific words** - "derivative", "thermodynamics", "عضوية" (organic chemistry)

---

## 📐 **How TF-IDF Works Mathematically**

TF-IDF has two components:

### **1. Term Frequency (TF)** - How often does a word appear in THIS document?

```
TF(word, document) = (Number of times word appears in document) / (Total words in document)
```

**Example:**
```
Title: "Bac 2024 Math Derivatives Exercise Derivatives Solution"
TF("derivatives") = 2 / 8 = 0.25
TF("bac") = 1 / 8 = 0.125
```

### **2. Inverse Document Frequency (IDF)** - How rare is this word across ALL documents?

```
IDF(word) = log(Total number of documents / Number of documents containing word)
```

**Example** (10,000 videos total):
```
"derivatives" appears in 500 videos: IDF = log(10000/500) = 2.996
"video" appears in 9,800 videos: IDF = log(10000/9800) = 0.020  ← Low score (common word)
"thermodynamics" appears in 50 videos: IDF = log(10000/50) = 5.298  ← High score (rare word)
```

### **3. Final TF-IDF Score**

```
TF-IDF(word, document) = TF(word, document) × IDF(word)
```

**Result:** Words that are:
- **Frequent in THIS video** (high TF) AND
- **Rare across ALL videos** (high IDF)

Get the **highest scores** → These are the discriminative features!

---

## 🔍 **Visual Example with Your Data**

Imagine you have 3 video titles:

| Video | Title | Engagement |
|-------|-------|------------|
| 1 | "Bac 2024 Math Derivatives Complete Course" | High |
| 2 | "Bac 2024 Physics Thermodynamics Lecture" | High |
| 3 | "Bac 2024 General Review Session" | Low |

**Step 1: Build Vocabulary**
```
Unique words: ["bac", "2024", "math", "derivatives", "complete", "course", 
               "physics", "thermodynamics", "lecture", "general", "review", "session"]
```

**Step 2: Calculate TF-IDF Matrix**

|       | bac | 2024 | math | derivatives | physics | thermodynamics | general | review |
|-------|-----|------|------|-------------|---------|----------------|---------|--------|
| Vid 1 | 0.1 | 0.1  | **0.8** | **0.8**   | 0.0     | 0.0            | 0.0     | 0.0    |
| Vid 2 | 0.1 | 0.1  | 0.0  | 0.0         | **0.8** | **0.8**        | 0.0     | 0.0    |
| Vid 3 | 0.1 | 0.1  | 0.0  | 0.0         | 0.0     | 0.0            | 0.6     | 0.6    |

**Key Observations:**
- "bac", "2024" get **low scores** (appear in all videos - not discriminative)
- "derivatives", "thermodynamics" get **high scores** (topic-specific)
- Each video becomes a **numerical vector** that can be fed to ML models

---

## 🛠️ **Implementation in Your Notebook**

### **Where to Add It**

Insert this **after Step 6** (after preparing X and y, before scaling):

```python
## Step 6.5: Extract TF-IDF Features from Text

from sklearn.feature_extraction.text import TfidfVectorizer

print("📝 Extracting TF-IDF features from title and description...")

# Combine title + description for richer context
# Title: "Bac Math Derivatives"
# Description: "Complete course covering derivative rules, applications..."
# Combined: More semantic information for the model
train_df['text_combined'] = (
    train_df['title'].fillna('') + ' ' + 
    train_df['description'].fillna('')
)
val_df['text_combined'] = (
    val_df['title'].fillna('') + ' ' + 
    val_df['description'].fillna('')
)
test_df['text_combined'] = (
    test_df['title'].fillna('') + ' ' + 
    test_df['description'].fillna('')
)

# Initialize TF-IDF vectorizer with carefully chosen parameters
tfidf = TfidfVectorizer(
    # === Core Parameters ===
    max_features=50,           # Keep only top 50 most important terms
                               # (prevents feature explosion, reduces overfitting)
    
    # === N-gram Range ===
    ngram_range=(1, 2),        # Extract both single words AND word pairs
                               # Examples:
                               # Unigrams (1): "derivative", "thermodynamics"
                               # Bigrams (2): "bac 2024", "exercice corrigé"
                               # Bigrams capture phrases like "organic chemistry"
    
    # === Frequency Filters ===
    min_df=5,                  # Ignore words appearing in <5 videos
                               # (too rare = likely typos or irrelevant)
    
    max_df=0.8,                # Ignore words appearing in >80% of videos
                               # (too common = not discriminative)
                               # Examples: "video", "lesson", "دروس" (lessons)
    
    # === Text Preprocessing ===
    strip_accents='unicode',   # Remove diacritics from Arabic/French
                               # "مُشْتَقّة" → "مشتقة"
                               # "dérivée" → "derivee"
    
    lowercase=True,            # Convert to lowercase
                               # "DERIVATIVE" → "derivative"
    
    # === Advanced Options ===
    sublinear_tf=True,         # Use log(TF) instead of raw TF
                               # Reduces impact of word repetition
                               # (prevents spam-like titles from dominating)
    
    use_idf=True,              # Apply IDF weighting (default=True)
                               # This is the "IDF" part of TF-IDF
    
    norm='l2'                  # Normalize each document vector to unit length
                               # Makes long and short descriptions comparable
)

# Fit vectorizer on TRAINING data only (prevent data leakage!)
# This learns the vocabulary from train set
print("  Fitting TF-IDF on training data...")
X_train_tfidf = tfidf.fit_transform(train_df['text_combined'])

# Transform val/test using the SAME vocabulary learned from training
# If a word appears in test but not train, it's ignored (out-of-vocabulary)
X_val_tfidf = tfidf.transform(val_df['text_combined'])
X_test_tfidf = tfidf.transform(test_df['text_combined'])

# Convert sparse matrices to dense arrays
X_train_tfidf_dense = X_train_tfidf.toarray()
X_val_tfidf_dense = X_val_tfidf.toarray()
X_test_tfidf_dense = X_test_tfidf.toarray()

# Concatenate TF-IDF features with existing numerical features
# Before: (6018, 35) - only numerical features
# After: (6018, 85) - numerical + 50 text features
X_train_with_text = np.hstack([X_train_scaled, X_train_tfidf_dense])
X_val_with_text = np.hstack([X_val_scaled, X_val_tfidf_dense])
X_test_with_text = np.hstack([X_test_scaled, X_test_tfidf_dense])

print(f"✅ TF-IDF features extracted:")
print(f"   Original numerical features: {X_train_scaled.shape[1]}")
print(f"   TF-IDF text features: {X_train_tfidf.shape[1]}")
print(f"   Total combined features: {X_train_with_text.shape[1]}")

# Inspect the vocabulary learned
feature_names = tfidf.get_feature_names_out()
print(f"\n📋 Sample TF-IDF features (top 20):")
for i, term in enumerate(feature_names[:20], 1):
    print(f"   {i}. {term}")

# Optional: Show IDF scores for most important terms
idf_scores = dict(zip(feature_names, tfidf.idf_))
sorted_terms = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\n🔥 Most discriminative terms (highest IDF):")
for term, score in sorted_terms[:10]:
    print(f"   {term}: {score:.3f}")
```

---

## 🔄 **Update Your Training Loop**

In your training cell (Step 7), replace `X_train_scaled` with `X_train_with_text`:

```python
# OLD:
# model.fit(X_train_scaled, y_train)
# y_val_pred = model.predict(X_val_scaled)

# NEW:
model.fit(X_train_with_text, y_train)
y_val_pred = model.predict(X_val_with_text)

# And in test evaluation:
y_test_pred = best_model.predict(X_test_with_text)
```

---

## 🎛️ **Parameter Deep Dive**

### **1. `max_features=50`**
**What it does:** Keeps only the top 50 most important words (by mean TF-IDF score across corpus)

**Why 50?**
- Too few (10): Misses nuanced subject-specific terms
- Too many (500): Overfitting, noise, slow training
- **Sweet spot (50-100):** Captures key topics without explosion

**How to tune:** Start with 50, check feature importance, increase to 100 if text features rank high

---

### **2. `ngram_range=(1, 2)`**
**What it does:** Extracts both single words and word pairs

**Examples:**
```python
Title: "Bac 2024 Math Derivatives Complete Course"

Unigrams (1-grams):
["bac", "2024", "math", "derivatives", "complete", "course"]

Bigrams (2-grams):
["bac 2024", "2024 math", "math derivatives", "derivatives complete", "complete course"]
```

**Why bigrams matter for your data:**
- "bac 2024" vs "bac 2023" (temporal context)
- "organic chemistry" vs separate "organic" + "chemistry"
- "exercice corrigé" (solved exercise - common phrase)
- "دروس مباشرة" (live lessons - Arabic phrase)

**Trade-off:**
- `(1, 1)`: Only unigrams, simpler, faster
- `(1, 2)`: Adds bigrams, captures phrases, **recommended**
- `(1, 3)`: Adds trigrams, too sparse, usually overkill

---

### **3. `min_df=5` & `max_df=0.8`**
**What they do:** Filter out rare and common words

**Visual explanation:**

```
10,000 videos total

min_df=5:
❌ "thermodynamix" appears in 2 videos → Ignored (likely typo)
❌ "quantique" appears in 4 videos → Ignored (too rare)
✅ "organic" appears in 50 videos → Kept

max_df=0.8:
❌ "video" appears in 9,900 videos (99%) → Ignored (too common)
❌ "lesson" appears in 9,500 videos (95%) → Ignored (not discriminative)
✅ "derivatives" appears in 1,200 videos (12%) → Kept
```

**How to tune:**
- **Small dataset (<1,000 videos):** `min_df=2`, `max_df=0.9`
- **Your dataset (10,000 videos):** `min_df=5`, `max_df=0.8` ✅
- **Large dataset (>100,000):** `min_df=10`, `max_df=0.7`

---

### **4. `strip_accents='unicode'`**
**Critical for multilingual content!**

**What it does:** Removes diacritical marks

**Examples:**
```python
# Arabic
"مُشْتَقّة" → "مشتقه"  # (derivative)
"تَفاضُل" → "تفاضل"   # (calculus)

# French
"dérivée" → "derivee"
"intégrale" → "integrale"
"phénomène" → "phenomene"
```

**Why it matters:**
- YouTube API sometimes returns text with/without diacritics inconsistently
- Users may type titles with or without accents
- Normalization ensures "derivée" and "derivee" are treated as the same word

---

### **5. `sublinear_tf=True`**
**What it does:** Uses `1 + log(TF)` instead of raw `TF`

**Visual difference:**

| Word Count in Title | Raw TF | Sublinear TF (1+log) |
|---------------------|--------|----------------------|
| 1 occurrence | 1.0 | 1.0 |
| 2 occurrences | 2.0 | 1.69 |
| 5 occurrences | 5.0 | 2.61 |
| 10 occurrences | 10.0 | 3.32 |

**Why use it:**
- Prevents spam-like titles from dominating: "BAC BAC BAC BAC MATH MATH MATH"
- **Diminishing returns:** The difference between 1 and 2 mentions is more important than 10 vs 11
- More robust for clickbait titles

---

## 🌍 **Handling Multilingual Content**

Your dataset has **Arabic, French, and English**. Here's how TF-IDF handles it:

### **Option 1: Language-Agnostic (Current Approach)**
```python
tfidf = TfidfVectorizer(
    strip_accents='unicode',  # Normalizes across languages
    lowercase=True,           # Works for Latin scripts (English/French)
    ngram_range=(1, 2)
)
```

**Pros:**
- ✅ Simple, works out of the box
- ✅ Captures code-switching ("bac math dérivée تفاضل")

**Cons:**
- ⚠️ Treats Arabic and transliterated Arabic differently
- ⚠️ No language-specific stopwords

---

### **Option 2: Language-Specific Processing (Advanced)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom stopwords for multilingual content
arabic_stopwords = ['في', 'من', 'على', 'هذا', 'هذه']  # in, from, on, this
french_stopwords = ['le', 'la', 'de', 'et', 'un', 'une']
english_stopwords = ['the', 'and', 'or', 'a', 'an']

custom_stopwords = arabic_stopwords + french_stopwords + english_stopwords

tfidf = TfidfVectorizer(
    max_features=50,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    strip_accents='unicode',
    lowercase=True,
    stop_words=custom_stopwords  # Remove common words
)
```

**For baseline, stick with Option 1** - it's simpler and often works just as well!

---

## 📊 **Expected Output & Interpretation**

After running the TF-IDF cell, you'll see output like:

```
✅ TF-IDF features extracted:
   Original numerical features: 35
   TF-IDF text features: 50
   Total combined features: 85

📋 Sample TF-IDF features (top 20):
   1. bac
   2. math
   3. derivatives
   4. physics
   5. thermodynamics
   6. bac 2024
   7. exercice corrigé
   8. organic chemistry
   9. تفاضل  (calculus)
   10. ميكانيك  (mechanics)
   ...

🔥 Most discriminative terms (highest IDF):
   quantum mechanics: 6.245  ← Rare, topic-specific
   polymer chemistry: 6.102
   philosophical: 5.987
   bac: 0.823  ← Common, low IDF
   lesson: 0.541
```

**Interpretation:**
- **High IDF (>5.0):** Very discriminative, appears in <1% of videos
- **Medium IDF (2.0-5.0):** Moderately discriminative, appears in 1-20% of videos
- **Low IDF (<1.0):** Common words, appear in >50% of videos

---

## 🎯 **Why This Works for Your Use Case**

### **Your Current Features (35):**
- ✅ Temporal: upload time, video age
- ✅ Metadata: duration, tags count
- ✅ Channel: video count, avg views
- ❌ **Content semantics: MISSING!**

### **What TF-IDF Adds:**
- ✅ **Topic discrimination:** Math vs Physics vs Philosophy
- ✅ **Exam focus detection:** "bac 2024", "révision", "correction"
- ✅ **Content depth indicators:** "complete course" vs "quick review"
- ✅ **Language patterns:** Arabic-heavy vs French-heavy videos

### **Example Prediction Improvement:**

**Without TF-IDF:**
```
Video A: "Derivatives" uploaded at 3PM, 600 seconds, 50 tags
Video B: "Quantum" uploaded at 3PM, 600 seconds, 50 tags

Model sees: Identical features → Similar prediction
```

**With TF-IDF:**
```
Video A: "Derivatives" → TF-IDF high on "math", "calculus"
Video B: "Quantum" → TF-IDF high on "physics", "quantum"

Model sees: Different semantic content → Different predictions
```

---

## 🚀 **Quick Start Checklist**

1. ✅ Copy the "Step 6.5" code block above
2. ✅ Insert it after Step 6 in your notebook
3. ✅ Replace `X_train_scaled` with `X_train_with_text` in training loop
4. ✅ Run and observe the sample features printed
5. ✅ Check feature importance plot - do text features appear in top 20?
6. ✅ Compare F1-score: expect +3-7% improvement

---

## 📈 **Next Steps After TF-IDF**

If TF-IDF gives good results (+5% F1), consider:

1. **Increase to 100 features:** `max_features=100`
2. **Title-only TF-IDF:** Some videos have empty descriptions
3. **Subject-specific TF-IDF:** Separate vectorizers for Math vs Physics
4. **TF-IDF + custom keywords:** Combine with domain expert terms

If TF-IDF doesn't help (<1% F1 gain), it means:
- Text content is too noisy
- Titles/descriptions are too generic
- Metadata features are already sufficient

---

## 🔍 **Analyzing TF-IDF Feature Importance**

After training, add this cell to see which text features matter most:

```python
## Analyze TF-IDF Feature Contribution

if hasattr(best_model, 'feature_importances_'):
    # Get all feature names (numerical + TF-IDF)
    numerical_feature_names = encoded_feature_cols
    tfidf_feature_names = [f"tfidf_{term}" for term in tfidf.get_feature_names_out()]
    all_feature_names = numerical_feature_names + tfidf_feature_names
    
    # Get importances
    importances = best_model.feature_importances_
    
    # Create dataframe
    feature_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances,
        'type': ['numerical']*len(numerical_feature_names) + ['tfidf']*len(tfidf_feature_names)
    }).sort_values('importance', ascending=False)
    
    # Show top TF-IDF features
    top_tfidf = feature_df[feature_df['type'] == 'tfidf'].head(15)
    print("📊 Top 15 TF-IDF Features by Importance:")
    print(top_tfidf[['feature', 'importance']].to_string(index=False))
    
    # Compare contribution: numerical vs text
    numerical_contrib = feature_df[feature_df['type'] == 'numerical']['importance'].sum()
    tfidf_contrib = feature_df[feature_df['type'] == 'tfidf']['importance'].sum()
    
    print(f"\n📈 Feature Type Contribution:")
    print(f"   Numerical features: {numerical_contrib:.3f} ({numerical_contrib/(numerical_contrib+tfidf_contrib)*100:.1f}%)")
    print(f"   TF-IDF features: {tfidf_contrib:.3f} ({tfidf_contrib/(numerical_contrib+tfidf_contrib)*100:.1f}%)")
```

---

## 💾 **Save TF-IDF Vectorizer**

Don't forget to save the fitted TF-IDF vectorizer for production use:

```python
import joblib

# Save TF-IDF vectorizer
tfidf_path = models_dir / "tfidf_vectorizer.pkl"
joblib.dump(tfidf, tfidf_path)
print(f"💾 Saved TF-IDF vectorizer: {tfidf_path}")
```

When making predictions on new videos:

```python
# Load saved artifacts
scaler = joblib.load("models/scaler_baseline.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/lightgbm_baseline.pkl")

# Prepare new video
new_video = pd.DataFrame({
    'title': ['Bac 2024 Math Derivatives Complete Guide'],
    'description': ['Full course covering all derivative rules...'],
    # ... other features
})

# Extract features
text_combined = new_video['title'] + ' ' + new_video['description']
X_numerical = new_video[numerical_features].values
X_tfidf = tfidf.transform(text_combined).toarray()

# Scale and concatenate
X_numerical_scaled = scaler.transform(X_numerical)
X_final = np.hstack([X_numerical_scaled, X_tfidf])

# Predict
prediction = model.predict(X_final)
```

---

## 📚 **Additional Resources**

- [Scikit-learn TfidfVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Understanding TF-IDF in NLP](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Text Feature Extraction Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
