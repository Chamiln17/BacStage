# Future work: comment sentiment

*By Abdelkebir Achraf.*

!!! note "Status: proposed, not implemented"
    This is a design for classifying the sentiment of comments on Bac lessons. It is not part of the current pipeline. Comments are personal data and YouTube API data, so the [publishing rules](data.md) apply: none would be published, and author names would never be used.

---


## Overview

This document outlines the methodology for performing sentiment analysis on comments from Algerian Baccalaureate (BAC) educational YouTube videos. The goal is to classify comments as **positive**, **negative**, or **neutral** to understand viewer sentiment and engagement.

### Challenges

The comment dataset presents unique challenges:

- **Code-switching**: Mixed languages (Arabic, French, English)
- **Dialectal variations**: Algerian Arabic dialect (Darija) vs. Modern Standard Arabic (MSA)
- **Informal text**: Slang, misspellings, abbreviations, and emojis
- **Transliteration**: Arabizi (Arabic written in Latin script)

---

## Tools and Libraries

### 1. Traditional NLP Libraries

#### NLTK (Natural Language Toolkit)

- Comprehensive toolkit for text preprocessing
- Suitable for basic sentiment analysis tasks
- **Use case**: Text cleaning, tokenization, stopword removal

#### TextBlob

- Simple API for sentiment polarity and subjectivity analysis
- **Limitation**: Primarily designed for English text

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)

- Lexicon-based sentiment analysis
- Effective for social media text
- **Limitation**: English-only, requires adaptation for Arabic

---

### 2. Arabic-Specific NLP Tools

#### AraBERT (Arabic BERT)

- **Repository**: [aub-mind/arabert](https://github.com/aub-mind/arabert)
- Pre-trained transformer models for Arabic NLP
- **Models available**:
  - `AraBERTv0.2-base/large`: Standard Arabic
  - `AraBERTv0.2-Twitter-base/large`: Optimized for dialects and social media
  - Trained on 200M+ Arabic sentences
- **Installation**:
  ```bash
  pip install arabert
  ```
- **Use case**: Fine-tuning for sentiment classification on Arabic text

#### CAMeL Tools

- **Repository**: [CAMeL-Lab/camel_tools](https://github.com/CAMeL-Lab/camel_tools)
- Comprehensive Arabic NLP toolkit from NYU Abu Dhabi
- **Features**:
  - Morphological analysis and disambiguation
  - Named Entity Recognition (NER)
  - Sentiment Analysis
  - Dialect identification (useful for detecting Algerian dialect)
- **Installation**:
  ```bash
  pip install camel-tools
  camel_data -i light  # Install required datasets
  ```

#### AraGPT2

- Arabic generative pre-trained transformer
- Available in base, medium, large, and mega sizes
- **Use case**: Text generation and understanding tasks

#### DzSentiA (Algerian Sentiment Analysis)

- **Reference**: Abdelli et al. (2019) - "Sentiment Analysis of Arabic Algerian Dialect Using a Supervised Method"
- Uses SVM and LSTM with TF-IDF and Word2Vec embeddings
- Specifically designed for **Algerian dialect**
- **Citation**:
  ```
  @INPROCEEDINGS{9068897,
    author={A. {Abdelli} and F. {Guerrouf} and O. {Tibermacine} and B. {Abdelli}},
    booktitle={2019 International Conference on Intelligent Systems and Advanced Computing Sciences (ISACS)},
    title={Sentiment Analysis of Arabic Algerian Dialect Using a Supervised Method},
    year={2019},
    pages={1-6}
  }
  ```

---

### 3. Modern Transformer-Based Models (Hugging Face)

#### Multilingual Models

**XLM-RoBERTa**

- **Model**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Trained on 200M+ tweets in 100+ languages
- Effective for multilingual sentiment analysis
- Handles code-switching well

**mBERT (Multilingual BERT)**

- Google's multilingual model
- Supports 104 languages including Arabic
- Good baseline for mixed-language text

**Tabular AI Multilingual Sentiment**

- **Model**: `tabularisai/multilingual-sentiment-analysis`
- Specialized for sentiment analysis across languages
- Pre-trained on diverse multilingual datasets

#### Usage Example (Hugging Face Transformers)

```python
from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis",
                     model="tabularisai/multilingual-sentiment-analysis")

# Classify comment
result = classifier("هذا الفيديو رائع ومفيد جدا!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9987}]
```

---

### 4. Additional Tools

#### PyArabic

- Arabic text processing utilities
- Character normalization, diacritics handling
- **Installation**: `pip install pyarabic`

#### Farasa

- Arabic segmentation and POS tagging
- Integrated with AraBERT preprocessing

#### YouTube Data API v3

- Official API for collecting video comments
- **Documentation**: [Google Developers](https://developers.google.com/youtube/v3)
- **Alternative**: `google-api-python-client` library

#### Sentiment Analysis Libraries

- **Transformers**: `pip install transformers`
- **Datasets**: `pip install datasets` (Hugging Face datasets)
- **Evaluate**: `pip install evaluate` (metrics library)

---

## Pipeline Workflow

### Step 1: Data Collection

**Input**: Videos dataset from `data/processed/videos_bac_only.csv`

**Columns**:

```
video_id, title, description, publish_date, channel_id, channel_title,
category_id, duration_iso, duration_sec, view_count, like_count,
comment_count, tags, thumbnail_url, snapshot_date, run_id,
is_bac_3as, filter_category, filter_confidence, filter_reason, subject
```

**Output**: Raw comments saved to `data/raw/youtube_comments.csv`

**Columns**:

```
video_id, comment_id, text_display, text_original,
author, published_at, like_count, reply_count
```

**Tools**:

- YouTube Data API v3
- Alternative: `youtube-comment-downloader` library

---

### Step 2: Data Preprocessing

**Tasks**:

1. **Text normalization**:

   - Remove HTML markup
   - Normalize Arabic characters (Alef variations, Ya/Ta Marbuta)
   - Handle diacritics (Tashkeel)
   - Map Hindi numerals to Arabic numerals

2. **Cleaning**:

   - Remove URLs, emails, mentions (@username)
   - Handle emojis (keep or convert to sentiment tokens)
   - Remove excessive repetition
   - Strip Tatweel (ـ)

3. **Tokenization**:

   - Use AraBERT preprocessing or CAMeL Tools
   - Handle code-switching (Arabic-French-English)

4. **Language detection**:
   - Identify comment language(s)
   - Use CAMeL Tools dialect identification for Algerian dialect

**Example with AraBERT**:

```python
from arabert.preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv02-twitter"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=True)

text = "الفيديو روعة 😍 merci bcp pour l'explication"
preprocessed = arabert_prep.preprocess(text)
```

---

### Step 3: Sentiment Classification

**Approach A: Fine-tune Pre-trained Model**

- Select base model (AraBERT, XLM-RoBERTa)
- Create labeled training dataset (manual annotation or distant supervision)
- Fine-tune model for 3-class classification (positive/negative/neutral)
- Use Hugging Face Trainer API

**Approach B: Zero-shot Classification**

- Use pre-trained multilingual sentiment models
- No training required
- Lower accuracy but faster deployment

**Approach C: Ensemble**

- Combine multiple models (AraBERT + XLM-RoBERTa + DzSentiA)
- Voting or weighted averaging for final prediction

---

### Step 4: Evaluation

**Metrics**:

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value per class
- **Recall**: True positive rate per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Class-wise performance visualization

**Validation Strategy**:

- Train/Validation/Test split (70/15/15)
- Stratified sampling to maintain class distribution
- Cross-validation for robust evaluation

---

### Step 5: Output Generation

**Output 1**: Comment-level sentiment

- File: `data/processed/youtube_comments_sentiment.csv`
- Columns:
  ```
  video_id, comment_id, text_display, text_original, published_at,
  sentiment, sentiment_score, model_used
  ```

**Output 2**: Video-level sentiment aggregation

- File: `data/processed/video_sentiment_summary.csv`
- Columns:
  ```
  video_id, total_comments, positive_count, negative_count,
  neutral_count, positive_rate, negative_rate, neutral_rate,
  avg_sentiment_score, sentiment_distribution
  ```

**Calculation**:

```python
positive_rate = (positive_count / total_comments) * 100
```

---

## Implementation Checklist

- [ ] Set up YouTube Data API credentials
- [ ] Collect comments for all videos
- [ ] Explore and analyze comment dataset
- [ ] Implement preprocessing pipeline with AraBERT/CAMeL Tools
- [ ] Select and test multiple sentiment models
- [ ] Fine-tune best-performing model (optional)
- [ ] Evaluate model performance
- [ ] Generate sentiment predictions
- [ ] Aggregate results at video level
- [ ] Visualize sentiment distributions
- [ ] Document findings and model performance

---

## References

1. **AraBERT**: Antoun et al. (2020) - "AraBERT: Transformer-based Model for Arabic Language Understanding"
2. **CAMeL Tools**: Obeid et al. (2020) - "CAMeL Tools: An Open Source Python Toolkit for Arabic NLP"
3. **DzSentiA**: Abdelli et al. (2019) - "Sentiment Analysis of Arabic Algerian Dialect Using a Supervised Method"
4. **XLM-RoBERTa**: Conneau et al. (2020) - "Unsupervised Cross-lingual Representation Learning at Scale"
5. **Hugging Face Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

## Notes

- For Algerian dialect-specific analysis, prioritize **AraBERT-Twitter** models and **DzSentiA** approaches
- Consider creating a small manually-labeled dataset for validation
- Code-switching may require language-specific sub-models or specialized multilingual models
- Monitor for potential bias in pre-trained models
- Document all preprocessing decisions for reproducibility
