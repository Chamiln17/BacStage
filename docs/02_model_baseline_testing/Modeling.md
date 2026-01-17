# Modeling Plan: Baseline Testing & Iteration Framework

**Objective:** Build a robust ML pipeline for predicting engagement on Algerian Bac educational videos using existing engineered features, then iterate based on results.

**Strategy:** Follow the unified CLI pipeline → engineer features → test baselines → analyze gaps → iterate.

---

## Overview: Full Modeling Workflow

```
Raw channels.csv
    ↓ collect (existing)
videos_metadata.csv
    ↓ filter_data (existing - balanced Bac filter)
videos_bac_only.csv
    ↓ engineer (existing - temporal, text, engagement, channel features)
videos_engineered.csv
    ↓ [THIS DOCUMENT] Modeling notebook
    → baseline testing
    → feature importance analysis
    → decision on next steps (add features vs. tune hyperparameters)
```

---

## Prerequisite: Ensure Pipeline is Complete

Before starting modeling, make sure data collection and feature engineering are done:

```bash
# Full pipeline (if starting fresh)
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv --filter

# Or, if you've already collected data, just run filter + engineer
uv run python run_pipeline.py filter_data
uv run python run_pipeline.py engineer --input data/processed/videos_bac_only.csv
```

**Verify outputs exist:**
- ✅ `data/processed/videos_bac_only.csv` (filtered Bac videos)
- ✅ `data/processed/videos_engineered.csv` (features ready for modeling)
- ✅ `data/processed/channel_priors.csv` (channel Bac ratios)
- ✅ `data/processed/tfidf_bac_terms.json` (discovered Bac keywords)

---

## Step 1: Setup & Load Engineered Data

### Create Notebook: `notebooks/02_model_baseline_testing.ipynb`

Start with this setup cell:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)

# XGBoost and LightGBM
try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError:
    print("⚠️  Install xgboost and lightgbm for best results")

# Setup
%matplotlib inline
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("✅ Setup complete")
```

### Load Engineered Data

```python
# Load engineered features
engineered_path = DATA_DIR / "processed" / "videos_engineered.csv"
df = pd.read_csv(engineered_path)

print(f"Loaded {len(df):,} videos with {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
```

### Data Quality Check

```python
# Check engagement target variables
print("=" * 60)
print("ENGAGEMENT TARGET VARIABLES")
print("=" * 60)

# engagement_category distribution
print("\nEngagement Category Distribution:")
print(df['engagement_category'].value_counts().sort_index())
print(f"Class balance: {(df['engagement_category'].value_counts() / len(df) * 100).round(2)}")

# engagement_score distribution
print("\nEngagement Score Statistics:")
print(df['engagement_score'].describe())

# Check for nulls in key columns
print("\nNull values in key columns:")
key_cols = ['video_id', 'channel_id', 'engagement_score', 'engagement_category']
print(df[key_cols].isnull().sum())

# Videos per channel (to verify group splitting makes sense)
print("\nVideos per channel:")
print(df['channel_id'].value_counts().describe())
```

---

## Step 2: Prepare Features & Target Variable

### Feature Selection

```python
# Define columns to exclude from modeling
exclude_cols = {
    'video_id', 'title', 'description', 'channel_id', 'channel_title',
    'publish_date', 'snapshot_date', 'run_id',  # Identifiers & metadata
    'view_count', 'like_count', 'comment_count',  # Raw counts (encoded in ratios)
    'engagement_score', 'engagement_category',  # Targets
    'duration_sec',  # Use engineered features instead
}

# Get feature columns
all_cols = set(df.columns)
feature_cols = list(all_cols - exclude_cols)

print(f"Total engineered features: {len(feature_cols)}")
print(f"\nFeatures for modeling:")
for feat in sorted(feature_cols):
    print(f"  - {feat}")
```

### Choose Target Variable

```python
# OPTION 1: Classification (recommended for interpretability)
target_col = 'engagement_category'
task = 'classification'

# OPTION 2: Regression (alternative)
# target_col = 'engagement_score'
# task = 'regression'

print(f"\n{'='*60}")
print(f"MODELING TASK: {task.upper()}")
print(f"TARGET: {target_col}")
print(f"{'='*60}")

X = df[feature_cols].copy()
y = df[target_col].copy()
groups = df['channel_id'].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
```

### Handle Categorical Features

```python
# Identify and encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ {col}: {len(le.classes_)} unique values")

# Save encoders for later use
with open(MODELS_DIR / "label_encoders.pkl", 'wb') as f:
    pickle.dump(label_encoders, f)
```

---

## Step 3: Group-Based Train/Val/Test Splits

**Critical:** Prevent data leakage by grouping videos from the same channel together.

```python
# Split 1: Train+Val (80%) vs Test (20%)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss1.split(X, y, groups))

X_train_val = X.iloc[train_val_idx].copy()
y_train_val = y.iloc[train_val_idx].copy()
groups_train_val = groups.iloc[train_val_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

# Split 2: Train (75% of train_val) vs Val (25% of train_val)
# This gives us: Train 60%, Val 20%, Test 20%
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))

X_train = X_train_val.iloc[train_idx].copy()
y_train = y_train_val.iloc[train_idx].copy()
groups_train = groups_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx].copy()
y_val = y_train_val.iloc[val_idx].copy()
groups_val = groups_train_val.iloc[val_idx]

# Print split summary
print("=" * 60)
print("TRAIN/VAL/TEST SPLIT SUMMARY")
print("=" * 60)
print(f"\nTrain: {len(X_train):,} videos from {groups_train.nunique()} channels ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val):,} videos from {groups_val.nunique()} channels ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test):,} videos from {groups.iloc[test_idx].nunique()} channels ({len(X_test)/len(X)*100:.1f}%)")

if task == 'classification':
    print(f"\nClass distribution:")
    print(f"  Train: {dict(y_train.value_counts())}")
    print(f"  Val:   {dict(y_val.value_counts())}")
    print(f"  Test:  {dict(y_test.value_counts())}")
```

### Standardize Features (Optional but Recommended)

```python
# Scale numerical features for linear models and distance-based models
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# Fit on train, transform all
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print("✅ Features scaled")

# Save scaler
with open(MODELS_DIR / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
```

---

## Step 4: Baseline Model Testing

### Classification Models (if task='classification')

```python
if task == 'classification':
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), 'scaled'),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 'original'),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, random_state=42), 'original'),
        'SVM': (SVC(kernel='rbf', random_state=42, probability=True), 'scaled'),
        'KNN': (KNeighborsClassifier(n_neighbors=5), 'scaled'),
        'Naive Bayes': (GaussianNB(), 'original'),
    }
    
    # Try XGBoost and LightGBM if available
    try:
        models['XGBoost'] = (xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0), 'original')
        models['LightGBM'] = (lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1), 'original')
    except:
        pass

else:  # Regression
    models = {
        'Linear Regression': (Ridge(alpha=1.0), 'scaled'),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 'original'),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), 'original'),
        'SVR': (SVR(kernel='rbf'), 'scaled'),
    }
    
    try:
        models['XGBoost'] = (xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0), 'original')
        models['LightGBM'] = (lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), 'original')
    except:
        pass
```

### Train & Evaluate All Models

```python
results = []
trained_models = {}

for name, (model, data_type) in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    # Choose data type (scaled or original)
    if data_type == 'scaled':
        X_tr, X_v = X_train_scaled, X_val_scaled
    else:
        X_tr, X_v = X_train, X_val
    
    try:
        # Train
        model.fit(X_tr, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_v)
        
        # Evaluate
        if task == 'classification':
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='macro', zero_division=0
            )
            
            # Get probabilities if available
            try:
                y_proba = model.predict_proba(X_v)
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
            except:
                auc = np.nan
            
            results.append({
                'model': name,
                'accuracy': accuracy,
                'precision_macro': precision,
                'recall_macro': recall,
                'f1_macro': f1,
                'auc_macro': auc,
            })
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score (Macro): {f1:.4f}")
            print(f"Precision (Macro): {precision:.4f}")
            print(f"Recall (Macro): {recall:.4f}")
            if not np.isnan(auc):
                print(f"AUC (Macro): {auc:.4f}")
            
        else:  # Regression
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            results.append({
                'model': name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mse': mse,
            })
            
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
        
    except Exception as e:
        print(f"❌ Error training {name}: {e}")
        continue

# Summary DataFrame
results_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*60}")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(MODELS_DIR / "baseline_results.csv", index=False)
print(f"\n✅ Results saved to {MODELS_DIR / 'baseline_results.csv'}")
```

---

## Step 5: Feature Importance Analysis

### Extract Feature Importance (for tree-based models)

```python
importance_data = []

for name, model in trained_models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for feat, imp in zip(feature_cols, importances):
            importance_data.append({
                'model': name,
                'feature': feat,
                'importance': imp
            })

if importance_data:
    importance_df = pd.DataFrame(importance_data)
    
    # Average importance across all tree-based models
    avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print("=" * 60)
    print("TOP 20 FEATURES BY IMPORTANCE (AVERAGE)")
    print("=" * 60)
    print(avg_importance.head(20))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_importance.head(20).plot(kind='barh', ax=ax)
    ax.set_xlabel('Average Importance')
    ax.set_title('Top 20 Features by Average Importance')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png", dpi=100)
    plt.show()
    
    # Save to file
    importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"\n✅ Feature importance saved")
else:
    print("⚠️  No models with feature_importances_ attribute")
```

---

## Step 6: Performance Analysis & Decision Criteria

### Classify Performance Level

```python
# Get best model by F1-score (classification) or R² (regression)
if task == 'classification':
    best_idx = results_df['f1_macro'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_score = results_df.loc[best_idx, 'f1_macro']
else:
    best_idx = results_df['r2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_score = results_df.loc[best_idx, 'r2']

print("=" * 60)
print("PERFORMANCE ASSESSMENT")
print("=" * 60)
print(f"Best model: {best_model_name}")
print(f"Best score: {best_score:.4f}")

# Decision criteria
if task == 'classification':
    if best_score > 0.70:
        recommendation = "EXCELLENT - Proceed with hyperparameter tuning"
        next_steps = [
            "Use GridSearchCV for best model",
            "Try ensemble methods (voting/stacking)",
            "Evaluate on test set",
        ]
    elif best_score > 0.65:
        recommendation = "GOOD - Add 2-3 high-impact features, then tune"
        next_steps = [
            "Add temporal features (days_until_bac, is_exam_season)",
            "Add text features (is_solution_video, is_summary_video)",
            "Re-train and compare",
        ]
    elif best_score > 0.55:
        recommendation = "MODERATE - Comprehensive feature engineering needed"
        next_steps = [
            "Implement all Phase 1 features from Feature engineering ideas.md",
            "Create interaction features",
            "Check for data quality issues",
        ]
    else:
        recommendation = "POOR - Significant model and feature work required"
        next_steps = [
            "Review data quality and target variable",
            "Implement Phase 1 + Phase 2 features (including visuals)",
            "Consider data augmentation or different modeling approach",
        ]
else:  # Regression
    if best_score > 0.40:
        recommendation = "GOOD - Proceed with hyperparameter tuning"
    elif best_score > 0.30:
        recommendation = "MODERATE - Add high-impact features"
    else:
        recommendation = "POOR - Comprehensive feature engineering needed"
    
    next_steps = [
        "Add temporal features (days_until_bac, is_exam_season)",
        "Add domain-specific features from ideas document",
        "Check feature interactions",
    ]

print(f"\n{recommendation}")
print(f"\nNext steps:")
for i, step in enumerate(next_steps, 1):
    print(f"  {i}. {step}")
```

### Test Set Evaluation

```python
# Evaluate best model on test set
if data_type == 'scaled':
    X_test_use = X_test_scaled
else:
    X_test_use = X_test

y_test_pred = trained_models[best_model_name].predict(X_test_use)

if task == 'classification':
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (Classification)")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=100)
    plt.show()

else:  # Regression
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (Regression)")
    print("=" * 60)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "predictions_scatter.png", dpi=100)
    plt.show()
```

---

## Step 7: Save Artifacts & Document Results

```python
# Save best model
best_model = trained_models[best_model_name]
with open(MODELS_DIR / f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl", 'wb') as f:
    pickle.dump(best_model, f)

# Create summary document
summary = f"""
# Baseline Modeling Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Task
Task: {task.upper()}
Target: {target_col}
Samples: {len(df):,} videos from {df['channel_id'].nunique()} channels

## Data Split
- Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)
- Val: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)
- Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)

## Results
Best Model: {best_model_name}
Best Validation Score: {best_score:.4f}

Full Results:
{results_df.to_string()}

## Recommendation
{recommendation}

## Next Steps
{chr(10).join([f'{i}. {step}' for i, step in enumerate(next_steps, 1)])}

## Models Trained
{', '.join(trained_models.keys())}

## Features Used ({len(feature_cols)})
{', '.join(sorted(feature_cols))}
"""

with open(MODELS_DIR / "baseline_summary.txt", 'w') as f:
    f.write(summary)

print("✅ All artifacts saved to models/")
print(f"\nFiles created:")
print(f"  - baseline_results.csv")
print(f"  - feature_importance.csv")
print(f"  - best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
print(f"  - baseline_summary.txt")
print(f"  - feature_importance.png")
print(f"  - confusion_matrix.png or predictions_scatter.png")
```

---

## Step 8: Feature Engineering Iteration (if needed)

### If Performance is Moderate/Poor, Add High-Impact Features

See [Feature engineering ideas.md](Feature%20engineering%20ideas.md) for comprehensive feature list.

**Phase 1 (High-impact, easy to implement):**

```python
# These features are already computed by VideoFeatureEngineer
# If not present, add manually:

# Temporal features
# - days_until_bac: (May 1 - publish_date).days or (June 1 - publish_date).days
# - is_exam_season: publish_date.month in [5, 6]

# Text features
# - is_solution_video: 'حل' in title OR 'corrige' in title
# - is_summary_video: 'ملخص' in title OR 'resumé' in title
```

### Iterate: Re-engineer and Re-test

```bash
# Modify VideoFeatureEngineer in src/features/engineer.py to add new features
# Then re-run the pipeline:

uv run python run_pipeline.py engineer --input data/processed/videos_bac_only.csv

# Re-run this notebook with updated data
```

---

## Implementation Checklist

- [ ] Ensure pipeline is complete (collect → filter → engineer)
- [ ] Verify `videos_engineered.csv` exists
- [ ] Create `notebooks/02_model_baseline_testing.ipynb`
- [ ] Run setup and load data
- [ ] Check data quality and target distribution
- [ ] Prepare features and target variable
- [ ] Create group-based train/val/test splits
- [ ] Train baseline models (6-8 models)
- [ ] Analyze feature importance
- [ ] Evaluate on test set
- [ ] Document results and recommendations
- [ ] Save best model and artifacts
- [ ] [Optional] Iterate with new features if performance is moderate/poor

---

## Expected Outcomes

After completing this baseline testing, you will know:

1. **Best model**: Which algorithm performs best
2. **Baseline performance**: Accuracy/F1/R² to beat in next iterations
3. **Feature quality**: Which features are most important
4. **Feature gaps**: What patterns are missing from current feature set
5. **Next direction**: Whether to add features or proceed to hyperparameter tuning

**Success criteria:**
- Classification: F1-macro > 0.65
- Regression: R² > 0.40

---

## References

- [README.md](README.md) - Project overview
- [TARGETS_AND_FILTERING_GUIDE.md](TARGETS_AND_FILTERING_GUIDE.md) - Target variable definitions
- [Feature engineering ideas.md](Feature%20engineering%20ideas.md) - Extended feature roadmap
- [New_Filtering_Guide.md](New_Filtering_Guide.md) - Bac filtering details
