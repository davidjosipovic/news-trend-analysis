# 🔄 Incremental Processing Documentation

## Overview
The pipeline now supports **incremental processing** to preserve historical data while adding new articles efficiently.

## ✅ What Happens at 8 O'Clock (Scheduled Run)

### Before (❌ OLD BEHAVIOR - Data Loss!)
```
8:00 AM: Fetch 50 new articles → news_20251103_080000.json
8:01 AM: Scrape 50 articles → OVERWRITES articles_scraped.json
8:05 AM: Preprocess 50 articles → OVERWRITES articles.csv (113 → 50 articles) ❌
8:10 AM: Sentiment on 50 → OVERWRITES articles_with_sentiment.csv ❌
8:15 AM: Topics on 50 → OVERWRITES articles_with_topics.csv ❌
8:20 AM: Summarize 50 → OVERWRITES articles_with_summary.csv ❌
Result: YOU LOST ALL 113 EXISTING ARTICLES! ❌
```

### After (✅ NEW BEHAVIOR - Incremental & Safe!)
```
8:00 AM: Fetch 50 new articles → news_20251103_080000.json
8:01 AM: Scrape ONLY new 50 → MERGES into articles_scraped.json (113 + 50 = 163 total)
8:05 AM: Preprocess 163 → MERGES with articles.csv, deduplicates → 150 unique ✅
8:10 AM: Sentiment on 50 NEW → MERGES with articles_with_sentiment.csv → 150 total ✅
8:15 AM: Topics on ALL 150 → Retrains model for consistency ✅
8:20 AM: Summarize 50 NEW → MERGES with articles_with_summary.csv → 150 total ✅
Result: 113 existing + 37 new = 150 total articles preserved! ✅
```

## 🛠️ Technical Implementation

### 1. Scraping (Already Incremental ✅)
**File**: `src/scrape_articles.py`

- Reads ALL `news_*.json` files
- Loads existing `articles_scraped.json`
- Only scrapes NEW articles (not in existing file)
- Saves merged result

```python
# Automatically merges historical + new articles
scrape_articles_from_json()  
```

### 2. Preprocessing (NOW Incremental ✅)
**File**: `src/preprocess_articles.py`

**New Parameter**: `incremental=True` (default)

```python
def clean_articles(input_file='data/raw/articles_scraped.json', 
                   output_file='data/processed/articles.csv', 
                   incremental=True):
```

**Behavior**:
- If `incremental=True` and `articles.csv` exists:
  - Loads existing articles
  - Merges with new articles
  - Deduplicates using `title + publishedAt`
- If `incremental=False`:
  - Overwrites completely (old behavior)

**Usage**:
```python
# Incremental (recommended for scheduled runs)
clean_articles()  # incremental=True by default

# Full reprocessing (for testing/debugging)
clean_articles(incremental=False)
```

### 3. Sentiment Analysis (NOW Incremental ✅)
**File**: `src/analyze_sentiment.py`

**New Parameter**: `incremental=True` (default)

```python
def analyze_sentiment(incremental=True):
```

**Behavior**:
- If `incremental=True`:
  - Loads existing `articles_with_sentiment.csv`
  - Identifies NEW articles (not in existing file)
  - Only processes NEW articles through RoBERTa model
  - Merges results with existing sentiment data
- If `incremental=False`:
  - Processes all articles from scratch

**Performance Benefit**:
- Old: Process 150 articles = ~2 minutes
- New: Process only 37 new = ~30 seconds ⚡

### 4. Topic Modeling (Retrain Mode)
**File**: `src/discover_topics.py`

**New Parameter**: `retrain=True` (default)

```python
def discover_topics(input_file='data/processed/articles_with_sentiment.csv', 
                    output_dir='models/topic_model', 
                    retrain=True):
```

**Behavior**:
- `retrain=True`: Retrains BERTopic on ALL articles (recommended)
  - Maintains topic consistency
  - Better topic quality with more data
  - Takes ~30 seconds on 150 articles
- `retrain=False`: Uses existing model (faster but less accurate)

**Why Retrain?**
Topic modeling is unsupervised clustering. Adding new articles can shift cluster boundaries, so retraining ensures all articles have consistent, up-to-date topic assignments.

### 5. Summarization (Already Incremental ✅)
**File**: `src/summarize_articles.py`

- Already implements incremental logic
- Only summarizes NEW articles
- Merges with existing summaries
- Uses `title + publishedAt` as unique ID

## 📊 Performance Comparison

| Stage | Old (Full Reprocess) | New (Incremental) | Speedup |
|-------|---------------------|-------------------|---------|
| Scraping | 163 articles × 2s = **5.4 min** | 50 new × 2s = **1.7 min** | **3.2× faster** |
| Preprocessing | 163 articles | 163 articles (merge) | Same |
| Sentiment | 163 articles = **2.0 min** | 50 new = **0.6 min** | **3.3× faster** |
| Topics | 163 articles = **0.5 min** | 163 articles = **0.5 min** | Same (retrain) |
| Summarization | 163 articles = **4.0 min** | 50 new = **1.2 min** | **3.3× faster** |
| **TOTAL** | **12.4 minutes** | **4.5 minutes** | **2.8× faster** ⚡ |

## 🚀 Usage Examples

### Daily Scheduled Run (8 AM)
```bash
# Run full pipeline with incremental processing
python run_pipeline.py
```

### Force Full Reprocessing (Testing)
```python
from src.preprocess_articles import clean_articles
from src.analyze_sentiment import analyze_sentiment
from src.discover_topics import discover_topics

# Rebuild everything from scratch
clean_articles(incremental=False)
analyze_sentiment(incremental=False)
discover_topics(retrain=True)
```

### Check What's New
```python
import pandas as pd

existing = pd.read_csv('data/processed/articles_with_sentiment.csv')
all_articles = pd.read_csv('data/processed/articles.csv')

print(f"Existing: {len(existing)} articles")
print(f"Total: {len(all_articles)} articles")
print(f"New articles: {len(all_articles) - len(existing)}")
```

## 🔍 Verification

After each run, verify no data loss:

```python
import pandas as pd

# Check article counts
articles = pd.read_csv('data/processed/articles.csv')
sentiment = pd.read_csv('data/processed/articles_with_sentiment.csv')
topics = pd.read_csv('data/processed/articles_with_topics.csv')
summaries = pd.read_csv('data/processed/articles_with_summary.csv')

print(f"Articles: {len(articles)}")
print(f"Sentiment: {len(sentiment)}")
print(f"Topics: {len(topics)}")
print(f"Summaries: {len(summaries)}")

# Verify no duplicates
for df, name in [(articles, 'articles'), (sentiment, 'sentiment'), 
                 (topics, 'topics'), (summaries, 'summaries')]:
    dups = len(df) - len(df.drop_duplicates(subset=['title', 'publishedAt']))
    status = "✅ CLEAN" if dups == 0 else f"❌ {dups} duplicates"
    print(f"{name}: {status}")
```

## 📝 Data Flow Summary

```
Raw Data:
├── news_20251030_*.json (10 articles)
├── news_20251031_*.json (15 articles) 
├── news_20251101_*.json (45 articles)
├── news_20251102_*.json (43 articles)
└── news_20251103_080000.json (50 NEW articles)
                ↓
        Scrape (incremental)
                ↓
    articles_scraped.json (163 total, 50 new scraped)
                ↓
        Preprocess (incremental)
                ↓
    articles.csv (150 unique after dedup)
                ↓
        Sentiment (incremental: only 37 new)
                ↓
    articles_with_sentiment.csv (150 total)
                ↓
        Topics (retrain on all 150)
                ↓
    articles_with_topics.csv (150 total)
                ↓
        Summarize (incremental: only 37 new)
                ↓
    articles_with_summary.csv (150 total)
```

## ⚠️ Important Notes

1. **Deduplication**: Always done on `title + publishedAt` combination
2. **Topic Model**: Retrained on all articles for consistency
3. **Summaries**: Only generated for new articles (expensive operation)
4. **Sentiment**: Only analyzed for new articles (expensive operation)
5. **Data Safety**: Original articles.csv backed up before merge

## 🎯 Best Practices

1. **Daily Runs**: Use default `incremental=True` for all stages
2. **Testing**: Use `incremental=False` to rebuild from scratch
3. **Monitoring**: Check article counts after each run
4. **Backup**: Keep raw JSON files for disaster recovery
5. **Disk Space**: Consider archiving old `news_*.json` files after 30 days

## 🐛 Troubleshooting

**Problem**: Article count decreasing after pipeline run
- **Cause**: Deduplication removed articles with same title+publishedAt
- **Solution**: Check raw JSON files for duplicate fetches

**Problem**: Sentiment/topics missing for some articles
- **Cause**: Incremental processing skipped them
- **Solution**: Run with `incremental=False` to reprocess all

**Problem**: Topic labels changed after new run
- **Cause**: BERTopic retrained with new data
- **Solution**: This is expected! Topics adapt to new data patterns

---

**Last Updated**: November 3, 2025
**Pipeline Version**: 2.0 (Incremental)
