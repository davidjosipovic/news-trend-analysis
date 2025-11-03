# Railway Deployment Guide

## What Gets Deployed to Railway

Railway only needs the **dashboard** and **processed data** to run. The NLP processing happens in GitHub Actions.

### Files Deployed ✅
```
dashboard/
  streamlit_app.py          # Main dashboard application
data/processed/
  articles.csv              # Raw articles
  articles_with_sentiment.csv   # With sentiment analysis
  articles_with_topics.csv      # With topic modeling
  articles_with_summary.csv     # With summaries
requirements.txt            # Lightweight dependencies (streamlit, plotly, pandas)
railway.json               # Railway configuration
.railwayignore             # Excludes unnecessary files
```

### Files NOT Deployed (Excluded) ❌
```
src/                       # NLP processing scripts (not needed)
models/                    # Trained models (not needed)
data/raw/                  # Raw JSON files (not needed)
data/evaluation/           # Evaluation reports (not needed)
config/                    # Configuration files (not needed)
requirements.full.txt      # Heavy NLP dependencies (not needed)
run_pipeline.py            # Pipeline script (runs in GitHub Actions)
.venv/                     # Virtual environment
.git/                      # Git repository
.github/                   # GitHub Actions workflows
__pycache__/               # Python cache
```

## Dependencies

### Railway (Lightweight) - ~100 MB
- streamlit
- plotly
- pandas
- python-dotenv

### GitHub Actions (Full) - ~2 GB
- All Railway dependencies plus:
- transformers
- torch
- bertopic
- sentence-transformers
- nltk
- scikit-learn
- beautifulsoup4

## Build Process

1. **GitHub Actions** (Daily at 8 AM UTC):
   - Fetches news articles
   - Runs NLP pipeline (sentiment, topics, summaries)
   - Commits processed CSV files to repository

2. **Railway** (Auto-deploy on commit):
   - Pulls updated CSV files
   - Deploys lightweight dashboard
   - Only installs 4 packages (~100 MB total)

## Size Comparison

| What | Before Optimization | After Optimization |
|------|-------------------|-------------------|
| Dependencies | ~2 GB (all NLP libs) | ~100 MB (dashboard only) |
| Deployment Files | ~1.5 MB | ~500 KB |
| Build Time | 3-5 minutes | 30-60 seconds |

## Troubleshooting

If Railway shows large downloads:
1. Check `.railwayignore` is excluding src/, models/, data/raw/
2. Verify `requirements.txt` only has dashboard dependencies
3. Clear Railway build cache and redeploy
