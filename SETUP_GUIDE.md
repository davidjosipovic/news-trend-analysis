# News Trend Analysis - Setup Guide

## ✅ What Has Been Fixed

All issues have been resolved:

1. ✅ **Python environment configured** - Virtual environment created (.venv)
2. ✅ **All dependencies installed** - All required packages are now available
3. ✅ **Import errors resolved** - All Python import errors fixed
4. ✅ **DVC pipeline updated** - Paths corrected to use `src/` instead of `scripts/`
5. ✅ **Data directories created** - `data/raw/`, `data/processed/`, `models/`, `mlflow_tracking/`
6. ✅ **Code issues fixed**:
   - `preprocess.py` - Now handles dynamic JSON input files
   - `summarize.py` - Fixed to use 'text' column instead of 'content'
   - Added better error handling and progress logging
7. ✅ **Requirements updated** - Added missing packages (sentence-transformers, scikit-learn, umap-learn, hdbscan)

## 🚀 Quick Start

### 1. Test with Sample Data

A sample dataset has been created for you to test the pipeline:

```bash
# Process sample data
python src/preprocess.py

# Run sentiment analysis
python src/train_sentiment_model.py

# Generate summaries
python src/summarize.py

# Launch dashboard
streamlit run dashboard/streamlit_app.py
```

### 2. Use the Complete Pipeline

Run the entire pipeline with one command:

```bash
python run_pipeline.py
```

### 3. Fetch Real News

To fetch real news articles, you need a NewsAPI key:

1. Get free API key from: https://newsapi.org/
2. Set the environment variable:
   ```bash
   export NEWS_API_KEY='your_actual_key_here'
   ```
3. Run the fetch script:
   ```bash
   python src/fetch_news.py
   ```

## 📁 Project Structure

```
news-trend-analysis/
├── src/                          # Source code
│   ├── fetch_news.py            # Fetch articles from NewsAPI
│   ├── preprocess.py            # Clean and prepare data
│   ├── train_sentiment_model.py # Sentiment analysis
│   ├── train_topic_model.py     # Topic modeling with BERTopic
│   ├── summarize.py             # Text summarization
│   └── evaluate.py              # MLflow tracking
├── dashboard/
│   └── streamlit_app.py         # Interactive dashboard
├── data/
│   ├── raw/                     # Raw JSON data
│   │   └── sample_news.json    # Sample data for testing
│   └── processed/               # Processed CSV files
├── models/                      # Saved models
├── mlflow_tracking/            # MLflow experiment tracking
├── requirements.txt            # Python dependencies
├── dvc.yaml                    # DVC pipeline definition
├── Procfile                    # Railway deployment config
├── run_pipeline.py             # Complete pipeline script
└── README.md                   # This file

```

## 🔧 Individual Script Usage

### Fetch News Articles

```bash
# Set your API key
export NEWS_API_KEY='your_key'

# Fetch articles
python src/fetch_news.py
```

### Preprocess Data

```bash
# Process the most recent JSON file
python src/preprocess.py

# Or specify a file
python -c "from src.preprocess import clean_articles; clean_articles('data/raw/sample_news.json')"
```

### Sentiment Analysis

```bash
python src/train_sentiment_model.py
```

### Topic Modeling

```bash
python src/train_topic_model.py
```

### Generate Summaries

```bash
python src/summarize.py
```

### Track with MLflow

```bash
python src/evaluate.py
```

## 📊 Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will be available at: http://localhost:8501

## 🔄 DVC Pipeline

Run the complete pipeline with DVC:

```bash
# Initialize DVC (first time only)
dvc init

# Run the entire pipeline
dvc repro

# View pipeline status
dvc status

# Visualize pipeline
dvc dag
```

## 🚀 Deployment to Railway

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set environment variables in Railway dashboard:
   - `NEWS_API_KEY` (your NewsAPI key)
4. Deploy automatically on push to main branch

## 🛠️ Troubleshooting

### Import Errors
All import errors should be resolved. If you see any:
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Issues
If you run out of memory during model loading:
- Use smaller models
- Process fewer articles at a time
- Increase system swap space

### API Rate Limits
NewsAPI free tier has limits:
- 100 requests per day
- No historical data beyond 1 month

## 📝 Notes

- The virtual environment is located in `.venv/`
- Use Python 3.13.7 (configured)
- All models are downloaded automatically on first run
- Processing time depends on number of articles and system specs

## 🔗 Useful Links

- [NewsAPI Documentation](https://newsapi.org/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DVC Documentation](https://dvc.org/doc)

## 📧 Support

If you encounter any issues:
1. Check this guide
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Try with sample data first

---

**Status:** ✅ All systems operational and ready to use!
