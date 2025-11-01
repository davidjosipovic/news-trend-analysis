# 📰 News Trend Analysis

**Automated news aggregation and analysis system with sentiment analysis, topic modeling, and interactive visualization dashboard.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-blue)](https://newstrendanalysis.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🎓 **University Projects**: 
> - **MOPJ (NLP)**: Automated NLP pipeline with sentiment analysis, topic modeling, and summarization
> - **PI (Business Intelligence)**: Predictive analytics dashboard with comprehensive evaluation system

## 🎯 Business Problem

**Challenge**: Manual tracking of economic news sentiment is time-consuming and subjective.

**Solution**: Automated AI-powered pipeline that:
- Fetches economic news every 12 hours
- Analyzes sentiment with 76% confidence (FinBERT)
- Discovers trending topics automatically (BERTopic)
- Generates concise summaries (37.7x compression)
- Visualizes insights in interactive dashboard

**Business Value**:
- 📈 **Investors**: Real-time market sentiment tracking
- 📰 **Media**: Identify trending topics and narratives
- 📊 **Analysts**: Automated research assistance

## 🌟 Features

- 🤖 **Automated News Collection**: Fetches latest economic news from NewsData.io API
- 🕷️ **Web Scraping**: Extracts full article content from news websites
- 🧠 **Advanced NLP Analysis**:
  - **Sentiment Analysis**: FinBERT transformer model (Prosus AI)
  - **Topic Modeling**: BERTopic with HDBSCAN clustering
  - **Automatic Summarization**: DistilBART (CNN-trained)
- 📊 **Interactive Dashboard**: Real-time visualization with Streamlit + Plotly
- ⚡ **Automated Pipeline**: GitHub Actions runs twice daily (8:00 & 20:00 UTC)
- 🎯 **Quality Filtering**: Removes paid content and short articles (< 200 words)
- 🔄 **Duplicate Detection**: Smart handling of cross-source articles

## 🚀 Live Demo

**Dashboard**: [https://newstrendanalysis.up.railway.app/](https://newstrendanalysis.up.railway.app/)

## 📸 Screenshots

### Dashboard Overview
- **4 Key Metrics**: Total articles, unique articles, sentiment, topics
- **Interactive Charts**: Sentiment distribution, topic clustering, time series
- **Smart Pagination**: Browse articles with customizable page size
- **Advanced Filtering**: By sentiment, topic, and date

### Features
- ✅ **Sentiment Over Time**: Track market sentiment trends
- ✅ **Topic Distribution**: Visualize news themes
- ✅ **Article Summaries**: AI-generated summaries for quick insights
- ✅ **Duplicate Toggle**: Show/hide articles from multiple sources

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **NLP Models** | Transformers (FinBERT, DistilBART), BERTopic, Sentence-Transformers |
| **Dashboard** | Streamlit, Plotly |
| **Deployment** | Railway (dashboard), GitHub Actions (pipeline) |
| **Data Source** | NewsData.io API |
| **Web Scraping** | Newspaper3k, BeautifulSoup |

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- NewsData.io API key ([Get free key](https://newsdata.io/))

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/davidjosipovic/news-trend-analysis.git
cd news-trend-analysis
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. **Install Dependencies**

For full pipeline (includes NLP models ~2GB):
```bash
pip install -r requirements.full.txt
```

For dashboard only (lightweight ~50MB):
```bash
pip install -r requirements.txt
```

4. **Configure API Key**
```bash
echo "NEWS_API_KEY=your_api_key_here" > .env
```

## 🎯 Usage

### Option 1: Run Full Pipeline

Execute complete analysis pipeline:

```bash
# Step 1: Fetch news articles
python src/fetch_articles.py

# Step 2: Scrape full content
python src/scrape_articles.py

# Step 3: Clean and preprocess
python src/preprocess_articles.py

# Step 4: Sentiment analysis
python src/analyze_sentiment.py

# Step 5: Topic modeling
python src/discover_topics.py

# Step 6: Generate summaries
python src/summarize_articles.py

# Step 7: Evaluate pipeline quality
python src/evaluate_pipeline.py
```

### Option 2: Run Dashboard Only

```bash
streamlit run dashboard/streamlit_app.py
```

Access dashboard at: `http://localhost:8501`

## 📁 Project Structure

```
news-trend-analysis/
├── 📂 src/                          # Core processing pipeline
│   ├── fetch_articles.py            # NewsData.io API integration
│   ├── scrape_articles.py           # Web scraper (newspaper3k)
│   ├── preprocess_articles.py       # Text cleaning & filtering
│   ├── analyze_sentiment.py         # FinBERT sentiment inference
│   ├── discover_topics.py           # BERTopic clustering
│   ├── summarize_articles.py        # DistilBART summarization
│   └── evaluate_pipeline.py         # Quality metrics & reporting
│
├── 📂 dashboard/
│   └── streamlit_app.py             # Interactive Streamlit dashboard
│
├── 📂 data/
│   ├── raw/                         # Raw JSON from API
│   │   ├── news_*.json              # Fetched articles
│   │   └── articles_scraped.json    # Scraped content
│   └── processed/                   # Processed CSV datasets
│       ├── articles.csv
│       ├── articles_with_sentiment.csv
│       ├── articles_with_topics.csv
│       └── articles_with_summary.csv
│
├── 📂 models/
│   └── topic_model/                 # Saved BERTopic models
│
├── 📂 .github/workflows/
│   └── daily-update.yml             # Automated pipeline (2x daily)
│
├── run_pipeline.py                  # Complete pipeline runner
├── requirements.txt                 # Lightweight deps (dashboard)
├── requirements.full.txt            # Full deps (NLP pipeline)
└── README.md
```

## 🎨 Dashboard Features

### 📊 Metrics & Visualizations

| Feature | Description |
|---------|-------------|
| **Total Articles** | Count of all processed articles |
| **Unique Articles** | Articles after duplicate removal |
| **Sentiment Distribution** | Pie chart of positive/neutral/negative |
| **Topics by Article** | Bar chart of topic distribution |
| **Sentiment Over Time** | Line chart tracking sentiment trends |

### 🎛️ Interactive Controls

- **Sentiment Filter**: Show only positive/neutral/negative articles
- **Topic Filter**: Filter by specific topic cluster
- **Sort Order**: Newest first / Oldest first
- **Pagination**: 5-50 articles per page
- **Duplicate Toggle**: Show/hide cross-source duplicates

## ⚙️ Configuration

### Data Quality Filters

Articles are **automatically filtered** based on:

| Filter | Threshold | Reason |
|--------|-----------|--------|
| **Minimum Words** | 200+ words | Ensures substantive content for analysis |
| **Paid Content** | Excluded | Removes "ONLY AVAILABLE IN PAID PLANS" |
| **Duplicate Titles** | Optional | Toggle to show/hide cross-source articles |

### Models

| Task | Model | Source | Why This Model? |
|------|-------|--------|-----------------|
| **Sentiment** | `ProsusAI/finbert` | HuggingFace | Fine-tuned on **financial news** (better for economic articles than Twitter-based models) |
| **Embeddings** | `all-MiniLM-L6-v2` | Sentence-Transformers | Fast, efficient semantic embeddings |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | HuggingFace | Compressed BART trained on **CNN news articles** |
| **Topic Modeling** | BERTopic + HDBSCAN | Custom configuration | Unsupervised clustering with auto-generated labels |

## 🚢 Deployment

### Railway (Dashboard)
```bash
# Automatic deployment on git push
# Uses: requirements.txt (lightweight)
# Environment variables: NEWS_API_KEY
```

### GitHub Actions (Pipeline)
```yaml
# Runs twice daily: 8:00 AM & 8:00 PM UTC
# Uses: requirements.full.txt (full NLP)
# Commits processed data back to repo
```

## 📊 Data Flow

```
NewsData.io API → fetch_articles.py → scrape_articles.py → preprocess_articles.py 
                                                                      ↓
                        evaluate_pipeline.py ← summarize_articles.py ← discover_topics.py ← analyze_sentiment.py
                                  ↓
                            Dashboard (Streamlit)
```

## 🔬 Analysis Details

### Why Pre-trained Models? (No Training Required)

This project uses **transfer learning** - applying pre-trained models rather than training from scratch. This approach is:

1. **Industry Standard**: Pre-trained transformers (FinBERT, BART) are trained on billions of tokens
2. **More Accurate**: FinBERT trained on 4.9M financial sentences vs. our 55 articles
3. **Practical**: Training BERT from scratch requires 4 TPUs for 4 days (~$500-1000)
4. **Academic**: Demonstrates proper use of state-of-the-art NLP (BERT, transformers)

**Models Used:**
- **FinBERT**: Fine-tuned BERT for financial sentiment (Prosus AI)
- **DistilBART**: Distilled BART for news summarization (trained on CNN/DailyMail)
- **BERTopic**: Unsupervised topic discovery (no training needed)

### Sentiment Analysis
- **Model**: FinBERT (BERT fine-tuned on financial news corpus)
- **Output**: Positive, Neutral, Negative (with confidence scores)
- **Inference**: Batch processing on CPU (no GPU required)
- **Advantages**: Understands financial terminology ("rate cut", "inflation", "growth")

### Topic Modeling
- **Algorithm**: HDBSCAN clustering on sentence embeddings
- **Dimensionality Reduction**: UMAP
- **Labels**: Auto-generated using KeyBERT (unsupervised)
- **Dynamic**: Discovers new topics as articles grow (no predefined categories)

### Summarization
- **Model**: DistilBART (compressed BART for efficiency)
- **Length**: 30-130 tokens per summary
- **Quality**: Requires 200+ word articles
- **Batch Processing**: Handles 8 articles simultaneously

## 📊 Results & Evaluation

### Overall Pipeline Quality: **85/100** (GOOD)

#### Sentiment Analysis Results:
```
📉 Negative:  10 articles (18.2%)
⚪ Neutral:   28 articles (50.9%)
📈 Positive:  17 articles (30.9%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Confidence: 76.0%
```

**Key Improvement**: Switched from Twitter-RoBERTa (68% confidence, 1.8% negative detection) to FinBERT (76% confidence, 18.2% negative detection) for better financial news understanding.

#### Topic Discovery Results:
```
Discovered 6 coherent topics:
1. Korea_Trump_China           - 16 articles (29%)
2. Economic_Sector_Sustainable - 9 articles (16%)
3. Inflation_Forecasts         - 8 articles (15%)
4. Tourism_Travel              - 7 articles (13%)
5. Reforms_Economy             - 6 articles (11%)
6. Business_Development        - 4 articles (7%)

Topic Quality Score: 100/100
```

#### Summarization Performance:
```
✓ Coverage: 100% (all articles summarized)
✓ Avg length: 51 words
✓ Compression ratio: 37.7x (from ~800 → 51 words)
✓ Processing time: ~5s per article (CPU)
```

### Evaluation Metrics (6 categories):
1. **Data Quality** (30% weight): 100/100
   - 100% completeness, proper filtering
2. **Sentiment Balance** (15% weight): 0/100
   - Expected imbalance (economic news naturally more negative/neutral)
3. **Topic Quality** (25% weight): 100/100
   - Coherent clusters, balanced distribution
4. **Summarization** (30% weight): 100/100
   - Full coverage, appropriate compression
5. **Temporal Analysis**: 2-day coverage with automated updates
6. **Confidence Tracking**: Real-time monitoring of prediction reliability

### Business Insights:
- **US-China relations** dominate economic news (29%)
- **Sustainability** emerging as major economic theme
- **Inflation concerns** persist across multiple articles
- **Tourism sector** showing recovery signals

## 🤝 Contributing

This is a university project, but suggestions are welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📝 License

MIT License - feel free to use for educational purposes

## 👨‍💻 Author

**Built as a university project** demonstrating:
- Automated NLP pipeline design
- Real-time data visualization
- Cloud deployment (Railway + GitHub Actions)
- Modern Python best practices

## 🙏 Acknowledgments

- **NewsData.io** for free news API access
- **HuggingFace** for pre-trained transformer models
- **Streamlit** for rapid dashboard development
---

⭐ **Star this repo** if you found it helpful for your own projects!
