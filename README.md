# 📰 News Trend Analysis - Automated NLP Pipeline

Automatska analiza trendova u ekonomskim vijestima pomoću NLP metoda sa 24/7 dashboardom i dnevnim ažuriranjima.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-purple.svg)](https://railway.app/)
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-green.svg)](https://github.com/features/actions)

---

## ✨ Features

- 🔍 **Automatsko prikupljanje** - Dnevni fetch novih članaka sa NewsData.io API
- 🕷️ **Web scraping** - BeautifulSoup4 ekstraktuje full content sa originalnih sajtova
- 😊 **Sentiment analiza** - Cardiff NLP RoBERTa model (positive/neutral/negative)
- 📊 **Topic modeling** - BERTopic sa automatskim imenovanjem tema
- 📝 **Summarization** - DistilBART generiše kratke sažetke
- 📈 **Interaktivni dashboard** - Streamlit sa Plotly vizualizacijama
- ⚡ **GitHub Actions** - Automatski update svaki dan u 8:00 UTC
- 🚀 **Railway deployment** - 24/7 javno dostupan dashboard
- 🔄 **Incremental updates** - Samo novi članci se procesuju (URL deduplication)
- ⚙️ **CPU optimized** - Batch processing, multi-threading (3-5x brži)

---

## 🏗️ Architecture

```
┌─────────────────────┐
│  NewsData.io API    │  ← Fetch 50 članaka dnevno
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Web Scraping       │  ← BeautifulSoup full content
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  NLP Pipeline       │  ← Sentiment + Topics + Summaries
│  (CPU Optimized)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit         │  ← Interactive Dashboard
│  Dashboard         │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  GitHub Actions     │  ← Daily automation (8:00 UTC)
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Railway Deploy     │  ← 24/7 public access
└─────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Lokalno pokretanje

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/news-trend-analysis.git
cd news-trend-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup .env file
cp .env.example .env
# Edit .env and add your NewsData.io API key

# Run pipeline manually
python daily_update.py --max-results 50

# Start dashboard
streamlit run dashboard/streamlit_app.py
```

Dashboard: http://localhost:8501

### Option 2: Automatizacija + Cloud Deploy

Prati **[QUICK_START.md](QUICK_START.md)** za:
1. GitHub Actions setup (5 min)
2. Railway deployment (10 min)
3. 24/7 automatski updates! 🎉

---

## 📊 NLP Pipeline Details

### 1. Data Collection
- **API:** NewsData.io (free tier: 200 requests/day)
- **Query:** "economy" OR custom keywords
- **Deduplication:** URL-based tracking (samo novi članci)
- **Storage:** `data/raw/news_YYYYMMDD_HHMMSS.json`

### 2. Web Scraping
- **Library:** BeautifulSoup4 + lxml
- **Selectors:** Multiple CSS selectors za različite sajtove
- **Success rate:** ~90% (9/10 članaka)
- **Fallback:** API description ako scraping fail-uje
- **Storage:** `data/raw/articles_scraped.json`

### 3. Sentiment Analysis
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Labels:** positive, neutral, negative
- **Optimization:** Batch processing (batch_size=16)
- **Performance:** ~5s za 20 članaka na CPU
- **Output:** `data/processed/articles_with_sentiment.csv`

### 4. Topic Modeling
- **Model:** BERTopic + SentenceTransformers
- **Embeddings:** `all-MiniLM-L6-v2`
- **Min cluster size:** 2 članka
- **Auto-labeling:** KeyBERTInspired (top 3 keywords)
- **Performance:** ~7s za 20 članaka na CPU
- **Output:** `data/processed/articles_with_topics.csv` + `models/topic_model/`

### 5. Summarization
- **Model:** `sshleifer/distilbart-cnn-12-6`
- **Min length:** 30 words (members kraći se preskače)
- **Optimization:** Batch processing (batch_size=8)
- **Performance:** ~30s za 18 članaka na CPU
- **Output:** `data/processed/articles_with_summary.csv`

---

## 📂 Project Structure

```
news-trend-analysis/
├── .github/
│   └── workflows/
│       └── daily-update.yml        # GitHub Actions automation
├── data/
│   ├── raw/
│   │   ├── news_*.json            # Fetched articles
│   │   └── articles_scraped.json  # Full scraped content
│   └── processed/
│       ├── articles.csv
│       ├── articles_with_sentiment.csv
│       ├── articles_with_topics.csv
│       └── articles_with_summary.csv
├── models/
│   └── topic_model/
│       └── bertopic_model/        # Saved BERTopic model
├── dashboard/
│   └── streamlit_app.py           # Interactive dashboard
├── src/
│   ├── fetch_news.py              # API fetching + deduplication
│   ├── scrape_articles.py         # Web scraping
│   ├── preprocess.py              # Text cleaning
│   ├── train_sentiment_model.py   # Sentiment analysis
│   ├── train_topic_model.py       # Topic modeling
│   └── summarize.py               # Text summarization
├── daily_update.py                # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── Procfile                       # Railway deployment config
├── railway.json                   # Railway build settings
├── runtime.txt                    # Python version
├── .env.example                   # Environment variables template
├── QUICK_START.md                 # Setup guide
├── AUTOMATION.md                  # Automation options
└── README.md                      # This file
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
NEWS_API_KEY=your_newsdata_io_api_key
```

Get your free API key: https://newsdata.io/register

### GitHub Secrets (for Actions)

```
NEWS_API_KEY=your_key_here
```

Add in: Repository Settings → Secrets → Actions

### Railway Variables

```
NEWS_API_KEY=your_key_here
PORT=8501
```

Add in: Railway Dashboard → Variables

---

## 🔄 Automation Options

### 1. GitHub Actions (Cloud - Recommended)
- ✅ Runs on GitHub servers
- ✅ 8:00 UTC daily
- ✅ Free for public repos
- See: `.github/workflows/daily-update.yml`

### 2. Cron Job (Local)
```bash
./setup_cron.sh
```

### 3. Systemd Timer (Local - Advanced)
```bash
./setup_systemd.sh
```

Details: **[AUTOMATION.md](AUTOMATION.md)**

---

## 📈 Performance

**Hardware:** AMD Ryzen (10 CPU threads)

| Stage         | Articles | Time  | Model                          |
|---------------|----------|-------|--------------------------------|
| Fetch         | 50       | ~2s   | NewsData.io API                |
| Scrape        | 50       | ~30s  | BeautifulSoup4                 |
| Sentiment     | 20       | ~5s   | RoBERTa (batch=16)             |
| Topics        | 20       | ~7s   | BERTopic + SentenceTransformer |
| Summarization | 18       | ~30s  | DistilBART (batch=8)           |
| **Total**     | **50**   | **~74s** | **Full pipeline**           |

**Optimizations:**
- ✅ Batch processing
- ✅ Multi-threading (10 threads)
- ✅ MKL/OpenMP enabled
- ✅ 3-5x faster than sequential

---

## 📊 Dashboard Features

- 📈 **Total articles** with filters (sentiment, topic)
- 😊 **Sentiment distribution** (bar chart)
- 📊 **Topic distribution** with automatic labels
- 📝 **Article cards** with full text & summaries
- 🔍 **Search & filter** functionality
- 📅 **Publication dates** timeline
- 🌐 **Source tracking** (news outlets)

---

## 🐛 Troubleshooting

### Issue: "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### Issue: GitHub Actions fails
1. Check `NEWS_API_KEY` secret exists
2. Verify workflow permissions (Read/Write)
3. Check Actions logs for specific error

### Issue: Railway deployment fails
1. Verify `requirements.txt` has all dependencies
2. Check Railway logs
3. Ensure `Procfile` exists

### Issue: Dashboard shows no data
```bash
# Run pipeline manually to generate data
python daily_update.py --max-results 10

# Push data to GitHub
git add data/ models/
git commit -m "Add initial data"
git push
```

---

## 🎯 Roadmap

- [ ] Email alerts for specific keywords/topics
- [ ] PDF export of daily reports
- [ ] Multi-language support (trenutno: English)
- [ ] Integration sa više news APIs
- [ ] Custom domain na Railway
- [ ] Docker containerization
- [ ] Historical trend analysis (time series)
- [ ] Entity extraction (companies, people, locations)

---

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Step-by-step setup guide
- **[AUTOMATION.md](AUTOMATION.md)** - All automation options
- **[RAILWAY_SETUP.md](RAILWAY_SETUP.md)** - Railway deployment details
- **[.github/GITHUB_ACTIONS_SETUP.md](.github/GITHUB_ACTIONS_SETUP.md)** - GitHub Actions guide

---

## 🤝 Contributing

Pull requests su dobrodošli! Za veće izmene, prvo otvori issue.

1. Fork projekat
2. Kreiraj feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit promene (`git commit -m 'Add AmazingFeature'`)
4. Push na branch (`git push origin feature/AmazingFeature`)
5. Otvori Pull Request

---

## 📄 License

MIT License - slobodno koristi za svoje projekte!

---

## 🙏 Acknowledgments

- **NewsData.io** - News API
- **Hugging Face** - NLP models
- **BERTopic** - Topic modeling
- **Streamlit** - Dashboard framework
- **Railway** - Deployment platform

---

## 📧 Contact

**David** - GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

**Project Link:** https://github.com/YOUR_USERNAME/news-trend-analysis

**Live Dashboard:** https://news-trend-analysis-production.up.railway.app

---

⭐ **Star this repo ako ti se sviđa!** ⭐
