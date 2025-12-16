# 🚀 Quick Start Guide - Predictive Analytics

## Brzi testni koraci

### 1️⃣ Instalacija dependencija

```bash
# Instaliraj sve dependencije (uključujući XGBoost, FastAPI, itd.)
pip install -r requirements.full.txt
```

### 2️⃣ Treniraj prediktivne modele

```bash
# Trenira sve modele (Sentiment Forecaster, Volume Forecaster, Spike Detector)
python train_models.py
```

**Izlaz:**
- `models/predictive/sentiment_forecaster_*.pkl`
- `models/predictive/volume_forecaster_*.pkl`
- `models/predictive/spike_detector.pkl`
- `models/predictive/metadata.json`

### 3️⃣ Pokreni API server (opcional)

```bash
# FastAPI server na portu 8000
uvicorn api.prediction_api:app --reload --port 8000
```

**Testiraj endpointe:**
```bash
# Health check
curl http://localhost:8000/api/health

# Tjedno predviđanje
curl http://localhost:8000/api/predictions/weekly

# Spike vjerojatnost
curl http://localhost:8000/api/predictions/spike-probability

# Trend analiza (30 dana)
curl http://localhost:8000/api/analytics/trends?period=30
```

**Ili u browseru:**
- API dokumentacija: http://localhost:8000/docs
- Interaktivni Swagger UI

### 4️⃣ Pokreni Dashboard s prediktivnom analitikom

```bash
# Streamlit dashboard
streamlit run dashboard/streamlit_app.py
```

**Dashboard ima 2 taba:**
1. **📰 News Analysis** - postojeća analiza
2. **🔮 Predictive Analytics** - NOVO!
   - Predviđeni vs. stvarni sentiment
   - Spike vjerojatnost gauge
   - Feature importance chart
   - Real-time upozorenja

---

## Detaljni koraci

### Korak 1: Provjeri postojeće podatke

```bash
# Provjeri ima li podataka
ls -lh data/processed/
```

Trebao bi vidjeti:
- `articles_with_sentiment.csv` (OBAVEZNO za treniranje)
- `articles_with_topics.csv`
- `articles_with_summary.csv`

**Ako nema podataka:**
```bash
python run_pipeline.py
```

### Korak 2: Treniraj modele

```bash
python train_models.py
```

**Što radi:**
1. Učitava `articles_with_sentiment.csv`
2. Generira 50+ time series značajki
3. Trenira 3 modela:
   - Sentiment Forecaster (Elastic Net + XGBoost)
   - Volume Forecaster (Elastic Net + XGBoost)
   - Spike Detector (XGBoost + SMOTE)
4. Walk-forward validation (TimeSeriesSplit)
5. Sprema modele u `models/predictive/`

**Preporučeno:** 30+ dana podataka za najbolje rezultate.

### Korak 3: Testiraj programe

**Python kod:**
```python
from models.predictive.weekly_forecaster import SentimentForecaster
from models.predictive.spike_detector import SpikeDetector
import pandas as pd

# Load forecaster
forecaster = SentimentForecaster.load('models/predictive/', 'sentiment')

# Load test data
df = pd.read_csv('data/processed/articles_with_sentiment.csv')
# ... prepare features ...

# Predict
result = forecaster.predict_next_week(latest_features)
print(f"Predicted sentiment: {result['predicted_value']}")
print(f"Confidence: {result['confidence']}")
```

**API pozivi:**
```bash
# JSON response
curl -X GET "http://localhost:8000/api/predictions/weekly" | jq

# Samo sentiment
curl "http://localhost:8000/api/predictions/weekly" | jq '.sentiment_forecast'
```

### Korak 4: Automatizacija (opcional)

**Dodaj u cron za dnevno treniranje:**
```bash
# Svaki dan u 9:00 ujutro
0 9 * * * cd /path/to/news-trend-analysis && /path/to/venv/bin/python train_models.py >> logs/training.log 2>&1
```

---

## Testiranje

```bash
# Pokreni sve testove
pytest tests/ -v

# Samo feature engineering testovi
pytest tests/test_features.py -v

# Samo model testovi
pytest tests/test_models.py -v

# Samo API testovi
pytest tests/test_api.py -v --cov=api
```

---

## Troubleshooting

### ❌ "No module named 'xgboost'"
```bash
pip install xgboost
```

### ❌ "No module named 'fastapi'"
```bash
pip install fastapi uvicorn
```

### ❌ "models not found" u API-ju
Prvo pokreni:
```bash
python train_models.py
```

### ❌ "Not enough data" upozorenje
Prikupi više podataka:
```bash
# Pokreni pipeline nekoliko dana zaredom
python run_pipeline.py
```

### ❌ Dashboard ne pokazuje prediktivni tab
1. Provjeri je li `dashboard/predictive_components.py` kreiran
2. Restartaj Streamlit server
3. Obriši cache: `streamlit cache clear`

---

## Konfiguracija

Svi hiperparametri u `config/config.yaml`:

```yaml
models:
  weekly_forecaster:
    forecast_horizon: 7  # Predikcija 7 dana unaprijed
    
  spike_detector:
    volume_std_threshold: 2.0      # Spike = volume > mean + 2σ
    sentiment_change_threshold: 0.5 # Ili veliki sentiment skok
```

Promijeni i ponovno treniraj:
```bash
python train_models.py
```

---

## Što dalje?

1. **Production deployment:**
   - Deploy API na Railway/Heroku
   -Schedulaj dnevno treniranje
   - Add authentication (API keys)

2. **Poboljšanja:**
   - Optuna tuning (`use_optuna=True`)
   - Više lag/rolling prozora
   - Dodatne značajke (news sources, keywords)

3. **Monitoring:**
   - MLflow za tracking eksperimenata
   - Alerting za velike spike-ove
   - Model drift detection
