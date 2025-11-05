# 🔥 Korištenje Fine-Tuned Sentiment Adaptera

## Brzi Start

```bash
# 1. Aktiviraj virtual environment
cd /home/David/Documents/GithubRepos/news-trend-analysis
source venv/bin/activate

# 2. Instaliraj adapter-transformers
pip install adapter-transformers

# 3. Kopiraj svoj adapter u projekt
cp -r /putanja/do/sentiment_adapter_best ./models/

# 4. Testiraj adapter
python compare_models.py --adapter-path ./models/sentiment_adapter_best

# 5. Koristi u pipeline
python src/analyze_sentiment.py --adapter --adapter-path ./models/sentiment_adapter_best
```

---

## 📋 Sadržaj

1. [Struktura Adaptera](#struktura-adaptera)
2. [Instalacija](#instalacija)
3. [Korištenje](#korištenje)
4. [Testiranje](#testiranje)
5. [Merging](#merging-adapter-u-base-model)
6. [Troubleshooting](#troubleshooting)

---

## 1. Struktura Adaptera

Tvoj istreniran adapter treba imati ovu strukturu:

```
sentiment_adapter_best/
├── adapter_config.json       # Konfiguracija adaptera
├── pytorch_model.bin          # Težine adaptera (~5-10 MB)
├── head_config.json           # Konfiguracija classification head-a
└── README.md                  # Opciono

artifacts/                     # Opciono ali preporučeno
└── label_mapping.json         # {0: "negative", 1: "neutral", 2: "positive"}
```

### Gdje Staviti Adapter?

```bash
# Struktura projekta
news-trend-analysis/
├── models/
│   ├── sentiment_adapter_best/      # <-- Ovdje
│   ├── artifacts/                   # <-- Ovdje
│   └── sentiment_roberta_merged/    # Nakon merge (opciono)
├── src/
│   └── analyze_sentiment.py
└── ...
```

---

## 2. Instalacija

### Korak 1: Kopiraj Adapter

```bash
# Ako je adapter na drugom mjestu
cp -r /putanja/do/treniranog/sentiment_adapter_best ./models/
cp -r /putanja/do/treniranog/artifacts ./models/

# Provjeri da je sve OK
ls -la ./models/sentiment_adapter_best/
# Očekuješ: adapter_config.json, pytorch_model.bin, head_config.json
```

### Korak 2: Instaliraj Dependencies

```bash
# Aktiviraj venv
source venv/bin/activate

# Instaliraj adapter-transformers
pip install adapter-transformers

# Provjeri instalaciju
python -c "from adapters import AutoAdapterModel; print('✅ OK')"
```

---

## 3. Korištenje

### Način 1: Command Line

```bash
# SA adapterom (brži, bolji za tvoju domenu)
python src/analyze_sentiment.py \
    --adapter \
    --adapter-path ./models/sentiment_adapter_best

# BEZ adaptera (base model)
python src/analyze_sentiment.py
```

### Način 2: Python Skripta

```python
from src.analyze_sentiment import analyze_sentiment

# Sa adapterom
analyze_sentiment(
    incremental=True,
    use_adapter=True,
    adapter_path='./models/sentiment_adapter_best'
)

# Bez adaptera
analyze_sentiment(incremental=True)
```

### Način 3: Integriraj u Pipeline

Ažuriraj `run_pipeline.py`:

```python
# Dodaj na vrhu
ADAPTER_PATH = './models/sentiment_adapter_best'
USE_ADAPTER = os.path.exists(ADAPTER_PATH)

# U Step 4 (sentiment analysis):
print("\n[4/7] 😊 Analyzing sentiment...")
print("-" * 60)
try:
    if USE_ADAPTER:
        print(f"📊 Using fine-tuned adapter from {ADAPTER_PATH}")
        analyze_sentiment(use_adapter=True, adapter_path=ADAPTER_PATH)
    else:
        analyze_sentiment()
    print("✅ Sentiment analysis complete")
except Exception as e:
    print(f"❌ Error during sentiment analysis: {e}")
    return
```

---

## 4. Testiranje

### Test 1: Provjeri da Adapter Radi

```bash
python -c "
from adapters import AutoAdapterModel
model = AutoAdapterModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model.load_adapter('./models/sentiment_adapter_best', load_as='test', set_active=True)
print('✅ Adapter loaded successfully!')
print(f'Active: {model.active_adapters}')
"
```

### Test 2: Usporedi Base vs Adapter

```bash
# Pokreni comparison test
python compare_models.py --adapter-path ./models/sentiment_adapter_best
```

Output:
```
Testing BASE MODEL
══════════════════════════════════════
✅ Base model loaded
  Stock prices surged to record highs...
  → POSITIVE (conf: 0.9123)
  ...

Testing ADAPTER MODEL
══════════════════════════════════════
✅ Adapter loaded
   Active: sentiment_adapter
  Stock prices surged to record highs...
  → POSITIVE (conf: 0.9567)
  ...

COMPARISON
══════════════════════════════════════
Agreement: 9/10 (90.0%)
Avg confidence difference: +0.0421
  → Adapter is MORE confident on average
```

### Test 3: Test na Svojim Podacima

```bash
# Kreiraj CSV s test tekstovima
echo "text
Stock prices jumped 20% after earnings beat
Market crashed badly losing billions
Trading remained flat today" > test_articles.csv

# Testiraj
python compare_models.py --adapter-path ./models/sentiment_adapter_best --test-file test_articles.csv
```

---

## 5. Merging Adapter u Base Model

Merge adapter u base model za **brži inference** (preporučeno za produkciju):

```bash
# Merge adapter
python merge_adapter.py \
    --adapter-path ./models/sentiment_adapter_best \
    --output ./models/sentiment_roberta_merged
```

Onda možeš koristiti merged model:

```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="./models/sentiment_roberta_merged",
    device=0  # GPU
)

result = sentiment_pipeline("Stock prices surged!")
print(result)  # [{'label': 'positive', 'score': 0.95}]
```

### Prednosti Merginga:

- ✅ Brži inference (1 model umjesto base + adapter)
- ✅ Manji memory footprint
- ✅ Jednostavnija distribucija (samo 1 model)
- ✅ Kompatibilno sa standardnim `pipeline()`

### Nedostaci:

- ❌ Veći storage (~500MB vs ~10MB za adapter)
- ❌ Ne možeš lako switchati između base i fine-tuned

---

## 6. Troubleshooting

### Problem: `adapters` modul ne postoji

```bash
# Instaliraj adapter-transformers
pip install adapter-transformers

# NE torch-adapters ili nešto drugo!
```

### Problem: Adapter ne učitava

```bash
# Provjeri strukturu
ls -la ./models/sentiment_adapter_best/

# Trebao bi vidjeti:
# - adapter_config.json
# - pytorch_model.bin
# - head_config.json (možda)
```

### Problem: "No matching distribution found for adapters"

```bash
# Možda imaš staru Python verziju
python --version  # Treba biti 3.8+

# Ili upgrade pip
pip install --upgrade pip
pip install adapter-transformers
```

### Problem: GPU Out of Memory

```python
# U analyze_sentiment.py, smanji batch_size:
batch_size = 8  # ili čak 4
```

ili koristi CPU:

```bash
CUDA_VISIBLE_DEVICES="" python src/analyze_sentiment.py --adapter
```

### Problem: Različiti rezultati base vs adapter

**Ovo je normalno!** Adapter je istreniran na tvojim podacima i može imati:
- Različite odluke za granične slučajeve
- Bolje performanse za tvoju specifičnu domenu
- Drugačiji confidence threshold

Provjeri:
```python
df = pd.read_csv('data/processed/articles_with_sentiment.csv')
print(df.groupby('sentiment').agg({
    'sentiment_confidence': ['mean', 'std', 'min', 'max']
}))
```

### Problem: Label Mapping ne radi

Kreiraj ručno `./models/artifacts/label_mapping.json`:

```json
{
  "id2label": {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
  },
  "classes": ["negative", "neutral", "positive"]
}
```

---

## 🎯 Best Practices

### 1. Kad Koristiti Adapter?

✅ **KORISTI** adapter ako:
- Imaš >1000 kvalitetnih train primjera
- Base model ne performira dovoljno dobro
- Trebaš bolju preciznost za specifične kategorije (npr. "negative")
- Radiaš u specifičnoj domeni (finance, news, itd.)

❌ **NE KORISTI** adapter ako:
- Base model već radi odlično
- Imaš <500 train primjera
- Ne trebaš extra preciznost

### 2. Performance Tips

```python
# 1. Koristi GPU (3-4x brži)
HSA_OVERRIDE_GFX_VERSION=10.3.0 python src/analyze_sentiment.py --adapter

# 2. Povećaj batch_size (ako imaš dovoljno RAM-a)
batch_size = 32  # umjesto 16

# 3. Za produkciju, merge adapter
python merge_adapter.py --adapter-path ./models/sentiment_adapter_best
```

### 3. Monitoring Quality

```python
import pandas as pd

# Učitaj rezultate
df = pd.read_csv('data/processed/articles_with_sentiment.csv')

# Prosječni confidence
print(f"Mean confidence: {df['sentiment_confidence'].mean():.3f}")

# Low confidence articles (možda trebaju review)
low_conf = df[df['sentiment_confidence'] < 0.7]
print(f"Low confidence: {len(low_conf)}/{len(df)} ({len(low_conf)/len(df)*100:.1f}%)")

# Per-sentiment confidence
print(df.groupby('sentiment')['sentiment_confidence'].describe())
```

### 4. A/B Testing

```bash
# Run 1: Base model
python src/analyze_sentiment.py
mv data/processed/articles_with_sentiment.csv results_base.csv

# Run 2: Adapter
python src/analyze_sentiment.py --adapter --adapter-path ./models/sentiment_adapter_best
mv data/processed/articles_with_sentiment.csv results_adapter.csv

# Usporedi
python -c "
import pandas as pd
base = pd.read_csv('results_base.csv')
adapter = pd.read_csv('results_adapter.csv')

print('Base model:')
print(base['sentiment'].value_counts(normalize=True))

print('\nAdapter model:')
print(adapter['sentiment'].value_counts(normalize=True))
"
```

---

## 📚 Dodatni Resursi

- [Adapter-Transformers Docs](https://docs.adapterhub.ml/)
- [AdapterHub](https://adapterhub.ml/) - pretrained adapteri
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751)

---

## 📧 Pitanja?

Ako nešto ne radi:
1. Provjeri [Troubleshooting](#troubleshooting)
2. Pokreni `compare_models.py` za debug
3. Testiraj prvo na malom dataset-u
4. Provjeri da je adapter correctly saved

Uživaj u boljoj sentiment analizi! 🎉
