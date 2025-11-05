# 🚀 Adapter Quick Reference

## Instalacija (jednom)
```bash
cd /home/David/Documents/GithubRepos/news-trend-analysis
source venv/bin/activate
pip install adapter-transformers
cp -r /putanja/do/sentiment_adapter_best ./models/
```

## Korištenje

### Sentiment analiza SA adapterom
```bash
python src/analyze_sentiment.py --adapter --adapter-path ./models/sentiment_adapter_best
```

### Sentiment analiza BEZ adaptera
```bash
python src/analyze_sentiment.py
```

### Usporedba base vs adapter
```bash
python compare_models.py --adapter-path ./models/sentiment_adapter_best
```

### Merge adapter (za produkciju)
```bash
python merge_adapter.py --adapter-path ./models/sentiment_adapter_best --output ./models/sentiment_roberta_merged
```

### Cijeli pipeline SA adapterom
```python
# U run_pipeline.py:
analyze_sentiment(use_adapter=True, adapter_path='./models/sentiment_adapter_best')
```

## Potrebna Struktura

```
models/
├── sentiment_adapter_best/
│   ├── adapter_config.json
│   ├── pytorch_model.bin
│   └── head_config.json
└── artifacts/
    └── label_mapping.json
```

## Troubleshooting One-Liners

```bash
# Provjeri adapter
ls -la ./models/sentiment_adapter_best/

# Test adapter učitavanje
python -c "from adapters import AutoAdapterModel; print('OK')"

# Provjeri GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Usporedi rezultate
python -c "import pandas as pd; df=pd.read_csv('data/processed/articles_with_sentiment.csv'); print(df['sentiment'].value_counts())"
```

## Performance Check

```python
import pandas as pd
df = pd.read_csv('data/processed/articles_with_sentiment.csv')
print(f"Mean confidence: {df['sentiment_confidence'].mean():.3f}")
print(f"Low conf (<0.7): {(df['sentiment_confidence'] < 0.7).sum()}")
print(df.groupby('sentiment')['sentiment_confidence'].mean())
```
