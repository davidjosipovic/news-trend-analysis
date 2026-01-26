#!/usr/bin/env python
"""Test sentiment classifier predictions"""

import pandas as pd
import joblib
import sys

# Load the classifier
print("Loading sentiment classifier...")
classifier_path = "models/predictive/sentiment_classifier.joblib"
classifier_data = joblib.load(classifier_path)

print("\nClassifier details:")
print(f"  Model type: {type(classifier_data['model'])}")
print(f"  Classes: {classifier_data['label_encoder'].classes_}")
print(f"  Feature names: {len(classifier_data['feature_names'])} features")
print(f"  Class distribution during training: {classifier_data.get('class_distribution', 'N/A')}")

# Load articles and prepare features
print("\nLoading articles...")
articles_df = pd.read_csv("data/processed/articles_with_sentiment.csv")
print(f"  Total articles: {len(articles_df)}")

# Import the classifier class
from models.predictive.sentiment_classifier import SentimentClassifier

print("\nPreparing daily features...")
classifier = SentimentClassifier.load(classifier_path)
daily_df = classifier.prepare_features_from_articles(articles_df)
print(f"  Daily data: {len(daily_df)} days")

# Show last few days of actual data
print("\nLast 5 days actual sentiment:")
for _, row in daily_df.tail(5).iterrows():
    date = row['date'].strftime('%Y-%m-%d')
    dominant = row['dominant_sentiment']
    pos_pct = row['positive_pct']
    neu_pct = row['neutral_pct']
    neg_pct = row['negative_pct']
    score = row['sentiment_score']
    print(f"  {date}: {dominant:8s} (score={score:+.3f}) P={pos_pct:.0%} N={neu_pct:.0%} Neg={neg_pct:.0%}")

# Make predictions
print("\nPredicting next 7 days...")
predictions = classifier.predict_next_days(daily_df, days=7)

print("\nPredictions:")
for pred in predictions:
    date_str = pred['date'].strftime('%Y-%m-%d (%A)')
    sentiment = pred['predicted_sentiment']
    conf = pred['confidence']
    probs = pred['probabilities']
    
    emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[sentiment]
    
    print(f"\n{date_str}:")
    print(f"  {emoji} {sentiment.upper()} ({conf:.0%} confidence)")
    print(f"  Probs: P={probs.get('positive',0):.0%} | N={probs.get('neutral',0):.0%} | Neg={probs.get('negative',0):.0%}")

print("\n" + "="*60)
print("Analysis:")
print("="*60)

# Count predicted classes
from collections import Counter
pred_classes = [p['predicted_sentiment'] for p in predictions]
pred_counts = Counter(pred_classes)
print(f"Predicted class distribution: {dict(pred_counts)}")

# Check if model is just predicting majority class
majority_class = max(pred_counts, key=pred_counts.get)
if pred_counts[majority_class] == len(predictions):
    print(f"\n⚠️  WARNING: Model predicts only '{majority_class}' - possible issues:")
    print("  1. Model may be overfitting to majority class")
    print("  2. Features may not have enough variation")
    print("  3. Class imbalance during training")
    print(f"  4. Training class distribution was: {classifier.class_distribution}")
