#!/usr/bin/env python
"""Retrain sentiment classifier with improved logic"""

from models.predictive.sentiment_classifier import train_sentiment_classifier

print("="*80)
print("RETRAINING SENTIMENT CLASSIFIER")
print("="*80)
print("\nChanges:")
print("  - Improved dominant_sentiment logic")
print("  - More balanced class detection")
print("  - Uses highest percentage class as primary indicator")
print()

metrics = train_sentiment_classifier(
    articles_path="data/processed/articles_with_sentiment.csv",
    save_path="models/predictive/sentiment_classifier.joblib"
)

print("\n" + "="*80)
print("RETRAINING COMPLETE")
print("="*80)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_weighted']:.2%}")
print(f"\nClass distribution in training data:")
for cls, pct in metrics['class_distribution'].items():
    emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[cls]
    print(f"  {emoji} {cls:8s}: {pct:.1%}")
