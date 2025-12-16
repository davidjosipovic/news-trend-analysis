#!/usr/bin/env python
"""
Train Predictive Models
=======================

Trains weekly forecaster and spike detector models using historical data.

Usage:
    python train_models.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from features.time_features import TimeSeriesFeatureEngineer
from models.predictive.model_trainer import ModelTrainer


def main():
    """Train all predictive models."""
    
    print("=" * 80)
    print("🔮 Training Predictive Models")
    print("=" * 80)
    
    # Check if data exists
    data_path = 'data/processed/articles_with_sentiment.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("\nPlease run the pipeline first:")
        print("  python run_pipeline.py")
        return
    
    # Load data
    print(f"\n[1/4] 📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} articles")
    
    # Check date range
    if 'publishedAt' in df.columns:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        date_range = f"{df['publishedAt'].min().date()} to {df['publishedAt'].max().date()}"
        n_days = (df['publishedAt'].max() - df['publishedAt'].min()).days
        print(f"   Date range: {date_range} ({n_days} days)")
        
        if n_days < 30:
            print(f"\n⚠️  Warning: Only {n_days} days of data available.")
            print("   Predictive models work best with 30+ days of historical data.")
            print("   Consider running the pipeline daily for better results.\n")
    
    # Feature engineering
    print("\n[2/4] 🔧 Engineering time series features...")
    engineer = TimeSeriesFeatureEngineer(random_seed=42)
    
    try:
        features_df = engineer.create_all_features(df)
        print(f"✅ Generated {len(features_df.columns)} features")
        
        # Show feature categories
        feature_names = engineer.get_feature_names(features_df)
        print(f"   - Lag features: {len(feature_names['lag_features'])}")
        print(f"   - Rolling features: {len(feature_names['rolling_features'])}")
        print(f"   - Calendar features: {len(feature_names['calendar_features'])}")
        print(f"   - Trend features: {len(feature_names['trend_features'])}")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return
    
    # Train models
    print("\n[3/4] 🤖 Training predictive models...")
    print("   This may take a few minutes...\n")
    
    trainer = ModelTrainer(
        n_splits=5,           # 5-fold time series cross-validation
        test_size=0.2,        # 20% holdout test set
        random_seed=42,
        use_optuna=False      # Set to True for hyperparameter tuning (slower)
    )
    
    try:
        results = trainer.train_all_models(features_df)
        
        print("\n" + "=" * 80)
        print("📊 Training Results")
        print("=" * 80)
        
        # Sentiment forecaster
        if 'sentiment_forecaster' in results:
            print("\n🎯 Sentiment Forecaster:")
            sent_metrics = results['sentiment_forecaster']['test_metrics']
            for model_name, metrics in sent_metrics.items():
                print(f"\n  {model_name.upper()}:")
                print(f"    MAE:  {metrics['MAE']:.4f}")
                print(f"    RMSE: {metrics['RMSE']:.4f}")
                print(f"    MAPE: {metrics['MAPE']:.2f}%")
        
        # Volume forecaster
        if 'volume_forecaster' in results:
            print("\n📈 Volume Forecaster:")
            vol_metrics = results['volume_forecaster']['test_metrics']
            for model_name, metrics in vol_metrics.items():
                print(f"\n  {model_name.upper()}:")
                print(f"    MAE:  {metrics['MAE']:.2f} articles")
                print(f"    RMSE: {metrics['RMSE']:.2f} articles")
                print(f"    MAPE: {metrics['MAPE']:.2f}%")
        
        # Spike detector
        if 'spike_detector' in results:
            print("\n⚡ Spike Detector:")
            spike_metrics = results['spike_detector']['test_metrics']
            print(f"    Precision: {spike_metrics['precision']:.3f}")
            print(f"    Recall:    {spike_metrics['recall']:.3f}")
            print(f"    F1 Score:  {spike_metrics['f1_score']:.3f}")
            print(f"    ROC-AUC:   {spike_metrics['roc_auc']:.3f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save models
    print("\n[4/4] 💾 Saving models...")
    
    save_path = 'models/predictive/'
    os.makedirs(save_path, exist_ok=True)
    
    try:
        trainer.save_models(save_path)
        print(f"✅ Models saved to {save_path}")
        
        # List saved files
        saved_files = os.listdir(save_path)
        print(f"\n   Saved files ({len(saved_files)}):")
        for filename in sorted(saved_files):
            if not filename.startswith('__'):
                filepath = os.path.join(save_path, filename)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"   - {filename} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"❌ Failed to save models: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run API server:  uvicorn api.prediction_api:app --reload")
    print("  2. View dashboard:  streamlit run dashboard/streamlit_app.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
