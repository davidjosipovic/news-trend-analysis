"""
Sentiment Class Predictor
=========================

Predicts which sentiment class will dominate for future days:
- positive
- neutral  
- negative

Uses classification (not regression) with:
- XGBoost Classifier
- Features: lag of past dominant classes, day of week, rolling sentiment distributions

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import joblib
import os
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """
    Classifier for predicting dominant sentiment class per day.
    
    Predicts: positive, neutral, or negative
    
    Uses time series features like:
    - Previous days' dominant sentiment (lag features)
    - Rolling sentiment distributions
    - Day of week patterns
    - Trend momentum
    """
    
    def __init__(self, forecast_days: int = 7, random_seed: int = 42):
        """
        Initialize the sentiment classifier.
        
        Args:
            forecast_days: How many days ahead to predict
            random_seed: For reproducibility
        """
        self.forecast_days = forecast_days
        self.random_seed = random_seed
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.class_distribution: Dict[str, float] = {}
        
        # Try XGBoost first, fallback to RandomForest
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_seed,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            logger.info("Using XGBoost Classifier")
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=random_seed
            )
            logger.info("Using RandomForest Classifier (XGBoost not available)")
    
    def prepare_features_from_articles(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily features from article-level data.
        
        Args:
            articles_df: DataFrame with columns: publishedAt, sentiment, is_original
            
        Returns:
            Daily aggregated DataFrame with features
        """
        df = articles_df.copy()
        
        # Filter to original articles only
        if 'is_original' in df.columns:
            original_count = df['is_original'].sum()
            df = df[df['is_original'] == True]
            logger.info(f"Using {len(df)} original articles (excluded {len(articles_df) - len(df)} copies)")
        
        # Parse date
        if 'publishedAt' in df.columns:
            df['date'] = pd.to_datetime(df['publishedAt']).dt.date
        elif 'date' not in df.columns:
            raise ValueError("Need 'publishedAt' or 'date' column")
        
        # Check if sentiment is string or numeric
        is_string_sentiment = df['sentiment'].dtype == 'object'
        
        if is_string_sentiment:
            # String sentiment - count each class
            daily = df.groupby('date').agg(
                article_count=('sentiment', 'count'),
                positive_count=('sentiment', lambda x: (x == 'positive').sum()),
                negative_count=('sentiment', lambda x: (x == 'negative').sum()),
                neutral_count=('sentiment', lambda x: (x == 'neutral').sum())
            ).reset_index()
            
            # Calculate numeric sentiment for trend analysis
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            df['sentiment_numeric'] = df['sentiment'].map(sentiment_map)
            daily_sentiment = df.groupby('date')['sentiment_numeric'].mean().reset_index()
            daily_sentiment.columns = ['date', 'avg_sentiment']
            daily = daily.merge(daily_sentiment, on='date')
            daily['std_sentiment'] = df.groupby('date')['sentiment_numeric'].std().values
        else:
            # Numeric sentiment - threshold to classes
            daily = df.groupby('date').agg(
                article_count=('sentiment', 'count'),
                avg_sentiment=('sentiment', 'mean'),
                std_sentiment=('sentiment', 'std'),
                positive_count=('sentiment', lambda x: (x > 0.1).sum()),
                negative_count=('sentiment', lambda x: (x < -0.1).sum()),
                neutral_count=('sentiment', lambda x: ((x >= -0.1) & (x <= 0.1)).sum())
            ).reset_index()
        
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date').reset_index(drop=True)
        
        # Calculate percentages
        total = daily['positive_count'] + daily['negative_count'] + daily['neutral_count']
        daily['positive_pct'] = daily['positive_count'] / total
        daily['negative_pct'] = daily['negative_count'] / total
        daily['neutral_pct'] = daily['neutral_count'] / total
        
        # Determine dominant sentiment (target variable)
        daily['dominant_sentiment'] = daily.apply(
            lambda row: 'positive' if row['positive_count'] > row['negative_count'] and row['positive_count'] > row['neutral_count']
                       else ('negative' if row['negative_count'] > row['positive_count'] and row['negative_count'] > row['neutral_count']
                             else 'neutral'),
            axis=1
        )
        
        # Store class distribution
        self.class_distribution = daily['dominant_sentiment'].value_counts(normalize=True).to_dict()
        logger.info(f"Class distribution: {self.class_distribution}")
        
        # Add time features
        daily['day_of_week'] = daily['date'].dt.dayofweek
        daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
        daily['day_of_month'] = daily['date'].dt.day
        daily['week_of_year'] = daily['date'].dt.isocalendar().week
        
        # Lag features for sentiment percentages
        for lag in [1, 2, 3, 7]:
            daily[f'positive_pct_lag_{lag}d'] = daily['positive_pct'].shift(lag)
            daily[f'negative_pct_lag_{lag}d'] = daily['negative_pct'].shift(lag)
            daily[f'neutral_pct_lag_{lag}d'] = daily['neutral_pct'].shift(lag)
            daily[f'avg_sentiment_lag_{lag}d'] = daily['avg_sentiment'].shift(lag)
        
        # Rolling features
        for window in [3, 7]:
            daily[f'positive_pct_rolling_{window}d'] = daily['positive_pct'].rolling(window).mean()
            daily[f'negative_pct_rolling_{window}d'] = daily['negative_pct'].rolling(window).mean()
            daily[f'avg_sentiment_rolling_{window}d'] = daily['avg_sentiment'].rolling(window).mean()
            daily[f'sentiment_volatility_{window}d'] = daily['avg_sentiment'].rolling(window).std()
        
        # Momentum features
        daily['sentiment_momentum'] = daily['avg_sentiment'] - daily['avg_sentiment'].shift(1)
        daily['positive_trend'] = daily['positive_pct'] - daily['positive_pct'].shift(3)
        daily['negative_trend'] = daily['negative_pct'] - daily['negative_pct'].shift(3)
        
        # Encode previous dominant sentiment as lag
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        daily['dominant_encoded'] = daily['dominant_sentiment'].map(sentiment_map)
        for lag in [1, 2, 3]:
            daily[f'dominant_lag_{lag}d'] = daily['dominant_encoded'].shift(lag)
        
        # Drop rows with NaN (due to lag/rolling)
        daily = daily.dropna()
        
        return daily
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for training."""
        return [
            'day_of_week', 'is_weekend', 'day_of_month',
            'positive_pct_lag_1d', 'negative_pct_lag_1d', 'neutral_pct_lag_1d',
            'positive_pct_lag_2d', 'negative_pct_lag_2d', 'neutral_pct_lag_2d',
            'positive_pct_lag_3d', 'negative_pct_lag_3d', 'neutral_pct_lag_3d',
            'positive_pct_lag_7d', 'negative_pct_lag_7d', 'neutral_pct_lag_7d',
            'avg_sentiment_lag_1d', 'avg_sentiment_lag_2d', 'avg_sentiment_lag_3d', 'avg_sentiment_lag_7d',
            'positive_pct_rolling_3d', 'negative_pct_rolling_3d', 'avg_sentiment_rolling_3d',
            'positive_pct_rolling_7d', 'negative_pct_rolling_7d', 'avg_sentiment_rolling_7d',
            'sentiment_volatility_3d', 'sentiment_volatility_7d',
            'sentiment_momentum', 'positive_trend', 'negative_trend',
            'dominant_lag_1d', 'dominant_lag_2d', 'dominant_lag_3d'
        ]
    
    def fit(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the classifier on daily data.
        
        Args:
            daily_df: DataFrame from prepare_features_from_articles()
            
        Returns:
            Dictionary with training metrics
        """
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in daily_df.columns]
        
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing features: {missing}")
        
        self.feature_names = available_cols
        
        X = daily_df[available_cols].fillna(0)
        y = daily_df['dominant_sentiment']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        
        logger.info(f"Training on {len(X_train)} days, testing on {len(X_test)} days")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get class names
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'classification_report': report,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': self.class_distribution,
            'classes': list(class_names)
        }
        
        logger.info(f"Accuracy: {accuracy:.2%}, F1: {f1:.2%}")
        logger.info(f"Classes: {class_names}")
        
        return metrics
    
    def predict_next_days(self, daily_df: pd.DataFrame, days: int = 7) -> List[Dict[str, Any]]:
        """
        Predict dominant sentiment for next N days.
        
        Args:
            daily_df: Current data (must have recent days for lag features)
            days: Number of days to predict
            
        Returns:
            List of predictions with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        predictions = []
        current_data = daily_df.copy()
        last_date = current_data['date'].max()
        
        for day_offset in range(1, days + 1):
            pred_date = last_date + timedelta(days=day_offset)
            
            # Create features for this day
            features = self._create_features_for_date(current_data, pred_date)
            
            if features is None:
                continue
            
            # Scale and predict
            X = pd.DataFrame([features])[self.feature_names].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Get class probabilities
            proba = self.model.predict_proba(X_scaled)[0]
            pred_class_idx = np.argmax(proba)
            pred_class = self.label_encoder.classes_[pred_class_idx]
            
            prediction = {
                'date': pred_date,
                'predicted_sentiment': pred_class,
                'confidence': float(proba[pred_class_idx]),
                'probabilities': {
                    cls: float(proba[i]) 
                    for i, cls in enumerate(self.label_encoder.classes_)
                }
            }
            predictions.append(prediction)
            
            # Update current_data with prediction for next iteration
            new_row = {
                'date': pred_date,
                'dominant_sentiment': pred_class,
                'dominant_encoded': {'positive': 1, 'neutral': 0, 'negative': -1}[pred_class],
                'positive_pct': prediction['probabilities'].get('positive', 0.33),
                'negative_pct': prediction['probabilities'].get('negative', 0.33),
                'neutral_pct': prediction['probabilities'].get('neutral', 0.33),
                'avg_sentiment': {'positive': 0.3, 'neutral': 0.0, 'negative': -0.3}[pred_class],
                'day_of_week': pred_date.weekday(),
                'is_weekend': 1 if pred_date.weekday() in [5, 6] else 0,
                'day_of_month': pred_date.day
            }
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
    
    def _create_features_for_date(self, data: pd.DataFrame, target_date) -> Optional[Dict]:
        """Create feature dict for a specific target date."""
        recent = data.tail(7)
        if len(recent) < 3:
            return None
        
        features = {
            'day_of_week': target_date.weekday(),
            'is_weekend': 1 if target_date.weekday() in [5, 6] else 0,
            'day_of_month': target_date.day,
        }
        
        # Lag features
        for i, lag in enumerate([1, 2, 3, 7]):
            if len(recent) > i:
                row = recent.iloc[-(i+1)] if i < 3 else (recent.iloc[-min(7, len(recent))] if len(recent) >= 7 else recent.iloc[0])
                features[f'positive_pct_lag_{lag}d'] = row.get('positive_pct', 0.33)
                features[f'negative_pct_lag_{lag}d'] = row.get('negative_pct', 0.33)
                features[f'neutral_pct_lag_{lag}d'] = row.get('neutral_pct', 0.33)
                features[f'avg_sentiment_lag_{lag}d'] = row.get('avg_sentiment', 0)
        
        # Rolling features
        for window in [3, 7]:
            window_data = recent.tail(window)
            features[f'positive_pct_rolling_{window}d'] = window_data['positive_pct'].mean() if 'positive_pct' in window_data else 0.33
            features[f'negative_pct_rolling_{window}d'] = window_data['negative_pct'].mean() if 'negative_pct' in window_data else 0.33
            features[f'avg_sentiment_rolling_{window}d'] = window_data['avg_sentiment'].mean() if 'avg_sentiment' in window_data else 0
            features[f'sentiment_volatility_{window}d'] = window_data['avg_sentiment'].std() if 'avg_sentiment' in window_data else 0
        
        # Momentum
        if len(recent) >= 2:
            features['sentiment_momentum'] = recent['avg_sentiment'].iloc[-1] - recent['avg_sentiment'].iloc[-2]
        else:
            features['sentiment_momentum'] = 0
            
        if len(recent) >= 4:
            features['positive_trend'] = recent['positive_pct'].iloc[-1] - recent['positive_pct'].iloc[-4]
            features['negative_trend'] = recent['negative_pct'].iloc[-1] - recent['negative_pct'].iloc[-4]
        else:
            features['positive_trend'] = 0
            features['negative_trend'] = 0
        
        # Dominant lag
        for i, lag in enumerate([1, 2, 3]):
            if len(recent) > i and 'dominant_encoded' in recent.columns:
                features[f'dominant_lag_{lag}d'] = recent['dominant_encoded'].iloc[-(i+1)]
            else:
                features[f'dominant_lag_{lag}d'] = 0
        
        return features
    
    def save(self, path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_distribution': self.class_distribution,
            'forecast_days': self.forecast_days
        }
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SentimentClassifier':
        """Load a trained model."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.label_encoder = data['label_encoder']
        instance.feature_names = data['feature_names']
        instance.class_distribution = data['class_distribution']
        instance.forecast_days = data['forecast_days']
        instance.is_fitted = True
        return instance


def train_sentiment_classifier(articles_path: str = "data/processed/articles_with_sentiment.csv",
                               save_path: str = "models/predictive/sentiment_classifier.joblib") -> Dict[str, Any]:
    """
    Train sentiment classifier from articles data.
    
    Args:
        articles_path: Path to articles CSV
        save_path: Where to save trained model
        
    Returns:
        Training metrics
    """
    logger.info("Loading articles data...")
    articles = pd.read_csv(articles_path)
    
    logger.info(f"Total articles: {len(articles)}")
    if 'is_original' in articles.columns:
        logger.info(f"Original articles: {articles['is_original'].sum()}")
    
    # Initialize and prepare features
    classifier = SentimentClassifier()
    daily_df = classifier.prepare_features_from_articles(articles)
    
    logger.info(f"Training on {len(daily_df)} days of data...")
    
    # Train
    metrics = classifier.fit(daily_df)
    
    # Save model
    classifier.save(save_path)
    
    # Test predictions
    predictions = classifier.predict_next_days(daily_df, days=7)
    
    print("\n" + "="*50)
    print("📊 SENTIMENT CLASS PREDICTIONS (Next 7 Days)")
    print("="*50)
    for pred in predictions:
        date_str = pred['date'].strftime('%Y-%m-%d (%A)')
        sentiment = pred['predicted_sentiment'].upper()
        conf = pred['confidence']
        probs = pred['probabilities']
        
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[pred['predicted_sentiment']]
        
        print(f"\n{date_str}:")
        print(f"  Prediction: {emoji} {sentiment} ({conf:.0%} confidence)")
        print(f"  Probabilities: P={probs.get('positive',0):.0%} | N={probs.get('neutral',0):.0%} | Neg={probs.get('negative',0):.0%}")
    
    return metrics


if __name__ == "__main__":
    metrics = train_sentiment_classifier()
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 Score: {metrics['f1_weighted']:.2%}")
