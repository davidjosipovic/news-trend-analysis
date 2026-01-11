"""
Spike Detector Module
=====================

Binary classification for detecting news volume/sentiment spikes.

Spike definition:
- Volume > mean + 2*std, OR
- Sentiment change > threshold

Uses XGBoost Classifier with SMOTE for class balancing.

Metrics: Precision, Recall, F1, ROC-AUC

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import joblib
import os
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpikeDetector:
    """
    Binary classifier for detecting news spikes.
    
    A spike is defined as:
    - Volume > mean + 1.5 * std (reduced from 2.0 for better sensitivity), OR
    - Absolute sentiment change > 0.4 (reduced from 0.5)
    
    Uses XGBoost Classifier with SMOTE for class balancing (enabled by default).
    
    Attributes:
        volume_std_threshold: Std deviations for volume spike
        sentiment_change_threshold: Threshold for sentiment spike
        xgboost_model: XGBoost classifier
        scaler: Feature scaler
        
    Example:
        >>> detector = SpikeDetector()
        >>> detector.fit(X_train, y_train)
        >>> probabilities = detector.predict_proba(X_test)
    """
    
    def __init__(
        self,
        volume_std_threshold: float = 1.5,
        sentiment_change_threshold: float = 0.4,
        use_smote: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the spike detector.
        
        Args:
            volume_std_threshold: Std deviations for volume spike
            sentiment_change_threshold: Threshold for sentiment spike
            use_smote: Whether to use SMOTE for balancing
            random_seed: Random seed for reproducibility
        """
        self.volume_std_threshold = volume_std_threshold
        self.sentiment_change_threshold = sentiment_change_threshold
        self.use_smote = use_smote
        self.random_seed = random_seed
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize XGBoost classifier
        self.model = None
        self._init_model()
        
        # SMOTE for class balancing
        self.smote = None
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                self.smote = SMOTE(random_state=random_seed)
                logger.info("SMOTE initialized for class balancing")
            except ImportError:
                logger.warning("imbalanced-learn not installed. "
                             "Install with: pip install imbalanced-learn")
                self.use_smote = False
        
        # Model metadata
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.class_distribution: Dict[int, int] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(f"SpikeDetector initialized (volume_std={volume_std_threshold}, "
                   f"sentiment_change={sentiment_change_threshold})")
    
    def _init_model(self) -> None:
        """Initialize the XGBoost classifier."""
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=1,  # Will be adjusted based on class distribution
                random_state=self.random_seed,
                n_jobs=-1,
                eval_metric='logloss',
                base_score=0.5  # Required for XGBoost 3.x
            )
            logger.info("XGBoost classifier initialized")
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise ImportError("XGBoost is required for SpikeDetector")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> 'SpikeDetector':
        """
        Fit the spike detector.
        
        Args:
            X: Feature matrix
            y: Binary spike labels (0 or 1)
            eval_set: Optional (X_val, y_val) for evaluation
            
        Returns:
            self
        """
        logger.info(f"Fitting spike detector on {len(X)} samples...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        self.class_distribution = dict(zip(unique.astype(int), counts.astype(int)))
        logger.info(f"Class distribution: {self.class_distribution}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_array = np.array(y)
        
        # Apply SMOTE if enabled and class imbalance exists
        if self.smote is not None and len(self.class_distribution) > 1:
            minority_class = min(self.class_distribution, key=self.class_distribution.get)
            minority_count = self.class_distribution[minority_class]
            
            # Only apply SMOTE if we have enough minority samples
            if minority_count >= 2:
                try:
                    X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y_array)
                    logger.info(f"SMOTE applied: {len(y_array)} -> {len(y_resampled)} samples")
                    X_scaled = X_resampled
                    y_array = y_resampled
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}. Training without balancing.")
            else:
                logger.warning(f"Not enough minority samples for SMOTE (n={minority_count})")
        
        # Adjust scale_pos_weight for remaining imbalance
        if len(self.class_distribution) > 1:
            neg_count = self.class_distribution.get(0, 1)
            pos_count = self.class_distribution.get(1, 1)
            scale_weight = neg_count / pos_count if pos_count > 0 else 1
            self.model.set_params(scale_pos_weight=scale_weight)
        
        # Fit model
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_scaled, y_array,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_scaled, y_array)
        
        self.is_fitted = True
        logger.info("Spike detector fitting complete")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict spike labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict spike probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of spike for each sample
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the spike detector.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        y_array = np.array(y)
        
        # Calculate metrics
        precision = precision_score(y_array, y_pred, zero_division=0)
        recall = recall_score(y_array, y_pred, zero_division=0)
        f1 = f1_score(y_array, y_pred, zero_division=0)
        
        # ROC-AUC (requires both classes in y_true)
        try:
            roc_auc = roc_auc_score(y_array, y_proba)
        except ValueError:
            roc_auc = None
            logger.warning("ROC-AUC could not be calculated (single class in test set)")
        
        results = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc is not None else None
        }
        
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        logger.info(f"Evaluation: Precision={precision:.4f}, Recall={recall:.4f}, "
                   f"F1={f1:.4f}, ROC-AUC={roc_auc_str}")
        
        # Confusion matrix
        cm = confusion_matrix(y_array, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': results,
            'confusion_matrix': cm.tolist()
        })
        
        return results
    
    def get_classification_report(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Classification report string
        """
        y_pred = self.predict(X)
        return classification_report(
            y, y_pred, 
            target_names=['No Spike', 'Spike'],
            zero_division=0
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from XGBoost.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def predict_spike_probability(
        self, 
        current_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict spike probability for current/next day.
        
        Args:
            current_features: Current day's features
            
        Returns:
            Dictionary with prediction details
        """
        probability = self.predict_proba(current_features)[0]
        is_spike = probability >= 0.5
        
        # Risk level based on probability
        if probability >= 0.8:
            risk_level = 'HIGH'
        elif probability >= 0.5:
            risk_level = 'MEDIUM'
        elif probability >= 0.3:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'spike_probability': round(probability, 4),
            'is_spike_predicted': bool(is_spike),
            'risk_level': risk_level,
            'prediction_date': datetime.now().isoformat(),
            'thresholds': {
                'volume_std': self.volume_std_threshold,
                'sentiment_change': self.sentiment_change_threshold
            }
        }
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model
        """
        os.makedirs(path, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_distribution': self.class_distribution,
            'volume_std_threshold': self.volume_std_threshold,
            'sentiment_change_threshold': self.sentiment_change_threshold,
            'metrics_history': self.metrics_history,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        filepath = os.path.join(path, 'spike_detector.joblib')
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, path: str) -> 'SpikeDetector':
        """
        Load model from disk.
        
        Args:
            path: Directory containing saved model
            
        Returns:
            Loaded SpikeDetector instance
        """
        filepath = os.path.join(path, 'spike_detector.joblib')
        model_data = joblib.load(filepath)
        
        instance = cls(
            volume_std_threshold=model_data['volume_std_threshold'],
            sentiment_change_threshold=model_data['sentiment_change_threshold'],
            use_smote=False  # SMOTE only needed for training
        )
        
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.class_distribution = model_data['class_distribution']
        instance.metrics_history = model_data['metrics_history']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from features.time_features import TimeSeriesFeatureEngineer
    
    data_path = "data/processed/articles_with_sentiment.csv"
    if os.path.exists(data_path):
        print("Loading and preparing data...")
        df = pd.read_csv(data_path)
        
        # Create features
        engineer = TimeSeriesFeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Prepare features and target
        exclude_cols = ['date', 'spike_label', 'volume_spike', 'sentiment_spike',
                       'dominant_sentiment']
        feature_cols = [col for col in df_features.columns 
                       if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64']]
        
        X = df_features[feature_cols].dropna()
        y = df_features.loc[X.index, 'spike_label']
        
        # Train/test split (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train detector
        print("\nTraining spike detector...")
        detector = SpikeDetector()
        detector.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluation results:")
        metrics = detector.evaluate(X_test, y_test)
        
        # Classification report
        print("\nClassification Report:")
        print(detector.get_classification_report(X_test, y_test))
        
        # Feature importance
        print("\nTop 10 important features:")
        importance = detector.get_feature_importance()
        print(importance.head(10))
        
        # Save model
        detector.save("models/predictive/saved")
        print("\nModel saved successfully!")
    else:
        print(f"Data file not found: {data_path}")
