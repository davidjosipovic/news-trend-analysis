"""
Weekly Forecaster Module
========================

Predicts weekly sentiment and volume using:
- Elastic Net (baseline, interpretable)
- XGBoost (accuracy-focused)

Metrics: MAE, RMSE, MAPE

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import joblib
import os
from datetime import datetime

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value (0-100 scale)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class WeeklyForecaster:
    """
    Weekly forecaster for predicting sentiment and volume.
    
    Uses Elastic Net as interpretable baseline and XGBoost for accuracy.
    
    Attributes:
        forecast_horizon: Days ahead to predict (default: 7)
        elastic_net_model: Elastic Net regressor
        xgboost_model: XGBoost regressor (if installed)
        scaler: Feature scaler
        
    Example:
        >>> forecaster = WeeklyForecaster(forecast_horizon=7)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(
        self,
        forecast_horizon: int = 7,
        target_type: str = 'sentiment',
        use_xgboost: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the weekly forecaster.
        
        Args:
            forecast_horizon: Days ahead to predict
            target_type: 'sentiment' or 'volume'
            use_xgboost: Whether to use XGBoost model
            random_seed: Random seed for reproducibility
        """
        self.forecast_horizon = forecast_horizon
        self.target_type = target_type
        self.use_xgboost = use_xgboost
        self.random_seed = random_seed
        
        # Initialize models
        self.scaler = StandardScaler()
        
        # Elastic Net (baseline)
        self.elastic_net_model = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=random_seed
        )
        
        # XGBoost (if available)
        self.xgboost_model = None
        if use_xgboost:
            try:
                from xgboost import XGBRegressor
                self.xgboost_model = XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_seed,
                    n_jobs=-1
                )
                logger.info("XGBoost model initialized")
            except ImportError:
                logger.warning("XGBoost not installed. Install with: pip install xgboost")
                self.use_xgboost = False
        
        # Model metadata
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        logger.info(f"WeeklyForecaster initialized for {target_type} "
                   f"(horizon={forecast_horizon} days)")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> 'WeeklyForecaster':
        """
        Fit both Elastic Net and XGBoost models.
        
        Args:
            X: Feature matrix
            y: Target values
            eval_set: Optional (X_val, y_val) for evaluation
            
        Returns:
            self
        """
        logger.info(f"Fitting models on {len(X)} samples...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Elastic Net
        logger.info("Training Elastic Net model...")
        self.elastic_net_model.fit(X_scaled, y)
        
        # Fit XGBoost
        if self.xgboost_model is not None:
            logger.info("Training XGBoost model...")
            if eval_set is not None:
                X_val, y_val = eval_set
                X_val_scaled = self.scaler.transform(X_val)
                self.xgboost_model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            else:
                self.xgboost_model.fit(X_scaled, y)
        
        self.is_fitted = True
        logger.info("Model fitting complete")
        
        return self
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model: str = 'xgboost'
    ) -> np.ndarray:
        """
        Make predictions using specified model.
        
        Args:
            X: Feature matrix
            model: 'elastic_net' or 'xgboost'
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if model == 'xgboost' and self.xgboost_model is not None:
            return self.xgboost_model.predict(X_scaled)
        else:
            return self.elastic_net_model.predict(X_scaled)
    
    def predict_both(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from both models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {
            'elastic_net': self.predict(X, model='elastic_net')
        }
        
        if self.xgboost_model is not None:
            predictions['xgboost'] = self.predict(X, model='xgboost')
        
        return predictions
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models and return metrics.
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            Dictionary with metrics for each model
        """
        predictions = self.predict_both(X)
        y_array = np.array(y)
        
        results = {}
        
        for model_name, y_pred in predictions.items():
            mae = mean_absolute_error(y_array, y_pred)
            rmse = np.sqrt(mean_squared_error(y_array, y_pred))
            mape = mean_absolute_percentage_error(y_array, y_pred)
            
            results[model_name] = {
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'MAPE': round(mape, 2) if not np.isnan(mape) else None
            }
            
            logger.info(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'target_type': self.target_type,
            'metrics': results
        })
        
        return results
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from both models.
        
        Returns:
            Dictionary with feature importance DataFrames
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importance = {}
        
        # Elastic Net coefficients
        en_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.elastic_net_model.coef_)
        }).sort_values('importance', ascending=False)
        importance['elastic_net'] = en_importance
        
        # XGBoost feature importance
        if self.xgboost_model is not None:
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.xgboost_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance['xgboost'] = xgb_importance
        
        return importance
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model
        """
        os.makedirs(path, exist_ok=True)
        
        model_data = {
            'elastic_net_model': self.elastic_net_model,
            'xgboost_model': self.xgboost_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'forecast_horizon': self.forecast_horizon,
            'target_type': self.target_type,
            'metrics_history': self.metrics_history,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        filepath = os.path.join(path, f'weekly_forecaster_{self.target_type}.joblib')
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, path: str, target_type: str = 'sentiment') -> 'WeeklyForecaster':
        """
        Load model from disk.
        
        Args:
            path: Directory containing saved model
            target_type: 'sentiment' or 'volume'
            
        Returns:
            Loaded WeeklyForecaster instance
        """
        filepath = os.path.join(path, f'weekly_forecaster_{target_type}.joblib')
        model_data = joblib.load(filepath)
        
        instance = cls(
            forecast_horizon=model_data['forecast_horizon'],
            target_type=model_data['target_type'],
            use_xgboost=model_data['xgboost_model'] is not None
        )
        
        instance.elastic_net_model = model_data['elastic_net_model']
        instance.xgboost_model = model_data['xgboost_model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.metrics_history = model_data['metrics_history']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def predict_next_week(
        self, 
        current_features: pd.DataFrame,
        model: str = 'xgboost'
    ) -> Dict[str, Any]:
        """
        Predict values for the next week.
        
        Args:
            current_features: Current day's features
            model: Model to use for prediction
            
        Returns:
            Dictionary with prediction and confidence
        """
        prediction = self.predict(current_features, model=model)[0]
        
        # Estimate confidence based on recent metrics
        confidence = 0.5  # Default
        if self.metrics_history:
            recent_metrics = self.metrics_history[-1]['metrics'].get(model, {})
            mape = recent_metrics.get('MAPE', 50)
            if mape is not None:
                confidence = max(0, min(1, 1 - (mape / 100)))
        
        return {
            'predicted_value': round(prediction, 4),
            'confidence': round(confidence, 2),
            'forecast_horizon': self.forecast_horizon,
            'model_used': model,
            'prediction_date': datetime.now().isoformat()
        }


class SentimentForecaster(WeeklyForecaster):
    """Convenience class for sentiment forecasting."""
    
    def __init__(self, forecast_horizon: int = 7, **kwargs):
        super().__init__(
            forecast_horizon=forecast_horizon,
            target_type='sentiment',
            **kwargs
        )


class VolumeForecaster(WeeklyForecaster):
    """Convenience class for volume forecasting."""
    
    def __init__(self, forecast_horizon: int = 7, **kwargs):
        super().__init__(
            forecast_horizon=forecast_horizon,
            target_type='volume',
            **kwargs
        )


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
        
        # Prepare data for sentiment prediction
        X, y = engineer.prepare_model_data(
            df_features, 
            target_col='avg_sentiment',
            forecast_horizon=7
        )
        
        # Train/test split (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train forecaster
        print("\nTraining sentiment forecaster...")
        forecaster = SentimentForecaster()
        forecaster.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluation results:")
        metrics = forecaster.evaluate(X_test, y_test)
        
        # Feature importance
        print("\nTop 10 important features (XGBoost):")
        importance = forecaster.get_feature_importance()
        if 'xgboost' in importance:
            print(importance['xgboost'].head(10))
        
        # Save model
        forecaster.save("models/predictive/saved")
        print("\nModel saved successfully!")
    else:
        print(f"Data file not found: {data_path}")
