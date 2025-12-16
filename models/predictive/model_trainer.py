"""
Model Trainer Module
====================

Unified training pipeline for predictive models with:
- Time series split (walk-forward validation)
- Hyperparameter tuning (Optuna)
- Model persistence
- Reproducibility (fixed seeds)

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
import joblib
import os
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified training pipeline for news trend prediction models.
    
    Features:
    - Time series cross-validation (walk-forward)
    - Optional Optuna hyperparameter tuning
    - Model versioning and persistence
    - Reproducibility with fixed random seeds
    
    Attributes:
        random_seed: Random seed for reproducibility
        n_splits: Number of time series splits
        model_version: Current model version
        
    Example:
        >>> trainer = ModelTrainer(n_splits=5)
        >>> trainer.train_all_models(df_features)
        >>> trainer.save_models("models/predictive/saved")
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42,
        use_optuna: bool = False,
        optuna_trials: int = 50
    ):
        """
        Initialize the model trainer.
        
        Args:
            n_splits: Number of time series splits for CV
            test_size: Fraction of data for final test
            random_seed: Random seed for reproducibility
            use_optuna: Whether to use Optuna for tuning
            optuna_trials: Number of Optuna trials
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_seed = random_seed
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Initialize time series splitter
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Optuna study storage
        self.optuna_studies: Dict[str, Any] = {}
        
        logger.info(f"ModelTrainer initialized (splits={n_splits}, seed={random_seed})")
    
    def prepare_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'avg_sentiment',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data with time-based train/test split.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        df = df.copy()
        
        # Default exclusions
        default_exclude = [
            'date', 'dominant_sentiment', 'spike_label', 
            'volume_spike', 'sentiment_spike'
        ]
        exclude_cols = exclude_cols or default_exclude
        exclude_cols.append(target_col)
        
        # Get feature columns
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
        ]
        
        # Handle missing values
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Time-based split (NOT random!)
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Data prepared: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X_train, y_train, X_test, y_test
    
    def cross_validate(
        self, 
        model: Any,
        X: pd.DataFrame, 
        y: pd.Series,
        scoring_func: Callable
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model with fit/predict methods
            X: Feature matrix
            y: Target values
            scoring_func: Function to calculate score
            
        Returns:
            Dictionary with CV results
        """
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Clone and fit model
            model_clone = joblib.loads(joblib.dumps(model))
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Score
            y_pred = model_clone.predict(X_val_fold)
            score = scoring_func(y_val_fold, y_pred)
            scores.append(score)
            
            logger.debug(f"Fold {fold + 1}: Score = {score:.4f}")
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'n_splits': self.n_splits
        }
    
    def tune_hyperparameters_forecaster(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        model_type: str = 'xgboost'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna for forecasting models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: 'xgboost' or 'elastic_net'
            
        Returns:
            Best hyperparameters
        """
        if not self.use_optuna:
            logger.info("Optuna disabled, using default hyperparameters")
            return {}
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed. Using default hyperparameters.")
            return {}
        
        from sklearn.metrics import mean_squared_error
        
        def objective(trial):
            if model_type == 'xgboost':
                from xgboost import XGBRegressor
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.random_seed
                }
                model = XGBRegressor(**params)
            else:  # elastic_net
                from sklearn.linear_model import ElasticNet
                params = {
                    'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                    'max_iter': 10000,
                    'random_state': self.random_seed
                }
                model = ElasticNet(**params)
            
            # Cross-validate
            cv_result = self.cross_validate(
                model, X_train, y_train,
                scoring_func=lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
            )
            
            return cv_result['mean_score']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=True)
        
        self.optuna_studies[f'forecaster_{model_type}'] = study
        
        logger.info(f"Best {model_type} params: {study.best_params}")
        return study.best_params
    
    def tune_hyperparameters_classifier(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna for spike detector.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best hyperparameters
        """
        if not self.use_optuna:
            logger.info("Optuna disabled, using default hyperparameters")
            return {}
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed. Using default hyperparameters.")
            return {}
        
        from sklearn.metrics import f1_score
        from xgboost import XGBClassifier
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                'random_state': self.random_seed,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            model = XGBClassifier(**params)
            
            cv_result = self.cross_validate(
                model, X_train, y_train,
                scoring_func=lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0)
            )
            
            return cv_result['mean_score']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=True)
        
        self.optuna_studies['spike_detector'] = study
        
        logger.info(f"Best spike detector params: {study.best_params}")
        return study.best_params
    
    def train_sentiment_forecaster(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train sentiment forecaster models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Training results
        """
        from .weekly_forecaster import SentimentForecaster
        
        logger.info("Training sentiment forecaster...")
        
        # Tune hyperparameters if enabled
        best_params = self.tune_hyperparameters_forecaster(X_train, y_train, 'xgboost')
        
        # Train model
        forecaster = SentimentForecaster(random_seed=self.random_seed)
        
        # Update XGBoost params if tuned
        if best_params and forecaster.xgboost_model is not None:
            forecaster.xgboost_model.set_params(**best_params)
        
        forecaster.fit(X_train, y_train)
        
        # Evaluate
        metrics = forecaster.evaluate(X_test, y_test)
        
        # Store model
        self.models['sentiment_forecaster'] = forecaster
        
        result = {
            'model_type': 'sentiment_forecaster',
            'metrics': metrics,
            'best_params': best_params,
            'trained_at': datetime.now().isoformat()
        }
        
        self.training_history.append(result)
        return result
    
    def train_volume_forecaster(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train volume forecaster models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Training results
        """
        from .weekly_forecaster import VolumeForecaster
        
        logger.info("Training volume forecaster...")
        
        # Tune hyperparameters if enabled
        best_params = self.tune_hyperparameters_forecaster(X_train, y_train, 'xgboost')
        
        # Train model
        forecaster = VolumeForecaster(random_seed=self.random_seed)
        
        if best_params and forecaster.xgboost_model is not None:
            forecaster.xgboost_model.set_params(**best_params)
        
        forecaster.fit(X_train, y_train)
        metrics = forecaster.evaluate(X_test, y_test)
        
        self.models['volume_forecaster'] = forecaster
        
        result = {
            'model_type': 'volume_forecaster',
            'metrics': metrics,
            'best_params': best_params,
            'trained_at': datetime.now().isoformat()
        }
        
        self.training_history.append(result)
        return result
    
    def train_spike_detector(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train spike detector.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data (labels should be spike_label)
            
        Returns:
            Training results
        """
        from .spike_detector import SpikeDetector
        
        logger.info("Training spike detector...")
        
        # Tune hyperparameters if enabled
        best_params = self.tune_hyperparameters_classifier(X_train, y_train)
        
        # Train model
        detector = SpikeDetector(random_seed=self.random_seed)
        
        if best_params:
            # Filter valid params for XGBClassifier
            valid_params = {k: v for k, v in best_params.items() 
                          if k in ['n_estimators', 'max_depth', 'learning_rate',
                                  'subsample', 'colsample_bytree', 'scale_pos_weight']}
            detector.model.set_params(**valid_params)
        
        detector.fit(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        
        self.models['spike_detector'] = detector
        
        result = {
            'model_type': 'spike_detector',
            'metrics': metrics,
            'best_params': best_params,
            'trained_at': datetime.now().isoformat()
        }
        
        self.training_history.append(result)
        return result
    
    def train_all_models(
        self, 
        df_features: pd.DataFrame,
        sentiment_target: str = 'avg_sentiment',
        volume_target: str = 'total_articles',
        spike_target: str = 'spike_label'
    ) -> Dict[str, Any]:
        """
        Train all predictive models.
        
        Args:
            df_features: DataFrame with all engineered features
            sentiment_target: Column for sentiment prediction
            volume_target: Column for volume prediction
            spike_target: Column for spike detection
            
        Returns:
            Dictionary with all training results
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive model training")
        logger.info("=" * 60)
        
        results = {
            'model_version': self.model_version,
            'training_started': datetime.now().isoformat(),
            'models': {}
        }
        
        # Prepare data for sentiment forecasting
        logger.info("\n[1/3] Training Sentiment Forecaster...")
        X_train, y_train, X_test, y_test = self.prepare_data(
            df_features, target_col=sentiment_target
        )
        results['models']['sentiment'] = self.train_sentiment_forecaster(
            X_train, y_train, X_test, y_test
        )
        
        # Prepare data for volume forecasting
        logger.info("\n[2/3] Training Volume Forecaster...")
        X_train, y_train, X_test, y_test = self.prepare_data(
            df_features, target_col=volume_target
        )
        results['models']['volume'] = self.train_volume_forecaster(
            X_train, y_train, X_test, y_test
        )
        
        # Prepare data for spike detection
        logger.info("\n[3/3] Training Spike Detector...")
        X_train, y_train, X_test, y_test = self.prepare_data(
            df_features, target_col=spike_target
        )
        results['models']['spike'] = self.train_spike_detector(
            X_train, y_train, X_test, y_test
        )
        
        results['training_completed'] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 60)
        logger.info("Model training complete!")
        logger.info("=" * 60)
        
        return results
    
    def save_models(self, path: str) -> None:
        """
        Save all trained models and metadata.
        
        Args:
            path: Directory to save models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            if hasattr(model, 'save'):
                model.save(path)
            else:
                filepath = os.path.join(path, f'{name}.joblib')
                joblib.dump(model, filepath)
                logger.info(f"Saved {name} to {filepath}")
        
        # Save training metadata
        metadata = {
            'model_version': self.model_version,
            'random_seed': self.random_seed,
            'n_splits': self.n_splits,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved training metadata to {metadata_path}")
    
    def load_models(self, path: str) -> None:
        """
        Load all models from disk.
        
        Args:
            path: Directory containing saved models
        """
        from .weekly_forecaster import WeeklyForecaster
        from .spike_detector import SpikeDetector
        
        # Load sentiment forecaster
        try:
            self.models['sentiment_forecaster'] = WeeklyForecaster.load(path, 'sentiment')
        except Exception as e:
            logger.warning(f"Could not load sentiment forecaster: {e}")
        
        # Load volume forecaster
        try:
            self.models['volume_forecaster'] = WeeklyForecaster.load(path, 'volume')
        except Exception as e:
            logger.warning(f"Could not load volume forecaster: {e}")
        
        # Load spike detector
        try:
            self.models['spike_detector'] = SpikeDetector.load(path)
        except Exception as e:
            logger.warning(f"Could not load spike detector: {e}")
        
        # Load metadata
        metadata_path = os.path.join(path, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_version = metadata.get('model_version', 'unknown')
            self.training_history = metadata.get('training_history', [])
        
        logger.info(f"Loaded models from {path}")
    
    def get_predictions(
        self, 
        current_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get predictions from all models.
        
        Args:
            current_features: Current day's feature values
            
        Returns:
            Dictionary with all predictions
        """
        predictions = {
            'prediction_date': datetime.now().isoformat(),
            'model_version': self.model_version
        }
        
        if 'sentiment_forecaster' in self.models:
            predictions['sentiment'] = self.models['sentiment_forecaster'].predict_next_week(
                current_features
            )
        
        if 'volume_forecaster' in self.models:
            predictions['volume'] = self.models['volume_forecaster'].predict_next_week(
                current_features
            )
        
        if 'spike_detector' in self.models:
            predictions['spike'] = self.models['spike_detector'].predict_spike_probability(
                current_features
            )
        
        return predictions


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
        
        # Initialize trainer
        trainer = ModelTrainer(
            n_splits=3,
            test_size=0.2,
            use_optuna=False  # Set to True for hyperparameter tuning
        )
        
        # Train all models
        results = trainer.train_all_models(df_features)
        
        # Save models
        save_path = "models/predictive/saved"
        trainer.save_models(save_path)
        
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"Data file not found: {data_path}")
