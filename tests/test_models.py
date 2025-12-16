"""
Predictive Models Tests
=======================

Unit tests for predictive models (forecaster and spike detector).

Run with: pytest tests/test_models.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.predictive.weekly_forecaster import WeeklyForecaster, SentimentForecaster, VolumeForecaster
from models.predictive.spike_detector import SpikeDetector
from models.predictive.model_trainer import ModelTrainer


class TestWeeklyForecaster:
    """Test cases for WeeklyForecaster class."""
    
    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randn(n_samples),
        })
        
        # Target with some signal
        y = pd.Series(
            0.3 * X['feature_1'] + 0.2 * X['feature_2'] + np.random.randn(n_samples) * 0.1
        )
        
        return X, y
    
    @pytest.fixture
    def forecaster(self):
        """Create WeeklyForecaster instance."""
        return WeeklyForecaster(
            forecast_horizon=7,
            target_type='sentiment',
            use_xgboost=True,
            random_seed=42
        )
    
    def test_initialization(self, forecaster):
        """Test forecaster initialization."""
        assert forecaster.forecast_horizon == 7
        assert forecaster.target_type == 'sentiment'
        assert forecaster.is_fitted == False
    
    def test_fit(self, forecaster, sample_train_data):
        """Test model fitting."""
        X, y = sample_train_data
        
        forecaster.fit(X, y)
        
        assert forecaster.is_fitted == True
        assert len(forecaster.feature_names) == 5
    
    def test_predict_elastic_net(self, forecaster, sample_train_data):
        """Test Elastic Net predictions."""
        X, y = sample_train_data
        
        forecaster.fit(X, y)
        predictions = forecaster.predict(X, model='elastic_net')
        
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()
    
    def test_predict_xgboost(self, forecaster, sample_train_data):
        """Test XGBoost predictions."""
        X, y = sample_train_data
        
        forecaster.fit(X, y)
        
        if forecaster.xgboost_model is not None:
            predictions = forecaster.predict(X, model='xgboost')
            assert len(predictions) == len(X)
    
    def test_predict_both(self, forecaster, sample_train_data):
        """Test getting predictions from both models."""
        X, y = sample_train_data
        
        forecaster.fit(X, y)
        predictions = forecaster.predict_both(X)
        
        assert 'elastic_net' in predictions
        if forecaster.xgboost_model is not None:
            assert 'xgboost' in predictions
    
    def test_evaluate(self, forecaster, sample_train_data):
        """Test model evaluation."""
        X, y = sample_train_data
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        y_train, y_test = y.iloc[:80], y.iloc[80:]
        
        forecaster.fit(X_train, y_train)
        metrics = forecaster.evaluate(X_test, y_test)
        
        assert 'elastic_net' in metrics
        assert 'MAE' in metrics['elastic_net']
        assert 'RMSE' in metrics['elastic_net']
        assert metrics['elastic_net']['MAE'] >= 0
    
    def test_get_feature_importance(self, forecaster, sample_train_data):
        """Test feature importance extraction."""
        X, y = sample_train_data
        
        forecaster.fit(X, y)
        importance = forecaster.get_feature_importance()
        
        assert 'elastic_net' in importance
        assert len(importance['elastic_net']) == 5
        assert 'feature' in importance['elastic_net'].columns
        assert 'importance' in importance['elastic_net'].columns
    
    def test_save_and_load(self, forecaster, sample_train_data):
        """Test model persistence."""
        X, y = sample_train_data
        forecaster.fit(X, y)
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            forecaster.save(temp_dir)
            
            # Load model
            loaded = WeeklyForecaster.load(temp_dir, 'sentiment')
            
            assert loaded.is_fitted == True
            assert loaded.feature_names == forecaster.feature_names
            
            # Check predictions match
            orig_pred = forecaster.predict(X, model='elastic_net')
            loaded_pred = loaded.predict(X, model='elastic_net')
            np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_predict_next_week(self, forecaster, sample_train_data):
        """Test next week prediction."""
        X, y = sample_train_data
        forecaster.fit(X, y)
        
        result = forecaster.predict_next_week(X.iloc[[-1]])
        
        assert 'predicted_value' in result
        assert 'confidence' in result
        assert 'forecast_horizon' in result
        assert result['forecast_horizon'] == 7
    
    def test_predict_unfitted_raises_error(self, forecaster, sample_train_data):
        """Test that predict raises error when not fitted."""
        X, _ = sample_train_data
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            forecaster.predict(X)


class TestSentimentForecaster:
    """Test cases for SentimentForecaster convenience class."""
    
    def test_initialization(self):
        """Test SentimentForecaster initialization."""
        forecaster = SentimentForecaster()
        assert forecaster.target_type == 'sentiment'


class TestVolumeForecaster:
    """Test cases for VolumeForecaster convenience class."""
    
    def test_initialization(self):
        """Test VolumeForecaster initialization."""
        forecaster = VolumeForecaster()
        assert forecaster.target_type == 'volume'


class TestSpikeDetector:
    """Test cases for SpikeDetector class."""
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
        })
        
        # Create imbalanced labels (more 0s than 1s)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))
        
        return X, y
    
    @pytest.fixture
    def detector(self):
        """Create SpikeDetector instance."""
        return SpikeDetector(
            volume_std_threshold=2.0,
            sentiment_change_threshold=0.5,
            use_smote=False,  # Disable for faster tests
            random_seed=42
        )
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.volume_std_threshold == 2.0
        assert detector.sentiment_change_threshold == 0.5
        assert detector.is_fitted == False
    
    def test_fit(self, detector, sample_classification_data):
        """Test detector fitting."""
        X, y = sample_classification_data
        
        detector.fit(X, y)
        
        assert detector.is_fitted == True
        assert len(detector.feature_names) == 3
        assert len(detector.class_distribution) > 0
    
    def test_predict(self, detector, sample_classification_data):
        """Test binary predictions."""
        X, y = sample_classification_data
        
        detector.fit(X, y)
        predictions = detector.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, detector, sample_classification_data):
        """Test probability predictions."""
        X, y = sample_classification_data
        
        detector.fit(X, y)
        probabilities = detector.predict_proba(X)
        
        assert len(probabilities) == len(X)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    def test_evaluate(self, detector, sample_classification_data):
        """Test model evaluation."""
        X, y = sample_classification_data
        X_train, X_test = X.iloc[:160], X.iloc[160:]
        y_train, y_test = y.iloc[:160], y.iloc[160:]
        
        detector.fit(X_train, y_train)
        metrics = detector.evaluate(X_test, y_test)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert metrics['precision'] >= 0 and metrics['precision'] <= 1
    
    def test_get_feature_importance(self, detector, sample_classification_data):
        """Test feature importance extraction."""
        X, y = sample_classification_data
        
        detector.fit(X, y)
        importance = detector.get_feature_importance()
        
        assert len(importance) == 3
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_predict_spike_probability(self, detector, sample_classification_data):
        """Test spike probability prediction."""
        X, y = sample_classification_data
        
        detector.fit(X, y)
        result = detector.predict_spike_probability(X.iloc[[-1]])
        
        assert 'spike_probability' in result
        assert 'is_spike_predicted' in result
        assert 'risk_level' in result
        assert result['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
    
    def test_save_and_load(self, detector, sample_classification_data):
        """Test model persistence."""
        X, y = sample_classification_data
        detector.fit(X, y)
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            detector.save(temp_dir)
            loaded = SpikeDetector.load(temp_dir)
            
            assert loaded.is_fitted == True
            
            # Check predictions match
            orig_pred = detector.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_equal(orig_pred, loaded_pred)
            
        finally:
            shutil.rmtree(temp_dir)


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_features_df(self):
        """Create sample features DataFrame."""
        np.random.seed(42)
        n_days = 60
        
        dates = pd.date_range(start='2025-10-01', periods=n_days, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'total_articles': np.random.randint(20, 60, n_days),
            'avg_sentiment': np.random.uniform(-0.3, 0.3, n_days),
            'std_sentiment': np.random.uniform(0, 0.2, n_days),
            'spike_label': np.random.choice([0, 1], n_days, p=[0.85, 0.15]),
            'feature_1': np.random.randn(n_days),
            'feature_2': np.random.randn(n_days),
            'feature_3': np.random.randn(n_days),
        })
        
        return df
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer(
            n_splits=3,
            test_size=0.2,
            random_seed=42,
            use_optuna=False
        )
    
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.n_splits == 3
        assert trainer.test_size == 0.2
        assert trainer.random_seed == 42
    
    def test_prepare_data(self, trainer, sample_features_df):
        """Test data preparation with time-based split."""
        X_train, y_train, X_test, y_test = trainer.prepare_data(
            sample_features_df, 
            target_col='avg_sentiment'
        )
        
        # Check split sizes
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.1  # Approximately 20% test
        
        # Check target not in features
        assert 'avg_sentiment' not in X_train.columns
    
    def test_save_and_load_models(self, trainer):
        """Test model saving and loading."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save empty trainer
            trainer.training_history = [{'test': 'data'}]
            trainer.save_models(temp_dir)
            
            # Load
            new_trainer = ModelTrainer()
            new_trainer.load_models(temp_dir)
            
            # Metadata should be loaded
            assert new_trainer.model_version == trainer.model_version
            
        finally:
            shutil.rmtree(temp_dir)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
