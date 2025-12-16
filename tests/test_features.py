"""
Feature Engineering Tests
=========================

Unit tests for time series feature engineering module.

Run with: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features.time_features import TimeSeriesFeatureEngineer, create_daily_aggregates_table


class TestTimeSeriesFeatureEngineer:
    """Test cases for TimeSeriesFeatureEngineer class."""
    
    @pytest.fixture
    def sample_articles_df(self):
        """Create sample articles DataFrame for testing."""
        np.random.seed(42)
        n_articles = 100
        
        # Generate dates over 30 days
        base_date = datetime(2025, 11, 1)
        dates = [base_date + timedelta(days=np.random.randint(0, 30)) 
                for _ in range(n_articles)]
        
        df = pd.DataFrame({
            'title': [f'Article {i}' for i in range(n_articles)],
            'publishedAt': dates,
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], n_articles),
            'topic': np.random.randint(0, 5, n_articles),
            'text': ['Sample text ' * 50 for _ in range(n_articles)]
        })
        
        return df
    
    @pytest.fixture
    def sample_daily_df(self):
        """Create sample daily aggregated DataFrame for testing."""
        np.random.seed(42)
        n_days = 30
        
        dates = pd.date_range(start='2025-11-01', periods=n_days, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'total_articles': np.random.randint(10, 50, n_days),
            'avg_sentiment': np.random.uniform(-0.3, 0.3, n_days),
            'std_sentiment': np.random.uniform(0, 0.2, n_days),
            'min_sentiment': np.random.uniform(-1, 0, n_days),
            'max_sentiment': np.random.uniform(0, 1, n_days),
            'dominant_sentiment': np.random.choice(['positive', 'neutral', 'negative'], n_days)
        })
        
        return df
    
    @pytest.fixture
    def engineer(self):
        """Create TimeSeriesFeatureEngineer instance."""
        return TimeSeriesFeatureEngineer(random_seed=42)
    
    def test_initialization(self, engineer):
        """Test engineer initialization with default parameters."""
        assert engineer.lag_days == [1, 2, 3, 7, 14]
        assert engineer.rolling_windows == [3, 7, 14, 30]
        assert engineer.random_seed == 42
    
    def test_custom_initialization(self):
        """Test engineer initialization with custom parameters."""
        engineer = TimeSeriesFeatureEngineer(
            lag_days=[1, 3, 5],
            rolling_windows=[7, 14],
            random_seed=123
        )
        assert engineer.lag_days == [1, 3, 5]
        assert engineer.rolling_windows == [7, 14]
    
    def test_aggregate_daily(self, engineer, sample_articles_df):
        """Test daily aggregation from article data."""
        daily_df = engineer.aggregate_daily(sample_articles_df)
        
        # Check required columns exist
        required_cols = ['date', 'total_articles', 'avg_sentiment', 'std_sentiment']
        for col in required_cols:
            assert col in daily_df.columns, f"Missing column: {col}"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(daily_df['date'])
        assert daily_df['total_articles'].dtype in ['int64', 'float64']
        
        # Check values are reasonable
        assert daily_df['total_articles'].sum() == len(sample_articles_df)
        assert daily_df['avg_sentiment'].between(-1, 1).all()
    
    def test_aggregate_daily_missing_columns(self, engineer):
        """Test aggregate_daily raises error for missing columns."""
        df = pd.DataFrame({'title': ['test'], 'text': ['content']})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.aggregate_daily(df)
    
    def test_create_lag_features(self, engineer, sample_daily_df):
        """Test lag feature creation."""
        df_with_lags = engineer.create_lag_features(sample_daily_df)
        
        # Check lag features were created
        for lag in engineer.lag_days:
            assert f'avg_sentiment_lag_{lag}d' in df_with_lags.columns
            assert f'total_articles_lag_{lag}d' in df_with_lags.columns
        
        # Verify lag values are correct
        for i in range(1, len(df_with_lags)):
            if i >= 1:
                expected_lag_1 = sample_daily_df['avg_sentiment'].iloc[i-1]
                actual_lag_1 = df_with_lags['avg_sentiment_lag_1d'].iloc[i]
                if not pd.isna(actual_lag_1):
                    assert abs(expected_lag_1 - actual_lag_1) < 1e-10
    
    def test_create_rolling_features(self, engineer, sample_daily_df):
        """Test rolling window feature creation."""
        df_with_rolling = engineer.create_rolling_features(sample_daily_df)
        
        # Check rolling features were created
        for window in engineer.rolling_windows:
            assert f'avg_sentiment_rolling_mean_{window}d' in df_with_rolling.columns
            assert f'avg_sentiment_rolling_std_{window}d' in df_with_rolling.columns
            assert f'avg_sentiment_rolling_min_{window}d' in df_with_rolling.columns
            assert f'avg_sentiment_rolling_max_{window}d' in df_with_rolling.columns
    
    def test_create_calendar_features(self, engineer, sample_daily_df):
        """Test calendar feature creation."""
        df_with_calendar = engineer.create_calendar_features(sample_daily_df)
        
        # Check calendar features exist
        calendar_features = [
            'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year',
            'month', 'quarter', 'is_month_start', 'is_month_end', 'is_holiday_hr'
        ]
        for feature in calendar_features:
            assert feature in df_with_calendar.columns, f"Missing: {feature}"
        
        # Verify day_of_week is 0-6
        assert df_with_calendar['day_of_week'].between(0, 6).all()
        
        # Verify is_weekend is binary
        assert df_with_calendar['is_weekend'].isin([0, 1]).all()
    
    def test_create_trend_features(self, engineer, sample_daily_df):
        """Test trend feature creation."""
        df_with_trends = engineer.create_trend_features(sample_daily_df)
        
        # Check trend features exist
        trend_features = [
            'sentiment_momentum_1d', 'sentiment_momentum_7d',
            'sentiment_acceleration', 'sentiment_trend',
            'volume_change_1d', 'volume_acceleration'
        ]
        for feature in trend_features:
            assert feature in df_with_trends.columns, f"Missing: {feature}"
    
    def test_create_spike_labels(self, engineer, sample_daily_df):
        """Test spike label creation."""
        df_with_spikes = engineer.create_spike_labels(sample_daily_df)
        
        # Check spike columns exist
        assert 'spike_label' in df_with_spikes.columns
        assert 'volume_spike' in df_with_spikes.columns
        assert 'sentiment_spike' in df_with_spikes.columns
        
        # Verify binary values
        assert df_with_spikes['spike_label'].isin([0, 1]).all()
    
    def test_create_all_features(self, engineer, sample_articles_df):
        """Test complete feature engineering pipeline."""
        df_all = engineer.create_all_features(sample_articles_df)
        
        # Check that we have more columns than the original
        assert len(df_all.columns) > 10
        
        # Check spike label exists (indicates all steps ran)
        assert 'spike_label' in df_all.columns
    
    def test_get_feature_names(self, engineer, sample_daily_df):
        """Test feature name categorization."""
        df_all = engineer.create_all_features(sample_daily_df, is_aggregated=True)
        feature_names = engineer.get_feature_names(df_all)
        
        assert 'lag_features' in feature_names
        assert 'rolling_features' in feature_names
        assert 'calendar_features' in feature_names
        assert 'trend_features' in feature_names
        
        # Check lists are not empty
        assert len(feature_names['lag_features']) > 0
        assert len(feature_names['rolling_features']) > 0
    
    def test_prepare_model_data(self, engineer, sample_daily_df):
        """Test model data preparation."""
        df_all = engineer.create_all_features(sample_daily_df, is_aggregated=True)
        X, y = engineer.prepare_model_data(df_all, target_col='avg_sentiment')
        
        # Check X doesn't contain target
        assert 'avg_sentiment' not in X.columns
        assert 'avg_sentiment_target' not in X.columns
        
        # Check no NaN in output
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_croatian_holiday_detection(self, engineer):
        """Test Croatian holiday detection."""
        # Christmas Day
        christmas = pd.Timestamp('2025-12-25')
        assert engineer._is_croatian_holiday(christmas) == True
        
        # New Year
        new_year = pd.Timestamp('2025-01-01')
        assert engineer._is_croatian_holiday(new_year) == True
        
        # Regular day
        regular_day = pd.Timestamp('2025-03-15')
        assert engineer._is_croatian_holiday(regular_day) == False


class TestDailyAggregatesTable:
    """Test cases for create_daily_aggregates_table function."""
    
    def test_create_daily_aggregates(self):
        """Test daily aggregates table creation."""
        np.random.seed(42)
        
        # Create sample data
        df = pd.DataFrame({
            'title': [f'Article {i}' for i in range(50)],
            'publishedAt': pd.date_range('2025-11-01', periods=50, freq='12H'),
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 50),
            'topic': np.random.randint(0, 3, 50)
        })
        
        daily_agg = create_daily_aggregates_table(df)
        
        # Check required columns
        required_cols = ['date', 'total_articles', 'avg_sentiment', 'spike_label']
        for col in required_cols:
            assert col in daily_agg.columns


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
