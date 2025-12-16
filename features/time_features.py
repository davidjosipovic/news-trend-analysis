"""
Time Series Feature Engineering Module
======================================

This module provides comprehensive feature engineering for time series
analysis of news sentiment and volume data.

Features include:
- Lag features (sentiment_lag_1d, sentiment_lag_7d, volume_lag_1d, etc.)
- Rolling features (rolling_mean_7d, rolling_std_7d, rolling_min/max)
- Calendar features (day_of_week, is_weekend, is_holiday_hr)
- Trend features (sentiment_momentum, volume_acceleration)

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Croatian holidays (fixed dates)
CROATIAN_HOLIDAYS = [
    (1, 1),   # Nova godina
    (1, 6),   # Bogojavljenje
    (5, 1),   # Praznik rada
    (5, 30),  # Dan državnosti
    (6, 22),  # Dan antifašističke borbe
    (8, 5),   # Dan pobjede
    (8, 15),  # Velika Gospa
    (11, 1),  # Svi sveti
    (11, 18), # Dan sjećanja
    (12, 25), # Božić
    (12, 26), # Sveti Stjepan
]


class TimeSeriesFeatureEngineer:
    """
    Feature engineering class for time series analysis of news data.
    
    This class generates features for predicting sentiment and volume
    trends in news articles.
    
    Attributes:
        lag_days: List of days for lag features
        rolling_windows: List of days for rolling window features
        
    Example:
        >>> engineer = TimeSeriesFeatureEngineer()
        >>> df_features = engineer.create_all_features(daily_df)
    """
    
    def __init__(
        self,
        lag_days: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the feature engineer.
        
        Args:
            lag_days: Days for lag features. Default: [1, 2, 3, 7, 14]
            rolling_windows: Days for rolling windows. Default: [3, 7, 14, 30]
            random_seed: Random seed for reproducibility
        """
        self.lag_days = lag_days or [1, 2, 3, 7, 14]
        self.rolling_windows = rolling_windows or [3, 7, 14, 30]
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized TimeSeriesFeatureEngineer with lag_days={self.lag_days}, "
                   f"rolling_windows={self.rolling_windows}")
    
    def aggregate_daily(self, df: pd.DataFrame, date_col: str = 'publishedAt') -> pd.DataFrame:
        """
        Aggregate article-level data to daily aggregates.
        
        Args:
            df: DataFrame with article data
            date_col: Name of date column
            
        Returns:
            DataFrame with daily aggregated metrics
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Aggregating {len(df)} articles to daily data...")
        
        # Validate required columns
        required_cols = [date_col, 'sentiment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse date
        df = df.copy()
        df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        
        # Map sentiment to numeric values
        sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['sentiment_numeric'] = df['sentiment'].map(sentiment_map).fillna(0)
        
        # Aggregate by date
        daily_agg = df.groupby('date').agg({
            'title': 'count',  # Total articles
            'sentiment_numeric': ['mean', 'std', 'min', 'max'],
            'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral',  # Dominant sentiment
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'date', 'total_articles', 
            'avg_sentiment', 'std_sentiment', 'min_sentiment', 'max_sentiment',
            'dominant_sentiment'
        ]
        
        # Fill NaN std with 0 (single article days)
        daily_agg['std_sentiment'] = daily_agg['std_sentiment'].fillna(0)
        
        # Convert date to datetime for proper time series operations
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
        
        # Add topic distribution if available
        if 'topic' in df.columns:
            topic_counts = df.groupby(['date', 'topic']).size().unstack(fill_value=0)
            topic_counts.columns = [f'topic_{col}_count' for col in topic_counts.columns]
            topic_counts = topic_counts.reset_index()
            topic_counts['date'] = pd.to_datetime(topic_counts['date'])
            daily_agg = daily_agg.merge(topic_counts, on='date', how='left')
        
        logger.info(f"Created daily aggregates with {len(daily_agg)} days")
        return daily_agg
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: DataFrame with daily aggregated data
            columns: Columns to create lags for. Default: ['avg_sentiment', 'total_articles']
            
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
        columns = columns or ['avg_sentiment', 'total_articles']
        
        logger.info(f"Creating lag features for columns: {columns}")
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping...")
                continue
                
            for lag in self.lag_days:
                feature_name = f'{col}_lag_{lag}d'
                df[feature_name] = df[col].shift(lag)
                logger.debug(f"Created feature: {feature_name}")
        
        return df
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features for specified columns.
        
        Args:
            df: DataFrame with daily aggregated data
            columns: Columns to create rolling features for
            
        Returns:
            DataFrame with added rolling features
        """
        df = df.copy()
        columns = columns or ['avg_sentiment', 'total_articles']
        
        logger.info(f"Creating rolling features for columns: {columns}")
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping...")
                continue
                
            for window in self.rolling_windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                
                # Rolling min
                df[f'{col}_rolling_min_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
                
                # Rolling max
                df[f'{col}_rolling_max_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Rolling median
                df[f'{col}_rolling_median_{window}d'] = df[col].rolling(
                    window=window, min_periods=1
                ).median()
        
        return df
    
    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar-based features.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            DataFrame with added calendar features
        """
        df = df.copy()
        
        logger.info("Creating calendar features...")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Day of month
        df['day_of_month'] = df['date'].dt.day
        
        # Week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Month
        df['month'] = df['date'].dt.month
        
        # Quarter
        df['quarter'] = df['date'].dt.quarter
        
        # Is start of month
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        
        # Is end of month
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Croatian holidays
        df['is_holiday_hr'] = df['date'].apply(self._is_croatian_holiday).astype(int)
        
        # Days since weekend
        df['days_since_weekend'] = df['day_of_week'].apply(
            lambda x: x if x <= 4 else 0
        )
        
        # Days until weekend
        df['days_until_weekend'] = df['day_of_week'].apply(
            lambda x: 5 - x if x < 5 else 0
        )
        
        return df
    
    def _is_croatian_holiday(self, date: pd.Timestamp) -> bool:
        """Check if a date is a Croatian holiday."""
        if pd.isna(date):
            return False
        return (date.month, date.day) in CROATIAN_HOLIDAYS
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based features (momentum, acceleration).
        
        Args:
            df: DataFrame with daily aggregated data
            
        Returns:
            DataFrame with added trend features
        """
        df = df.copy()
        
        logger.info("Creating trend features...")
        
        # Sentiment momentum (difference from previous day)
        if 'avg_sentiment' in df.columns:
            # 1-day momentum
            df['sentiment_momentum_1d'] = df['avg_sentiment'].diff(1)
            
            # 7-day momentum
            df['sentiment_momentum_7d'] = df['avg_sentiment'].diff(7)
            
            # Sentiment acceleration (change in momentum)
            df['sentiment_acceleration'] = df['sentiment_momentum_1d'].diff(1)
            
            # Sentiment trend direction (1=up, 0=flat, -1=down)
            df['sentiment_trend'] = np.sign(df['sentiment_momentum_1d']).fillna(0)
            
            # Consecutive days of same trend
            df['sentiment_trend_streak'] = self._calculate_streak(df['sentiment_trend'])
        
        # Volume momentum
        if 'total_articles' in df.columns:
            # 1-day change
            df['volume_change_1d'] = df['total_articles'].diff(1)
            
            # 7-day change
            df['volume_change_7d'] = df['total_articles'].diff(7)
            
            # Percentage change
            df['volume_pct_change_1d'] = df['total_articles'].pct_change(1).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            
            # Volume acceleration
            df['volume_acceleration'] = df['volume_change_1d'].diff(1)
            
            # Volume trend
            df['volume_trend'] = np.sign(df['volume_change_1d']).fillna(0)
        
        return df
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive streak of same values."""
        streak = pd.Series(index=series.index, dtype=int)
        current_streak = 0
        prev_value = None
        
        for idx, value in series.items():
            if value == prev_value and not pd.isna(value):
                current_streak += 1
            else:
                current_streak = 1
            streak[idx] = current_streak
            prev_value = value
        
        return streak
    
    def create_spike_labels(
        self, 
        df: pd.DataFrame, 
        volume_std_threshold: float = 2.0,
        sentiment_change_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Create spike labels for binary classification.
        
        Spike is defined as:
        - Volume > mean + threshold * std, OR
        - Absolute sentiment change > threshold
        
        Args:
            df: DataFrame with daily aggregated data
            volume_std_threshold: Number of std deviations for volume spike
            sentiment_change_threshold: Threshold for sentiment change spike
            
        Returns:
            DataFrame with added spike_label column
        """
        df = df.copy()
        
        logger.info(f"Creating spike labels (volume_std={volume_std_threshold}, "
                   f"sentiment_change={sentiment_change_threshold})")
        
        # Volume spike
        volume_mean = df['total_articles'].mean()
        volume_std = df['total_articles'].std()
        volume_threshold = volume_mean + volume_std_threshold * volume_std
        df['volume_spike'] = (df['total_articles'] > volume_threshold).astype(int)
        
        # Sentiment spike (large change)
        if 'sentiment_momentum_1d' not in df.columns:
            df['sentiment_momentum_1d'] = df['avg_sentiment'].diff(1)
        
        df['sentiment_spike'] = (
            df['sentiment_momentum_1d'].abs() > sentiment_change_threshold
        ).astype(int)
        
        # Combined spike label
        df['spike_label'] = ((df['volume_spike'] == 1) | (df['sentiment_spike'] == 1)).astype(int)
        
        spike_count = df['spike_label'].sum()
        logger.info(f"Detected {spike_count} spike days ({spike_count/len(df)*100:.1f}%)")
        
        return df
    
    def create_all_features(
        self, 
        df: pd.DataFrame,
        is_aggregated: bool = False,
        date_col: str = 'publishedAt'
    ) -> pd.DataFrame:
        """
        Create all features at once.
        
        Args:
            df: Input DataFrame (article-level or daily aggregated)
            is_aggregated: If True, df is already daily aggregated
            date_col: Name of date column for article-level data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating all features...")
        
        # Step 1: Aggregate if needed
        if not is_aggregated:
            df = self.aggregate_daily(df, date_col=date_col)
        
        # Step 2: Apply all feature engineering
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_calendar_features(df)
        df = self.create_trend_features(df)
        df = self.create_spike_labels(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        logger.info(f"Total features created: {len(df.columns)}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get categorized list of feature names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature categories and names
        """
        feature_names = {
            'lag_features': [col for col in df.columns if '_lag_' in col],
            'rolling_features': [col for col in df.columns if '_rolling_' in col],
            'calendar_features': [
                'day_of_week', 'is_weekend', 'day_of_month', 'week_of_year',
                'month', 'quarter', 'is_month_start', 'is_month_end',
                'is_holiday_hr', 'days_since_weekend', 'days_until_weekend'
            ],
            'trend_features': [col for col in df.columns if 'momentum' in col or 
                             'acceleration' in col or 'trend' in col or 
                             'change' in col],
            'target_features': ['spike_label', 'volume_spike', 'sentiment_spike'],
            'base_features': ['avg_sentiment', 'std_sentiment', 'total_articles']
        }
        
        # Filter to only existing columns
        for category in feature_names:
            feature_names[category] = [f for f in feature_names[category] if f in df.columns]
        
        return feature_names
    
    def prepare_model_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'avg_sentiment',
        forecast_horizon: int = 7,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and target (y) for model training.
        
        Args:
            df: DataFrame with all features
            target_col: Column to predict
            forecast_horizon: Days ahead to predict
            drop_na: Whether to drop rows with NaN
            
        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        df = df.copy()
        
        # Create future target
        df[f'{target_col}_target'] = df[target_col].shift(-forecast_horizon)
        
        # Define features to exclude
        exclude_cols = [
            'date', target_col, f'{target_col}_target',
            'dominant_sentiment', 'spike_label', 'volume_spike', 'sentiment_spike'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[f'{target_col}_target']
        
        if drop_na:
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            logger.info(f"Dropped {(~mask).sum()} rows with NaN values")
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y


def create_daily_aggregates_table(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregates table from articles data.
    
    This function creates the daily_aggregates dataset schema:
    - date, total_articles, avg_sentiment, std_sentiment
    - dominant_topics, spike_label
    
    Args:
        articles_df: DataFrame with article data
        
    Returns:
        DataFrame with daily aggregates
    """
    engineer = TimeSeriesFeatureEngineer()
    daily_df = engineer.create_all_features(articles_df, is_aggregated=False)
    
    # Select columns for daily_aggregates schema
    schema_cols = [
        'date', 'total_articles', 'avg_sentiment', 'std_sentiment',
        'min_sentiment', 'max_sentiment', 'dominant_sentiment', 'spike_label'
    ]
    
    # Add topic columns if available
    topic_cols = [col for col in daily_df.columns if col.startswith('topic_')]
    schema_cols.extend(topic_cols)
    
    # Filter to existing columns
    schema_cols = [col for col in schema_cols if col in daily_df.columns]
    
    return daily_df[schema_cols]


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load sample data
    data_path = "data/processed/articles_with_sentiment.csv"
    if os.path.exists(data_path):
        print("Loading articles data...")
        df = pd.read_csv(data_path)
        
        # Create feature engineer
        engineer = TimeSeriesFeatureEngineer()
        
        # Generate all features
        df_features = engineer.create_all_features(df)
        
        # Display results
        print(f"\nFeature DataFrame shape: {df_features.shape}")
        print(f"\nFeature categories:")
        feature_names = engineer.get_feature_names(df_features)
        for category, features in feature_names.items():
            print(f"  {category}: {len(features)} features")
        
        # Save daily aggregates
        daily_agg = create_daily_aggregates_table(df)
        output_path = "data/processed/daily_aggregates.csv"
        daily_agg.to_csv(output_path, index=False)
        print(f"\nSaved daily aggregates to {output_path}")
    else:
        print(f"Data file not found: {data_path}")
        print("Run the main pipeline first to generate article data.")
