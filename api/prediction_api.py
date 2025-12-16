"""
Prediction API Module
=====================

FastAPI endpoints for news trend predictions:
- GET /api/predictions/weekly - Weekly sentiment/volume predictions
- GET /api/predictions/spike-probability - Spike probability
- GET /api/analytics/trends - Trend analysis
- POST /api/models/retrain - Trigger model retraining

Author: News Trend Analysis Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API responses
class WeeklyPrediction(BaseModel):
    """Weekly prediction response model."""
    prediction_date: str
    forecast_horizon: int = 7
    sentiment: Dict[str, Any]
    volume: Dict[str, Any]
    model_version: str
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_date": "2025-12-16T10:00:00",
                "forecast_horizon": 7,
                "sentiment": {
                    "predicted_value": 0.15,
                    "confidence": 0.78,
                    "model_used": "xgboost"
                },
                "volume": {
                    "predicted_value": 45,
                    "confidence": 0.82,
                    "model_used": "xgboost"
                },
                "model_version": "20251216_100000"
            }
        }


class SpikeProbability(BaseModel):
    """Spike probability response model."""
    prediction_date: str
    spike_probability: float
    is_spike_predicted: bool
    risk_level: str
    thresholds: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_date": "2025-12-16T10:00:00",
                "spike_probability": 0.35,
                "is_spike_predicted": False,
                "risk_level": "LOW",
                "thresholds": {
                    "volume_std": 2.0,
                    "sentiment_change": 0.5
                }
            }
        }


class TrendAnalysis(BaseModel):
    """Trend analysis response model."""
    period_days: int
    start_date: str
    end_date: str
    metrics: Dict[str, Any]
    sentiment_trend: str
    volume_trend: str
    
    class Config:
        schema_extra = {
            "example": {
                "period_days": 30,
                "start_date": "2025-11-16",
                "end_date": "2025-12-16",
                "metrics": {
                    "avg_sentiment": 0.12,
                    "sentiment_change": 0.05,
                    "total_articles": 450,
                    "volume_change_pct": 15.5
                },
                "sentiment_trend": "IMPROVING",
                "volume_trend": "INCREASING"
            }
        }


class RetrainResponse(BaseModel):
    """Retrain response model."""
    status: str
    message: str
    model_version: str
    training_metrics: Optional[Dict[str, Any]] = None


class DailyAggregate(BaseModel):
    """Daily aggregate data model."""
    date: str
    total_articles: int
    avg_sentiment: float
    std_sentiment: float
    dominant_sentiment: str
    spike_label: int


# API Application
app = FastAPI(
    title="News Trend Analysis API",
    description="Predictive analytics API for news sentiment and volume forecasting",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state for models and data
class AppState:
    """Application state container."""
    trainer = None
    feature_engineer = None
    daily_data = None
    last_updated = None
    api_key = os.environ.get('API_KEY', 'development-key')


state = AppState()


def verify_api_key(x_api_key: str = Header(None)) -> bool:
    """Verify API key for protected endpoints."""
    if state.api_key == 'development-key':
        return True  # Allow all in development
    if x_api_key != state.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def load_models_if_needed():
    """Lazy load models when first needed."""
    if state.trainer is None:
        try:
            from models.predictive.model_trainer import ModelTrainer
            from features.time_features import TimeSeriesFeatureEngineer
            
            state.trainer = ModelTrainer()
            state.feature_engineer = TimeSeriesFeatureEngineer()
            
            # Try to load saved models
            model_path = "models/predictive/saved"
            if os.path.exists(model_path):
                state.trainer.load_models(model_path)
                logger.info("Models loaded successfully")
            else:
                logger.warning("No saved models found")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise HTTPException(status_code=500, detail="Failed to load models")


def load_data_if_needed():
    """Lazy load daily aggregated data."""
    if state.daily_data is None or state.last_updated is None or \
       (datetime.now() - state.last_updated).seconds > 3600:  # Refresh hourly
        try:
            from features.time_features import TimeSeriesFeatureEngineer
            
            # Load article data
            data_path = "data/processed/articles_with_sentiment.csv"
            if not os.path.exists(data_path):
                data_path = "data/processed/articles.csv"
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                engineer = TimeSeriesFeatureEngineer()
                state.daily_data = engineer.create_all_features(df)
                state.last_updated = datetime.now()
                logger.info(f"Loaded {len(state.daily_data)} days of data")
            else:
                logger.warning("No data files found")
                state.daily_data = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            state.daily_data = pd.DataFrame()


# API Endpoints
@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "News Trend Analysis API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": state.trainer is not None,
        "data_loaded": state.daily_data is not None
    }


@app.get("/api/predictions/weekly", response_model=WeeklyPrediction)
def get_weekly_predictions():
    """
    Get weekly predictions for sentiment and volume.
    
    Returns predicted sentiment and article volume for the next 7 days.
    """
    load_models_if_needed()
    load_data_if_needed()
    
    if state.trainer is None or not state.trainer.models:
        raise HTTPException(
            status_code=503, 
            detail="Models not trained. Please run /api/models/retrain first."
        )
    
    if state.daily_data is None or len(state.daily_data) == 0:
        raise HTTPException(status_code=404, detail="No data available")
    
    try:
        # Get latest features
        latest_features = state.daily_data.iloc[[-1]].copy()
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'spike_label', 'volume_spike', 'sentiment_spike',
                       'dominant_sentiment', 'avg_sentiment', 'total_articles']
        feature_cols = [col for col in latest_features.columns 
                       if col not in exclude_cols and 
                       latest_features[col].dtype in ['int64', 'float64']]
        
        features_for_prediction = latest_features[feature_cols]
        
        # Get predictions
        predictions = state.trainer.get_predictions(features_for_prediction)
        
        return WeeklyPrediction(
            prediction_date=predictions.get('prediction_date', datetime.now().isoformat()),
            forecast_horizon=7,
            sentiment=predictions.get('sentiment', {
                'predicted_value': 0.0,
                'confidence': 0.0,
                'model_used': 'unavailable'
            }),
            volume=predictions.get('volume', {
                'predicted_value': 0,
                'confidence': 0.0,
                'model_used': 'unavailable'
            }),
            model_version=predictions.get('model_version', 'unknown')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/spike-probability", response_model=SpikeProbability)
def get_spike_probability():
    """
    Get probability of a news spike today/tomorrow.
    
    Returns spike probability, risk level, and prediction details.
    """
    load_models_if_needed()
    load_data_if_needed()
    
    if state.trainer is None or 'spike_detector' not in state.trainer.models:
        raise HTTPException(
            status_code=503,
            detail="Spike detector not trained. Please run /api/models/retrain first."
        )
    
    if state.daily_data is None or len(state.daily_data) == 0:
        raise HTTPException(status_code=404, detail="No data available")
    
    try:
        # Get latest features
        latest_features = state.daily_data.iloc[[-1]].copy()
        
        exclude_cols = ['date', 'spike_label', 'volume_spike', 'sentiment_spike',
                       'dominant_sentiment']
        feature_cols = [col for col in latest_features.columns 
                       if col not in exclude_cols and 
                       latest_features[col].dtype in ['int64', 'float64']]
        
        features_for_prediction = latest_features[feature_cols]
        
        # Get spike prediction
        spike_pred = state.trainer.models['spike_detector'].predict_spike_probability(
            features_for_prediction
        )
        
        return SpikeProbability(**spike_pred)
        
    except Exception as e:
        logger.error(f"Spike prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/trends", response_model=TrendAnalysis)
def get_trend_analysis(period: int = Query(30, ge=7, le=365, description="Analysis period in days")):
    """
    Get trend analysis for the specified period.
    
    Analyzes sentiment and volume trends over the given time period.
    """
    load_data_if_needed()
    
    if state.daily_data is None or len(state.daily_data) == 0:
        raise HTTPException(status_code=404, detail="No data available")
    
    try:
        df = state.daily_data.copy()
        
        # Filter to requested period
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=period)
            df_period = df[df['date'] >= start_date]
        else:
            df_period = df.tail(period)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
        
        if len(df_period) == 0:
            raise HTTPException(status_code=404, detail="No data for specified period")
        
        # Calculate metrics
        avg_sentiment = df_period['avg_sentiment'].mean()
        sentiment_start = df_period['avg_sentiment'].iloc[:7].mean() if len(df_period) >= 7 else df_period['avg_sentiment'].iloc[0]
        sentiment_end = df_period['avg_sentiment'].iloc[-7:].mean() if len(df_period) >= 7 else df_period['avg_sentiment'].iloc[-1]
        sentiment_change = sentiment_end - sentiment_start
        
        total_articles = int(df_period['total_articles'].sum())
        volume_start = df_period['total_articles'].iloc[:7].mean() if len(df_period) >= 7 else df_period['total_articles'].iloc[0]
        volume_end = df_period['total_articles'].iloc[-7:].mean() if len(df_period) >= 7 else df_period['total_articles'].iloc[-1]
        volume_change_pct = ((volume_end - volume_start) / volume_start * 100) if volume_start > 0 else 0
        
        # Determine trends
        if sentiment_change > 0.05:
            sentiment_trend = "IMPROVING"
        elif sentiment_change < -0.05:
            sentiment_trend = "DECLINING"
        else:
            sentiment_trend = "STABLE"
        
        if volume_change_pct > 10:
            volume_trend = "INCREASING"
        elif volume_change_pct < -10:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        return TrendAnalysis(
            period_days=period,
            start_date=start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else str(start_date),
            end_date=end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else str(end_date),
            metrics={
                "avg_sentiment": round(avg_sentiment, 4),
                "sentiment_change": round(sentiment_change, 4),
                "total_articles": total_articles,
                "volume_change_pct": round(volume_change_pct, 2)
            },
            sentiment_trend=sentiment_trend,
            volume_trend=volume_trend
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/daily-aggregates")
def get_daily_aggregates(
    days: int = Query(30, ge=1, le=365, description="Number of days to return")
) -> List[Dict[str, Any]]:
    """
    Get daily aggregated data for visualization.
    
    Returns daily sentiment and volume aggregates.
    """
    load_data_if_needed()
    
    if state.daily_data is None or len(state.daily_data) == 0:
        raise HTTPException(status_code=404, detail="No data available")
    
    try:
        df = state.daily_data.tail(days).copy()
        
        # Select columns for output
        output_cols = ['date', 'total_articles', 'avg_sentiment', 'std_sentiment', 
                      'dominant_sentiment', 'spike_label']
        available_cols = [col for col in output_cols if col in df.columns]
        
        df_output = df[available_cols].copy()
        
        # Convert date to string
        if 'date' in df_output.columns:
            df_output['date'] = df_output['date'].astype(str)
        
        return df_output.to_dict(orient='records')
        
    except Exception as e:
        logger.error(f"Daily aggregates error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/retrain", response_model=RetrainResponse)
def retrain_models(authorized: bool = Depends(verify_api_key)):
    """
    Trigger model retraining.
    
    Protected endpoint - requires API key.
    Retrains all predictive models with latest data.
    """
    try:
        from models.predictive.model_trainer import ModelTrainer
        from features.time_features import TimeSeriesFeatureEngineer
        
        # Load fresh data
        data_path = "data/processed/articles_with_sentiment.csv"
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        df = pd.read_csv(data_path)
        
        # Create features
        engineer = TimeSeriesFeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Train models
        trainer = ModelTrainer(n_splits=3, use_optuna=False)
        results = trainer.train_all_models(df_features)
        
        # Save models
        save_path = "models/predictive/saved"
        trainer.save_models(save_path)
        
        # Update state
        state.trainer = trainer
        state.daily_data = df_features
        state.last_updated = datetime.now()
        
        return RetrainResponse(
            status="success",
            message="Models retrained successfully",
            model_version=trainer.model_version,
            training_metrics=results.get('models', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/history")
def get_prediction_history(days: int = Query(7, ge=1, le=30)) -> List[Dict[str, Any]]:
    """
    Get historical predictions vs actual values.
    
    Returns past predictions compared with actual outcomes.
    """
    # This would typically query a predictions database
    # For now, return a placeholder response
    return [
        {
            "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_sentiment": round(np.random.uniform(-0.2, 0.3), 4),
            "actual_sentiment": round(np.random.uniform(-0.2, 0.3), 4),
            "predicted_volume": int(np.random.uniform(30, 60)),
            "actual_volume": int(np.random.uniform(30, 60)),
            "spike_predicted": np.random.random() > 0.7,
            "spike_actual": np.random.random() > 0.8
        }
        for i in range(days)
    ]


# Run with: uvicorn api.prediction_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
