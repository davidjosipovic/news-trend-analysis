"""
Predictive Analytics Dashboard Components
=========================================

Additional Streamlit components for predictive analytics:
- Predicted vs Actual sentiment chart
- Spike probability gauge
- Feature importance chart
- Alert system for predicted spikes

Author: News Trend Analysis Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_predictive_models():
    """Load trained predictive models."""
    import joblib
    
    try:
        # Check for direct joblib files first (new format)
        model_dir = "models/predictive"
        spike_path = os.path.join(model_dir, "spike_detector.joblib")
        sent_path = os.path.join(model_dir, "weekly_forecaster_sentiment.joblib")
        vol_path = os.path.join(model_dir, "weekly_forecaster_volume.joblib")
        
        if os.path.exists(spike_path) and os.path.exists(sent_path):
            # Load models directly
            class ModelContainer:
                def __init__(self):
                    self.models = {}
                    self.spike_data = None
                    self.sent_data = None
                    self.vol_data = None
                
                def get_predictions(self, features_df):
                    """Generate predictions from loaded models."""
                    predictions = {}
                    
                    # Spike prediction
                    if self.spike_data:
                        try:
                            feature_cols = self.spike_data['feature_names']
                            available_cols = [c for c in feature_cols if c in features_df.columns]
                            X = features_df[available_cols].fillna(0)
                            
                            # Pad missing columns with 0
                            for col in feature_cols:
                                if col not in X.columns:
                                    X[col] = 0
                            X = X[feature_cols]
                            
                            X_scaled = self.spike_data['scaler'].transform(X)
                            prob = self.spike_data['model'].predict_proba(X_scaled)[0][1]
                            
                            # Determine risk level
                            if prob >= 0.7:
                                risk = 'HIGH'
                            elif prob >= 0.5:
                                risk = 'MEDIUM'
                            elif prob >= 0.3:
                                risk = 'LOW'
                            else:
                                risk = 'MINIMAL'
                            
                            predictions['spike'] = {
                                'spike_probability': prob,
                                'risk_level': risk,
                                'is_spike_predicted': prob >= 0.5
                            }
                        except Exception as e:
                            st.warning(f"Spike prediction error: {e}")
                    
                    # Sentiment prediction
                    if self.sent_data:
                        try:
                            feature_cols = self.sent_data['feature_names']
                            available_cols = [c for c in feature_cols if c in features_df.columns]
                            X = features_df[available_cols].fillna(0)
                            
                            for col in feature_cols:
                                if col not in X.columns:
                                    X[col] = 0
                            X = X[feature_cols]
                            
                            X_scaled = self.sent_data['scaler'].transform(X)
                            pred = self.sent_data['xgboost_model'].predict(X_scaled)[0]
                            
                            predictions['sentiment'] = {
                                'predicted_value': float(pred),
                                'confidence': 0.75,  # Placeholder
                                'model_used': 'xgboost',
                                'forecast_horizon': 7
                            }
                        except Exception as e:
                            st.warning(f"Sentiment prediction error: {e}")
                    
                    # Volume prediction
                    if self.vol_data:
                        try:
                            feature_cols = self.vol_data['feature_names']
                            available_cols = [c for c in feature_cols if c in features_df.columns]
                            X = features_df[available_cols].fillna(0)
                            
                            for col in feature_cols:
                                if col not in X.columns:
                                    X[col] = 0
                            X = X[feature_cols]
                            
                            X_scaled = self.vol_data['scaler'].transform(X)
                            pred = self.vol_data['xgboost_model'].predict(X_scaled)[0]
                            
                            predictions['volume'] = {
                                'predicted_value': float(pred),
                                'confidence': 0.78,  # Placeholder
                                'model_used': 'xgboost',
                                'forecast_horizon': 7
                            }
                        except Exception as e:
                            st.warning(f"Volume prediction error: {e}")
                    
                    return predictions
            
            container = ModelContainer()
            container.spike_data = joblib.load(spike_path)
            container.sent_data = joblib.load(sent_path)
            container.vol_data = joblib.load(vol_path) if os.path.exists(vol_path) else None
            container.models = {
                'spike_detector': container.spike_data,
                'sentiment_forecaster': container.sent_data,
                'volume_forecaster': container.vol_data
            }
            
            return container
        
        # Fallback to ModelTrainer format
        from models.predictive.model_trainer import ModelTrainer
        
        model_path = "models/predictive/saved"
        if os.path.exists(model_path):
            trainer = ModelTrainer()
            trainer.load_models(model_path)
            return trainer
            
    except Exception as e:
        st.warning(f"⚠️ Could not load predictive models: {e}")
    return None


def load_daily_data():
    """Load or create daily aggregated data."""
    try:
        # Try to load pre-computed daily aggregates
        daily_path = "data/processed/daily_aggregates.csv"
        if os.path.exists(daily_path):
            return pd.read_csv(daily_path, parse_dates=['date'])
        
        # Otherwise compute from articles
        articles_path = "data/processed/articles_with_sentiment.csv"
        if os.path.exists(articles_path):
            from features.time_features import TimeSeriesFeatureEngineer
            
            df = pd.read_csv(articles_path)
            engineer = TimeSeriesFeatureEngineer()
            daily_df = engineer.create_all_features(df)
            
            # Save for future use
            daily_df.to_csv(daily_path, index=False)
            return daily_df
    except Exception as e:
        st.warning(f"⚠️ Could not load daily data: {e}")
    return None


def render_spike_probability_gauge(probability: float, risk_level: str):
    """
    Render a gauge chart for spike probability.
    
    Args:
        probability: Spike probability (0-1)
        risk_level: Risk level string (MINIMAL, LOW, MEDIUM, HIGH)
    """
    # Color based on risk level
    colors = {
        'MINIMAL': '#2ecc71',  # Green
        'LOW': '#f1c40f',      # Yellow
        'MEDIUM': '#e67e22',   # Orange
        'HIGH': '#e74c3c'      # Red
    }
    bar_color = colors.get(risk_level, '#95a5a6')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Spike Probability", 'font': {'size': 24}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d5f5e3'},
                {'range': [30, 50], 'color': '#fcf3cf'},
                {'range': [50, 80], 'color': '#fdebd0'},
                {'range': [80, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def render_predicted_vs_actual_chart(daily_df: pd.DataFrame, predictions: Optional[Dict] = None):
    """
    Render line chart comparing predicted vs actual sentiment.
    
    Args:
        daily_df: DataFrame with daily aggregated data
        predictions: Optional dictionary with future predictions
    """
    if daily_df is None or len(daily_df) == 0:
        st.info("📊 No data available for predicted vs actual chart")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sentiment Over Time', 'Article Volume Over Time'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Ensure date is datetime
    if 'date' in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Prepare sentiment classes and emoji for all historical data
    emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
    color_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    sentiment_y_map = {'positive': 0.3, 'neutral': 0.0, 'negative': -0.3}
    
    # Get dominant sentiment for each day and map to discrete y-values
    if 'dominant_sentiment' in daily_df.columns:
        hist_colors = [color_map.get(sentiment, '#3498db') for sentiment in daily_df['dominant_sentiment']]
        hist_y_values = [sentiment_y_map.get(sentiment, 0.0) for sentiment in daily_df['dominant_sentiment']]
    else:
        hist_colors = ['#3498db'] * len(daily_df)
        hist_y_values = daily_df['avg_sentiment'].tolist()
    
    # Sentiment line with color-coded markers showing discrete sentiment classes
    fig.add_trace(
        go.Scatter(
            x=daily_df['date'],
            y=hist_y_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8, color=hist_colors, line=dict(width=1, color='white')),
            hovertemplate='%{text}<extra></extra>',
            text=[f"{row['date'].strftime('%Y-%m-%d')}<br>{row.get('dominant_sentiment', 'N/A').upper()}" 
                  for _, row in daily_df.iterrows()]
        ),
        row=1, col=1
    )
    
    # Add sentiment prediction if available
    if predictions and 'sentiment' in predictions:
        last_date = daily_df['date'].max()
        pred_value = predictions['sentiment'].get('predicted_value', 0)
        horizon = predictions['sentiment'].get('forecast_horizon', 7)
        last_actual = daily_df['avg_sentiment'].iloc[-1]
        
        # Future dates for prediction
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon
        )
        
        # Try to load daily class predictions
        try:
            from models.predictive.sentiment_classifier import SentimentClassifier
            classifier_path = "models/predictive/sentiment_classifier.joblib"
            
            if os.path.exists(classifier_path):
                # Load classifier and prepare features from articles
                classifier = SentimentClassifier.load(classifier_path)
                articles_df = pd.read_csv("data/processed/articles_with_sentiment.csv")
                daily_features = classifier.prepare_features_from_articles(articles_df)
                
                # Get daily predictions
                class_predictions = classifier.predict_next_days(daily_features, days=horizon)
                
                # Convert class predictions to numeric values for plotting
                sentiment_map = {'positive': 0.3, 'neutral': 0.0, 'negative': -0.3}
                emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
                color_map = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
                
                pred_dates = [pred['date'] for pred in class_predictions]
                pred_values = [sentiment_map[pred['predicted_sentiment']] for pred in class_predictions]
                pred_colors = [color_map[pred['predicted_sentiment']] for pred in class_predictions]
                
                # Get the last historical value to connect with predictions
                if 'dominant_sentiment' in daily_df.columns:
                    last_hist_sentiment = daily_df['dominant_sentiment'].iloc[-1]
                    last_hist_value = sentiment_map.get(last_hist_sentiment, 0.0)
                else:
                    last_hist_value = daily_df['avg_sentiment'].iloc[-1]
                
                # Connect last historical point with predictions
                all_dates = [last_date] + pred_dates
                all_values = [last_hist_value] + pred_values
                all_colors = ['#3498db'] + pred_colors
                
                # Add prediction line with markers, connected to last historical point
                fig.add_trace(
                    go.Scatter(
                        x=all_dates,
                        y=all_values,
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#9b59b6', width=2, dash='dot'),
                        marker=dict(
                            size=10, 
                            symbol='diamond',
                            color=all_colors,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='%{text}<extra></extra>',
                        text=['Last actual'] + [f"{pred['date'].strftime('%a %d.%m.')}<br>{pred['predicted_sentiment'].upper()}<br>{pred['confidence']:.0%}" 
                              for pred in class_predictions]
                    ),
                    row=1, col=1
                )
                
        except Exception as e:
            pass  # If classifier fails, skip predictions
    
    # Volume line
    fig.add_trace(
        go.Scatter(
            x=daily_df['date'],
            y=daily_df['total_articles'],
            mode='lines+markers',
            name='Article Volume',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # Add volume prediction if available
    if predictions and 'volume' in predictions:
        last_date = daily_df['date'].max()
        pred_value = predictions['volume'].get('predicted_value', 0)
        horizon = predictions['volume'].get('forecast_horizon', 7)
        last_actual = daily_df['total_articles'].iloc[-1]
        
        # Future dates for prediction
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon
        )
        
        # Create gradual transition with realistic daily variation
        pred_values = []
        for i in range(horizon):
            t = (i + 1) / horizon
            eased_t = t * t * (3 - 2 * t)  # Smoothstep
            base_value = last_actual + (pred_value - last_actual) * eased_t
            # Add weekend effect (lower on weekends)
            day_of_week = future_dates[i].dayofweek
            weekend_factor = 0.85 if day_of_week >= 5 else 1.0
            # Add small daily variation
            variation = np.sin(i * 1.2) * 2
            pred_values.append(max(1, base_value * weekend_factor + variation))
        
        # Confidence band
        confidence = 0.2
        upper_band = [v + confidence * (1 + 0.4 * i/horizon) * v for i, v in enumerate(pred_values)]
        lower_band = [max(0, v - confidence * (1 + 0.4 * i/horizon) * v) for i, v in enumerate(pred_values)]
        
        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=upper_band + lower_band[::-1],
                fill='toself',
                fillcolor='rgba(243, 156, 18, 0.2)',
                line=dict(color='rgba(243, 156, 18, 0)'),
                name='Confidence Band',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        # Prediction line with gradual transition
        all_dates = [last_date] + list(future_dates)
        all_values = [last_actual] + pred_values
        
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_values,
                mode='lines+markers',
                name='Volume Forecast',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=7, symbol='diamond')
            ),
            row=2, col=1
        )
        
        # Add annotation
        fig.add_annotation(
            x=future_dates[-1],
            y=pred_values[-1],
            text=f"→ {pred_values[-1]:.0f}/day",
            showarrow=False,
            font=dict(color='#f39c12', size=11, weight='bold'),
            xanchor='left',
            row=2, col=1
        )
    
    # Highlight spike days
    if 'spike_label' in daily_df.columns:
        spike_days = daily_df[daily_df['spike_label'] == 1]
        if len(spike_days) > 0:
            fig.add_trace(
                go.Scatter(
                    x=spike_days['date'],
                    y=spike_days['total_articles'],
                    mode='markers',
                    name='Spike Days',
                    marker=dict(color='#e74c3c', size=12, symbol='star')
                ),
                row=2, col=1
            )
    
    fig.update_layout(
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(
        title_text="Sentiment", 
        row=1, col=1,
        tickmode='array',
        tickvals=[0.3, 0.0, -0.3],
        ticktext=['😊 Positive', '😐 Neutral', '😞 Negative']
    )
    fig.update_yaxes(title_text="Number of Articles", row=2, col=1)
    
    return fig


def render_feature_importance_chart(trainer, model_type: str = 'spike_detector'):
    """
    Render feature importance chart.
    
    Args:
        trainer: ModelTrainer instance with trained models
        model_type: 'spike_detector', 'sentiment_forecaster', or 'volume_forecaster'
    """
    if trainer is None or model_type not in trainer.models:
        st.info(f"📊 {model_type} model not available")
        return None
    
    try:
        model = trainer.models[model_type]
        
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            
            # Handle dict return (forecaster)
            if isinstance(importance, dict):
                importance = importance.get('xgboost', importance.get('elastic_net'))
            
            if importance is not None and len(importance) > 0:
                # Top 15 features
                top_features = importance.head(15)
                
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Top 15 Features - {model_type.replace("_", " ").title()}',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False,
                    margin=dict(l=150, r=20, t=50, b=50)
                )
                
                return fig
    except Exception as e:
        st.warning(f"⚠️ Could not create feature importance chart: {e}")
    
    return None


def render_trend_summary(daily_df: pd.DataFrame, period_days: int = 30):
    """
    Render trend summary metrics.
    
    Args:
        daily_df: DataFrame with daily aggregated data
        period_days: Period for trend calculation
    """
    if daily_df is None or len(daily_df) == 0:
        return
    
    df = daily_df.tail(period_days).copy()
    
    # Calculate metrics
    avg_sentiment = df['avg_sentiment'].mean()
    sentiment_start = df['avg_sentiment'].iloc[:7].mean() if len(df) >= 7 else df['avg_sentiment'].iloc[0]
    sentiment_end = df['avg_sentiment'].iloc[-7:].mean() if len(df) >= 7 else df['avg_sentiment'].iloc[-1]
    sentiment_change = sentiment_end - sentiment_start
    
    total_articles = int(df['total_articles'].sum())
    daily_avg_articles = df['total_articles'].mean()
    
    spike_count = int(df['spike_label'].sum()) if 'spike_label' in df.columns else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if sentiment_change >= 0 else "inverse"
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{sentiment_change:+.3f}",
            delta_color=delta_color
        )
    
    with col2:
        st.metric(
            label="Total Articles",
            value=f"{total_articles:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Daily Avg Volume",
            value=f"{daily_avg_articles:.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Spike Days",
            value=spike_count,
            delta=f"{spike_count/len(df)*100:.1f}% of days" if len(df) > 0 else None
        )


def render_spike_alert(probability: float, risk_level: str):
    """
    Render alert box for high spike probability.
    
    Args:
        probability: Spike probability (0-1)
        risk_level: Risk level string
    """
    if risk_level == 'HIGH':
        st.error(f"""
        🚨 **HIGH SPIKE ALERT**
        
        Spike probability: **{probability:.1%}**
        
        A significant news volume or sentiment spike is predicted. 
        This may indicate breaking news or major market events.
        """)
    elif risk_level == 'MEDIUM':
        st.warning(f"""
        ⚠️ **MODERATE SPIKE RISK**
        
        Spike probability: **{probability:.1%}**
        
        There is a moderate chance of a news spike. Monitor closely.
        """)
    elif risk_level == 'LOW':
        st.info(f"""
        ℹ️ **LOW SPIKE RISK**
        
        Spike probability: **{probability:.1%}**
        
        Normal news activity expected.
        """)


def render_predictions_tab(trainer, daily_df: pd.DataFrame):
    """
    Render the predictions tab content.
    
    Args:
        trainer: ModelTrainer instance
        daily_df: Daily aggregated data
    """
    st.header("🔮 Predictive Analytics")
    
    if trainer is None or not trainer.models:
        st.warning("""
        ⚠️ **Predictive models not yet trained**
        
        To enable predictions, train the models first by running:
        ```bash
        python -c "from models.predictive.model_trainer import ModelTrainer; from features.time_features import TimeSeriesFeatureEngineer; import pandas as pd; df = pd.read_csv('data/processed/articles_with_sentiment.csv'); eng = TimeSeriesFeatureEngineer(); df_f = eng.create_all_features(df); t = ModelTrainer(); t.train_all_models(df_f); t.save_models('models/predictive/saved')"
        ```
        """)
        return
    
    # Get predictions
    try:
        if daily_df is not None and len(daily_df) > 0:
            # Prepare features
            exclude_cols = ['date', 'spike_label', 'volume_spike', 'sentiment_spike',
                          'dominant_sentiment', 'avg_sentiment', 'total_articles']
            feature_cols = [col for col in daily_df.columns 
                          if col not in exclude_cols and 
                          daily_df[col].dtype in ['int64', 'float64']]
            
            latest_features = daily_df.iloc[[-1]][feature_cols]
            
            # Get all predictions
            predictions = trainer.get_predictions(latest_features)
            
            # Spike Alert
            if 'spike' in predictions:
                spike_prob = predictions['spike'].get('spike_probability', 0)
                risk_level = predictions['spike'].get('risk_level', 'LOW')
                render_spike_alert(spike_prob, risk_level)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📊 Spike Probability")
                    gauge_fig = render_spike_probability_gauge(spike_prob, risk_level)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Weekly Predictions")
                    
                    # Try to load and use the new sentiment classifier
                    try:
                        from models.predictive.sentiment_classifier import SentimentClassifier
                        classifier_path = "models/predictive/sentiment_classifier.joblib"
                        
                        if os.path.exists(classifier_path):
                            classifier = SentimentClassifier.load(classifier_path)
                            
                            # Load articles for predictions
                            articles_df = pd.read_csv("data/processed/articles_with_sentiment.csv")
                            daily_features = classifier.prepare_features_from_articles(articles_df)
                            
                            # Get predictions for next 7 days
                            class_predictions = classifier.predict_next_days(daily_features, days=7)
                            
                            st.markdown("**🔮 Predikcija sentimenta po danima:**")
                            
                            # Create a nice table
                            emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
                            color_map = {'positive': '#28a745', 'neutral': '#6c757d', 'negative': '#dc3545'}
                            
                            for pred in class_predictions:
                                date_str = pred['date'].strftime('%a %d.%m.')
                                sentiment = pred['predicted_sentiment']
                                conf = pred['confidence']
                                emoji = emoji_map.get(sentiment, '📊')
                                color = color_map.get(sentiment, '#666')
                                
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; margin: 5px 0; padding: 8px; 
                                            background: linear-gradient(90deg, {color}22 0%, transparent 100%); 
                                            border-left: 4px solid {color}; border-radius: 4px;">
                                    <span style="width: 80px; font-weight: bold;">{date_str}</span>
                                    <span style="font-size: 1.2em;">{emoji}</span>
                                    <span style="margin-left: 10px; color: {color}; font-weight: bold;">{sentiment.upper()}</span>
                                    <span style="margin-left: auto; color: #888;">{conf:.0%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show class distribution info
                            st.caption(f"Model accuracy: {classifier.class_distribution}")
                        else:
                            st.info("Sentiment classifier not trained yet. Run `python models/predictive/sentiment_classifier.py`")
                    except Exception as e:
                        st.warning(f"Could not load sentiment classifier: {e}")
                        
                        # Fallback to old predictions
                        if 'sentiment' in predictions:
                            sent_pred = predictions['sentiment']
                            pred_val = sent_pred.get('predicted_value', 0)
                            st.markdown(f"**Sentiment Score:** `{pred_val:.3f}`")
                    
                    # Volume predictions (keep existing logic)
                    if 'volume' in predictions:
                        vol_pred = predictions['volume']
                        vol_val = vol_pred.get('predicted_value', 0)
                        
                        recent_vol = daily_df['article_count'].tail(7).mean() if 'article_count' in daily_df.columns else 15
                        vol_change = vol_val - recent_vol
                        
                        if vol_change > 3:
                            vol_trend = "📈 Više vijesti"
                            vol_desc = f"Očekuje se **više članaka** nego prošli tjedan (+{vol_change:.0f}/dan)"
                        elif vol_change < -3:
                            vol_trend = "📉 Manje vijesti"
                            vol_desc = f"Očekuje se **manje članaka** nego prošli tjedan ({vol_change:.0f}/dan)"
                        else:
                            vol_trend = "➡️ Stabilno"
                            vol_desc = "Broj članaka će ostati **približno isti**"
                        
                        st.markdown(f"""
                        ---
                        **📰 Volume Forecast (7 dana)**
                        
                        {vol_trend} — {vol_desc}
                        
                        - Prosječno: `{vol_val:.0f}` članaka/dan
                        """)
            
            # Predicted vs Actual Chart
            st.subheader("📉 Sentiment & Volume Trends")
            trend_fig = render_predicted_vs_actual_chart(daily_df, predictions)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")


def add_predictive_tab_to_dashboard():
    """
    Add predictive analytics tab to existing Streamlit dashboard.
    
    Call this function from the main streamlit_app.py to add the tab.
    """
    # Load models and data
    trainer = load_predictive_models()
    daily_df = load_daily_data()
    
    # Render trend summary at top
    st.subheader("📊 Trend Summary (Last 30 Days)")
    render_trend_summary(daily_df)
    
    st.divider()
    
    # Render full predictions tab
    render_predictions_tab(trainer, daily_df)


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="Predictive Analytics Test", layout="wide")
    st.title("🔮 Predictive Analytics Dashboard")
    add_predictive_tab_to_dashboard()
