# Predictive Models Module
# ========================
# This module provides predictive models for news trend analysis.

from .weekly_forecaster import WeeklyForecaster
from .spike_detector import SpikeDetector
from .model_trainer import ModelTrainer

__all__ = ['WeeklyForecaster', 'SpikeDetector', 'ModelTrainer']
