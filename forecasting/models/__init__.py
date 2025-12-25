"""
Модели прогнозирования
"""

from .ml_forecaster import MLForecaster
from .kalman_forecaster import KalmanForecaster

__all__ = ['MLForecaster', 'KalmanForecaster']
