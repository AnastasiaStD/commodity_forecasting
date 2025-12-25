"""
Commodity Forecasting Library
=============================

Библиотека для прогнозирования цен на сырьевые товары.

Включает две модели:
1. MLForecaster - машинное обучение (CatBoost, XGBoost, LightGBM и др.)
2. KalmanForecaster - Time-Varying Kalman Filter

Пример использования:
    from forecasting import MLForecaster, KalmanForecaster
    
    # ML модель
    ml = MLForecaster(config='configs/sugar.yaml')
    ml.fit()
    ml.evaluate()
    ml.save_results()
    
    # Kalman модель
    kf = KalmanForecaster(config='configs/sugar.yaml')
    kf.fit()
    kf.evaluate()
    kf.save_results()
"""

from .models.ml_forecaster import MLForecaster
from .models.kalman_forecaster import KalmanForecaster

__version__ = '1.0.0'
__author__ = 'Commodity Forecasting Team'
__all__ = ['MLForecaster', 'KalmanForecaster']
