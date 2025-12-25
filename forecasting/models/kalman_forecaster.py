"""
Kalman Forecaster - Time-Varying Kalman Filter для прогнозирования
==================================================================

Методы прогнозирования β:
- last_beta: последний β постоянен
- random_walk: random walk с неопределенностью
- linear_trend: линейная экстраполяция
- ar_beta: AR модель для каждого β
- damped_trend: затухающий тренд
- bayesian_shrinkage: байесовское сжатие
- adaptive_ensemble: адаптивный ансамбль
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import warnings
import joblib
import yaml

from pykalman import KalmanFilter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


class KalmanForecaster:
    """
    Time-Varying Kalman Filter для прогнозирования.
    
    Модель:
        Observation: y_t = X_t @ β_t + v_t  (v_t ~ N(0, R))
        State:       β_t = β_{t-1} + w_t    (w_t ~ N(0, Q))
    
    Parameters:
    -----------
    config : str or dict
        Путь к YAML конфигу или словарь с параметрами
    """
    
    AVAILABLE_METHODS = [
        'last_beta', 'random_walk', 'linear_trend',
        'ar_beta', 'damped_trend', 'bayesian_shrinkage', 'adaptive_ensemble'
    ]
    
    def __init__(self, config=None, **kwargs):
        """Инициализация"""
        self.config = self._load_config(config, kwargs)
        self._validate_config()
        
        self.kf_results = None
        self.results = {}
        self.metrics = {}
        self.data = None
        self.is_fitted = False
        
        self._set_seed()
        
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(self.config['models_dir'], exist_ok=True)
    
    def _load_config(self, config, kwargs):
        """Загрузка конфигурации"""
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            cfg = config.copy()
        else:
            cfg = {}
        
        defaults = {
            'input_file': 'data/input_data.xlsx',
            'date_column': 'dt',
            'target_column': 'Price',
            'features': [],
            'initial_state_covariance': 100.0,
            'observation_covariance': 1.0,
            'transition_covariance': 0.01,
            'em_iterations': 5,
            'forecast_methods': [
                'last_beta', 'random_walk', 'linear_trend',
                'ar_beta', 'damped_trend', 'bayesian_shrinkage', 'adaptive_ensemble'
            ],
            'test_size_ratio': 0.1,
            'random_seed': 42,
            'results_dir': 'results',
            'models_dir': 'saved_models'
        }
        
        for key, value in defaults.items():
            if key not in cfg:
                cfg[key] = kwargs.get(key, value)
        
        return cfg
    
    def _validate_config(self):
        """Валидация"""
        for method in self.config['forecast_methods']:
            if method not in self.AVAILABLE_METHODS:
                raise ValueError(f"Unknown method: {method}. Available: {self.AVAILABLE_METHODS}")
    
    def _set_seed(self):
        """Установка seed"""
        import random
        seed = self.config['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def load_data(self, filepath=None):
        """Загрузка данных"""
        filepath = filepath or self.config['input_file']
        print(f"Загрузка данных из {filepath}...")
        
        self.data = pd.read_excel(filepath)
        self.data.dropna(inplace=True)
        
        date_col = self.config['date_column']
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data[date_col] = self.data[date_col].dt.to_period('M').dt.to_timestamp()
        
        print(f"Загружено строк: {len(self.data)}")
        return self
    
    def prepare_data(self):
        """Подготовка данных"""
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        if self.config['features']:
            self.feature_cols = self.config['features']
        else:
            self.feature_cols = [col for col in self.data.columns 
                                if col not in [date_col, target_col]]
        
        print(f"Признаки ({len(self.feature_cols)}): {self.feature_cols}")
        
        # Индексация по дате
        self.data_indexed = self.data[[date_col, target_col] + self.feature_cols].set_index(date_col)
        
        return self
    
    def split_data(self):
        """Разделение данных"""
        n = len(self.data_indexed)
        test_size = int(n * self.config['test_size_ratio'])
        train_size = n - test_size
        
        self.data_train = self.data_indexed.iloc[:train_size]
        self.data_test = self.data_indexed.iloc[train_size:]
        
        target_col = self.config['target_column']
        self.X_train = self.data_train[self.feature_cols].values
        self.y_train = self.data_train[target_col].values
        self.X_test = self.data_test[self.feature_cols].values
        self.y_test = self.data_test[target_col].values
        
        print(f"\nРазделение данных:")
        print(f"  Train: {train_size} ({100*train_size/n:.1f}%)")
        print(f"  Test: {test_size} ({100*test_size/n:.1f}%)")
        
        return self
    
    def _fit_kalman(self):
        """Обучение Kalman Filter"""
        target_col = self.config['target_column']
        
        y = self.data_train[target_col].values
        X = self.X_train
        T, k = X.shape
        
        observation_matrices = X.reshape(T, 1, k)
        
        kf = KalmanFilter(
            transition_matrices=np.eye(k),
            observation_matrices=observation_matrices,
            initial_state_mean=np.zeros(k),
            initial_state_covariance=np.eye(k) * self.config['initial_state_covariance'],
            observation_covariance=np.array([[self.config['observation_covariance']]]),
            transition_covariance=np.eye(k) * self.config['transition_covariance']
        )
        
        try:
            kf = kf.em(y, n_iter=self.config['em_iterations'])
        except:
            print("  EM не сошелся, используем начальные параметры")
        
        state_means, state_covs = kf.filter(y)
        smoothed_means, smoothed_covs = kf.smooth(y)
        
        self.kf_results = {
            'kalman_filter': kf,
            'betas_filtered': state_means,
            'betas_smoothed': smoothed_means,
            'state_covs_filtered': state_covs,
            'state_covs_smoothed': smoothed_covs
        }
        
        self.betas_train = state_means
        self.y_train_fitted = np.sum(self.X_train * self.betas_train, axis=1)
        
        return self
    
    def _forecast(self, method):
        """Прогнозирование методом"""
        h, k = self.X_test.shape
        kf = self.kf_results['kalman_filter']
        betas_history = self.kf_results['betas_filtered']
        
        beta_cov_forecast = None
        
        if method == 'last_beta':
            beta_forecast = np.tile(betas_history[-1], (h, 1))
            
        elif method == 'random_walk':
            beta_forecast = []
            beta_cov_forecast = []
            current_beta = betas_history[-1]
            current_cov = self.kf_results['state_covs_filtered'][-1]
            
            for t in range(h):
                next_beta = kf.transition_matrices @ current_beta
                next_cov = (kf.transition_matrices @ current_cov @ 
                           kf.transition_matrices.T + kf.transition_covariance)
                beta_forecast.append(next_beta)
                beta_cov_forecast.append(next_cov)
                current_beta, current_cov = next_beta, next_cov
            
            beta_forecast = np.array(beta_forecast)
            beta_cov_forecast = np.array(beta_cov_forecast)
            
        elif method == 'linear_trend':
            N = min(20, len(betas_history))
            beta_recent = betas_history[-N:]
            t_hist = np.arange(N)
            t_fut = np.arange(N, N + h)
            
            beta_forecast = []
            for i in range(k):
                coeffs = np.polyfit(t_hist, beta_recent[:, i], deg=1)
                beta_forecast.append(np.polyval(coeffs, t_fut))
            beta_forecast = np.array(beta_forecast).T
            
        elif method == 'ar_beta':
            beta_forecast = []
            for i in range(k):
                series = betas_history[:, i]
                try:
                    best_aic, best_model = np.inf, None
                    for p in range(1, min(6, len(series)//3)):
                        try:
                            model = AutoReg(series, lags=p, trend='c').fit()
                            if model.aic < best_aic:
                                best_aic, best_model = model.aic, model
                        except:
                            continue
                    if best_model:
                        beta_forecast.append(best_model.forecast(steps=h))
                    else:
                        beta_forecast.append(np.repeat(series[-1], h))
                except:
                    beta_forecast.append(np.repeat(series[-1], h))
            beta_forecast = np.array(beta_forecast).T
            
        elif method == 'damped_trend':
            N = min(20, len(betas_history))
            beta_recent = betas_history[-N:]
            damping = 0.95
            
            beta_forecast = []
            for i in range(k):
                t_hist = np.arange(N)
                coeffs = np.polyfit(t_hist, beta_recent[:, i], deg=1)
                slope = coeffs[0]
                last = beta_recent[-1, i]
                hist_mean = np.mean(betas_history[:, i])
                
                forecast_i = []
                for t in range(1, h + 1):
                    trend = slope * (1 - damping**t) / (1 - damping)
                    reversion = (hist_mean - last) * (1 - damping**t)
                    forecast_i.append(last + trend * 0.5 + reversion * 0.3)
                beta_forecast.append(forecast_i)
            beta_forecast = np.array(beta_forecast).T
            
        elif method == 'bayesian_shrinkage':
            N = min(30, len(betas_history))
            beta_recent = betas_history[-N:]
            
            beta_forecast = []
            for i in range(k):
                hist_mean = np.mean(betas_history[:, i])
                hist_std = np.std(betas_history[:, i])
                t_hist = np.arange(N)
                coeffs = np.polyfit(t_hist, beta_recent[:, i], deg=1)
                last = betas_history[-1, i]
                
                forecast_i = []
                for t in range(h):
                    if t == 0:
                        beta_t = last
                    else:
                        trend = np.polyval(coeffs, N + t)
                        shrink = min(0.9, 0.3 + 0.02 * t)
                        beta_t = shrink * hist_mean + (1 - shrink) * trend
                        beta_t = np.clip(beta_t, hist_mean - 3*hist_std, hist_mean + 3*hist_std)
                    forecast_i.append(beta_t)
                beta_forecast.append(forecast_i)
            beta_forecast = np.array(beta_forecast).T
            
        elif method == 'adaptive_ensemble':
            # Компоненты ансамбля
            last = np.tile(betas_history[-1], (h, 1))
            
            # AR
            ar_forecast = []
            for i in range(k):
                try:
                    model = AutoReg(betas_history[:, i], lags=min(3, len(betas_history)//4), trend='c').fit()
                    ar_forecast.append(model.forecast(steps=h))
                except:
                    ar_forecast.append(np.repeat(betas_history[-1, i], h))
            ar_forecast = np.array(ar_forecast).T
            
            # Damped
            N = min(20, len(betas_history))
            damped_forecast = []
            for i in range(k):
                t = np.arange(N)
                coeffs = np.polyfit(t, betas_history[-N:, i], deg=1)
                slope = coeffs[0]
                last_val = betas_history[-1, i]
                hist_mean = np.mean(betas_history[:, i])
                
                forecast_i = []
                for s in range(1, h + 1):
                    d = 0.95**s
                    forecast_i.append(last_val + slope * s * d * 0.5 + (hist_mean - last_val) * (1 - d) * 0.3)
                damped_forecast.append(forecast_i)
            damped_forecast = np.array(damped_forecast).T
            
            # Веса по волатильности
            beta_forecast = []
            for i in range(k):
                vol = np.std(betas_history[-20:, i]) / (np.abs(np.mean(betas_history[:, i])) + 1e-6)
                if vol < 0.2:
                    w = [0.2, 0.5, 0.3]
                elif vol < 0.5:
                    w = [0.3, 0.4, 0.3]
                else:
                    w = [0.5, 0.3, 0.2]
                beta_i = w[0] * last[:, i] + w[1] * ar_forecast[:, i] + w[2] * damped_forecast[:, i]
                beta_forecast.append(beta_i)
            beta_forecast = np.array(beta_forecast).T
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        y_pred = np.sum(self.X_test * beta_forecast, axis=1)
        
        result = {
            'y_pred': y_pred,
            'beta_forecast': beta_forecast,
            'method': method
        }
        
        if method == 'random_walk' and beta_cov_forecast is not None:
            y_std = []
            for t in range(h):
                var_y = (self.X_test[t]**2 @ np.diag(beta_cov_forecast[t]) + 
                        kf.observation_covariance[0, 0])
                y_std.append(np.sqrt(var_y))
            result['y_std'] = np.array(y_std)
            result['beta_cov_forecast'] = beta_cov_forecast
        
        return result
    
    def fit(self, data_path=None):
        """Полный цикл обучения"""
        self.load_data(data_path)
        self.prepare_data()
        self.split_data()
        
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ TIME-VARYING KALMAN FILTER")
        print("="*60)
        
        self._fit_kalman()
        
        # Метрики на train
        train_mae = mean_absolute_error(self.y_train, self.y_train_fitted)
        train_mape = np.mean(np.abs((self.y_train - self.y_train_fitted) / self.y_train)) * 100
        print(f"\nМетрики на Train:")
        print(f"  MAE: {train_mae:.3f}")
        print(f"  MAPE: {train_mape:.2f}%")
        
        # Тест стационарности
        spread = self.y_train - self.y_train_fitted
        adf = adfuller(spread)
        print(f"\nТест Дики-Фуллера: p-value = {adf[1]:.4f}")
        
        # Прогнозирование
        print("\nПрогнозирование методами:")
        for method in self.config['forecast_methods']:
            print(f"  - {method}")
            result = self._forecast(method)
            self.results[method] = result
            
            self.metrics[method] = {
                'mae': mean_absolute_error(self.y_test, result['y_pred']),
                'rmse': np.sqrt(mean_squared_error(self.y_test, result['y_pred'])),
                'mape': np.mean(np.abs((self.y_test - result['y_pred']) / self.y_test)) * 100,
                'r2': r2_score(self.y_test, result['y_pred'])
            }
        
        self.is_fitted = True
        self.best_method = min(self.metrics, key=lambda m: self.metrics[m]['mae'])
        print(f"\n🏆 Лучший метод: {self.best_method} (MAE = {self.metrics[self.best_method]['mae']:.4f})")
        
        return self
    
    def predict(self, X, method=None):
        """Прогноз на новых данных"""
        method = method or self.best_method
        # Здесь нужна реализация для новых данных
        raise NotImplementedError("Use fit() for new data")
    
    def save_results(self):
        """Сохранение результатов"""
        results_dir = self.config['results_dir']
        models_dir = self.config['models_dir']
        
        print("\nСохранение результатов...")
        
        # Метрики
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = 'Method'
        metrics_df = metrics_df.sort_values('mae')
        metrics_df.to_excel(os.path.join(results_dir, 'kalman_metrics.xlsx'))
        
        # Прогнозы
        preds_df = pd.DataFrame({
            'Date': self.data_test.index,
            'Actual': self.y_test
        })
        for method, res in self.results.items():
            preds_df[f'{method}_pred'] = res['y_pred']
        preds_df.to_excel(os.path.join(results_dir, 'kalman_predictions.xlsx'), index=False)
        
        # Betas
        betas_df = pd.DataFrame(self.betas_train, index=self.data_train.index, columns=self.feature_cols)
        betas_df.to_excel(os.path.join(results_dir, 'kalman_betas_train.xlsx'))
        
        # Модель
        joblib.dump({
            'kf_results': self.kf_results,
            'feature_cols': self.feature_cols,
            'best_method': self.best_method,
            'config': self.config
        }, os.path.join(models_dir, 'kalman_model.pkl'))
        
        print(f"  Результаты: {results_dir}")
        print(f"  Модели: {models_dir}")
        
        return self
    
    def plot_results(self):
        """Визуализация"""
        results_dir = self.config['results_dir']
        
        colors = {
            'last_beta': 'red', 'random_walk': 'green', 'linear_trend': 'purple',
            'ar_beta': 'cyan', 'damped_trend': 'magenta',
            'bayesian_shrinkage': 'brown', 'adaptive_ensemble': 'olive'
        }
        
        # 1. Все прогнозы
        plt.figure(figsize=(14, 7))
        plt.plot(self.data_train.index, self.y_train, 'k-', label='Train', alpha=0.7)
        plt.plot(self.data_train.index, self.y_train_fitted, 'orange', ls='--', label='Fitted', alpha=0.7)
        plt.plot(self.data_test.index, self.y_test, 'b-', label='Test (факт)', linewidth=2)
        
        for method, res in self.results.items():
            lw = 2.5 if method == self.best_method else 1.5
            plt.plot(self.data_test.index, res['y_pred'], 
                    label=f"{method} (MAE={self.metrics[method]['mae']:.1f})",
                    color=colors.get(method, 'gray'), linewidth=lw)
        
        plt.axvline(self.data_train.index[-1], color='gray', ls=':', lw=2)
        plt.title('Kalman Filter: сравнение методов', fontsize=14)
        plt.legend(fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kalman_all_forecasts.png'), dpi=150)
        plt.close()
        
        # 2. Эволюция β
        n_feat = len(self.feature_cols)
        fig, axes = plt.subplots(n_feat, 1, figsize=(14, 3*n_feat), sharex=True)
        if n_feat == 1:
            axes = [axes]
        
        for i, col in enumerate(self.feature_cols):
            ax = axes[i]
            ax.plot(self.data_train.index, self.betas_train[:, i], label=f'β({col})', lw=2)
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.set_ylabel(f'β({col})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Дата')
        fig.suptitle('Эволюция коэффициентов β(t)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kalman_beta_evolution.png'), dpi=150)
        plt.close()
        
        # 3. Сравнение метрик
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        methods = list(self.metrics.keys())
        
        for ax, metric in zip(axes, ['mae', 'mape', 'r2']):
            values = [self.metrics[m][metric] for m in methods]
            bars = ax.bar(methods, values, color=[colors.get(m, 'gray') for m in methods], alpha=0.7)
            best_idx = methods.index(self.best_method)
            bars[best_idx].set_alpha(1.0)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_title(metric.upper())
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'kalman_metrics_comparison.png'), dpi=150)
        plt.close()
        
        print(f"Графики сохранены в {results_dir}")
        return self
    
    def summary(self):
        """Итоги"""
        print("\n" + "="*60)
        print("ИТОГИ KALMAN FORECASTING")
        print("="*60)
        
        print(f"\n{'Method':<25} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
        print("-"*65)
        
        for method, m in sorted(self.metrics.items(), key=lambda x: x[1]['mae']):
            marker = '★' if method == self.best_method else ' '
            print(f"{marker}{method:<24} {m['mae']:<10.3f} {m['rmse']:<10.3f} "
                  f"{m['mape']:<10.2f} {m['r2']:<10.4f}")
        
        print("-"*65)
        print(f"\n🏆 Лучший метод: {self.best_method}")
        print(f"   MAE: {self.metrics[self.best_method]['mae']:.3f}")
        
        return self
