"""
ML Forecaster - Модуль машинного обучения для прогнозирования
=============================================================

Поддерживаемые модели:
- CatBoost
- XGBoost
- LightGBM
- Ridge
- Lasso
- KNN
- SVR
- SGD
- Random Forest
- AdaBoost

Использование для прогноза на новых данных:
    # Вариант 1: Загрузка модели и прогноз
    forecaster = MLForecaster.load('saved_models/sugar')
    predictions = forecaster.predict_from_file('data/future_features.xlsx')
    
    # Вариант 2: Через конфиг
    forecaster = MLForecaster(config='configs/sugar.yaml')
    forecaster.load_model()
    predictions = forecaster.predict_from_file()  # использует forecast_file из конфига
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
import joblib
import yaml
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import SGDRegressor, Ridge, Lasso
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLForecaster:
    """
    Класс для прогнозирования с использованием ML моделей.
    
    Parameters:
    -----------
    config : str or dict
        Путь к YAML конфигу или словарь с параметрами
    
    Attributes:
    -----------
    models : dict
        Обученные модели
    results : dict
        Результаты прогнозирования
    metrics : dict
        Метрики качества
    """
    
    AVAILABLE_MODELS = [
        'catboost', 'xgb', 'lgb', 'ridge', 'lasso',
        'knn', 'svr', 'sgd', 'random_forest', 'adaboost'
    ]
    
    def __init__(self, config=None, **kwargs):
        """Инициализация с конфигурацией"""
        self.config = self._load_config(config, kwargs)
        self._validate_config()
        
        self.models = {}
        self.results = {}
        self.metrics = {}
        self.feature_importances = {}
        self.scaler = None
        self.feature_cols = None
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
            'forecast_file': '', 
            'date_column': 'Date',
            'target_column': 'Price',
            'features': [],
            'expert_features': [],
            'max_features': 10,
            'correlation_threshold': 0.3,
            'create_features': True,
            'lags': [3, 6, 9, 12, 18, 24],
            'ma_windows': [3, 6, 9, 12],
            'test_size_ratio': 0.1,
            'val_size_ratio': 0.15,
            'models_to_train': ['catboost', 'xgb', 'lgb', 'ridge', 'random_forest'],
            'n_optuna_trials': 100,
            'random_seed': 42,
            'results_dir': 'results',
            'models_dir': 'saved_models'
        }
        
        for key, value in defaults.items():
            if key not in cfg:
                cfg[key] = kwargs.get(key, value)
        
        return cfg
    
    def _validate_config(self):
        """Валидация конфигурации"""
        for model in self.config['models_to_train']:
            if model not in self.AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {model}. Available: {self.AVAILABLE_MODELS}")
    
    def _set_seed(self):
        """Установка random seed"""
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
        
        date_col = self.config['date_column']
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        print(f"Загружено строк: {len(self.data)}")
        return self
    
    def prepare_features(self):
        """Подготовка признаков"""
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        if self.config['features']:
            base_cols = self.config['features']
        else:
            base_cols = [col for col in self.data.columns 
                        if col not in [date_col, target_col]]
        
        print(f"Базовые признаки ({len(base_cols)}): {base_cols}")

        if self.config['create_features']:
            print(f"Создание лагов: {self.config['lags']}")
            for col in base_cols:
                for lag in self.config['lags']:
                    self.data[f"{col}_lag_{lag}"] = self.data[col].shift(lag)
            
            print(f"Создание MA: {self.config['ma_windows']}")
            for col in base_cols:
                for window in self.config['ma_windows']:
                    self.data[f"{col}_ma_{window}"] = self.data[col].rolling(window=window).mean()
        
        all_features = [col for col in self.data.columns 
                       if col not in [date_col, target_col]]
        
        print(f"Всего признаков после генерации: {len(all_features)}")
        
        return self
    
    def select_features(self):
        """Отбор признаков"""
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        all_features = [col for col in self.data.columns 
                       if col not in [date_col, target_col]]
        
        df_valid = self.data.dropna(subset=all_features + [target_col])
        X = df_valid[all_features]
        y = df_valid[target_col]
        
        if self.config['expert_features']:
            print("Режим: ЭКСПЕРТНЫЙ ВЫБОР")
            missing = [f for f in self.config['expert_features'] if f not in all_features]
            if missing:
                raise ValueError(f"Признаки не найдены: {missing}")
            
            selected = self.config['expert_features'][:self.config['max_features']]
        else:
            print("Режим: АВТОМАТИЧЕСКИЙ ОТБОР")
            correlations = {}
            for col in all_features:
                valid_mask = X[col].notna() & y.notna()
                if valid_mask.sum() > 0:
                    correlations[col] = X.loc[valid_mask, col].corr(y.loc[valid_mask])
            
            corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
            corr_df['abs_correlation'] = corr_df['correlation'].abs()
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            
            threshold = self.config['correlation_threshold']
            selected = corr_df[corr_df['abs_correlation'] >= threshold].index.tolist()

            max_feat = self.config['max_features']
            if len(selected) > max_feat:
                selected = corr_df.head(max_feat).index.tolist()
            elif len(selected) == 0:
                selected = corr_df.head(max_feat).index.tolist()
        
        self.feature_cols = selected
        print(f"Выбрано признаков: {len(self.feature_cols)}")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"  {i}. {col}")
        
        return self
    
    def split_data(self):
        """Разделение данных на train/val/test"""
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        df = self.data[[date_col, target_col] + self.feature_cols].dropna()
        
        n = len(df)
        test_size = int(n * self.config['test_size_ratio'])
        val_size = int(n * self.config['val_size_ratio'])
        train_size = n - val_size - test_size
        
        self.train_df = df.iloc[:train_size].copy()
        self.val_df = df.iloc[train_size:train_size+val_size].copy()
        self.test_df = df.iloc[train_size+val_size:].copy()
        
        print(f"\nРазделение данных:")
        print(f"  Train: {train_size} ({100*train_size/n:.1f}%)")
        print(f"  Val: {val_size} ({100*val_size/n:.1f}%)")
        print(f"  Test: {test_size} ({100*test_size/n:.1f}%)")
        
        self.scaler = StandardScaler()
        
        self.X_train = self.scaler.fit_transform(self.train_df[self.feature_cols])
        self.y_train = self.train_df[target_col].values
        
        self.X_val = self.scaler.transform(self.val_df[self.feature_cols])
        self.y_val = self.val_df[target_col].values
        
        self.X_test = self.scaler.transform(self.test_df[self.feature_cols])
        self.y_test = self.test_df[target_col].values
        
        self.train_dates = self.train_df[date_col].values
        self.val_dates = self.val_df[date_col].values
        self.test_dates = self.test_df[date_col].values
        
        return self
    
    def fit(self, data_path=None):
        """Полный цикл обучения"""
        self.load_data(data_path)
        self.prepare_features()
        self.select_features()
        self.split_data()
        
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        for model_name in self.config['models_to_train']:
            print(f"\n{'─'*50}")
            print(f"Модель: {model_name.upper()}")
            print(f"{'─'*50}")
            
            model = self._train_model(model_name)
            if model is not None:
                self.models[model_name] = model
                
                train_pred = model.predict(self.X_train)
                val_pred = model.predict(self.X_val)
                test_pred = model.predict(self.X_test)
                
                self.results[model_name] = {
                    'train_pred': train_pred,
                    'val_pred': val_pred,
                    'test_pred': test_pred
                }
                
                self.metrics[model_name] = {
                    'mae': mean_absolute_error(self.y_test, test_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)),
                    'mape': np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100,
                    'r2': r2_score(self.y_test, test_pred)
                }
                
                print(f"  MAE: {self.metrics[model_name]['mae']:.4f}")
                print(f"  RMSE: {self.metrics[model_name]['rmse']:.2f}")
                print(f"  MAPE: {self.metrics[model_name]['mape']:.2f}%")
        
        self.is_fitted = True
        self.best_model = min(self.metrics, key=lambda m: self.metrics[m]['mae'])
        print(f"\nЛучшая модель: {self.best_model} (MAE = {self.metrics[self.best_model]['mae']:.4f})")
        
        print("\n" + "-"*50)
        print("Расчёт важности признаков...")
        self._calculate_all_feature_importances()
        
        return self
    
    def _train_model(self, model_name):
        """Обучение одной модели"""
        n_trials = self.config['n_optuna_trials']
        seed = self.config['random_seed']
        
        try:
            if model_name == 'catboost':
                return self._train_catboost(n_trials, seed)
            elif model_name == 'xgb':
                return self._train_xgb(n_trials, seed)
            elif model_name == 'lgb':
                return self._train_lgb(n_trials, seed)
            elif model_name == 'ridge':
                return self._train_ridge(n_trials, seed)
            elif model_name == 'lasso':
                return self._train_lasso(n_trials, seed)
            elif model_name == 'knn':
                return self._train_knn(n_trials, seed)
            elif model_name == 'svr':
                return self._train_svr(seed)
            elif model_name == 'sgd':
                return self._train_sgd(n_trials, seed)
            elif model_name == 'random_forest':
                return self._train_random_forest(n_trials, seed)
            elif model_name == 'adaboost':
                return self._train_adaboost(n_trials, seed)
        except Exception as e:
            print(f"  Ошибка: {str(e)}")
            return None
    
    def _train_catboost(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
                'random_seed': seed, 'verbose': 0
            }
            model = CatBoostRegressor(**params)
            model.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val), 
                     early_stopping_rounds=50, verbose=0)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = CatBoostRegressor(**study.best_params, random_seed=seed, verbose=0)
        model.fit(X_full, y_full)
        return model
    
    def _train_xgb(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'random_state': seed, 'verbosity': 0
            }
            model = xgb.XGBRegressor(**params)
            model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = xgb.XGBRegressor(**study.best_params, random_state=seed, verbosity=0)
        model.fit(X_full, y_full)
        return model
    
    def _train_lgb(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 700),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'random_state': seed, 'verbosity': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)],
                     callbacks=[lgb.early_stopping(5, verbose=False)])
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = lgb.LGBMRegressor(**study.best_params, random_state=seed, verbosity=-1)
        model.fit(X_full, y_full)
        return model
    
    def _train_ridge(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-6, 100, log=True)
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = Ridge(**study.best_params, random_state=seed)
        model.fit(X_full, y_full)
        return model
    
    def _train_lasso(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-6, 100, log=True)
            model = Lasso(alpha=alpha, random_state=seed, max_iter=10000)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = Lasso(**study.best_params, random_state=seed, max_iter=10000)
        model.fit(X_full, y_full)
        return model
    
    def _train_knn(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
            }
            model = KNeighborsRegressor(**params)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = KNeighborsRegressor(**study.best_params)
        model.fit(X_full, y_full)
        return model
    
    def _train_svr(self, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 0.2, 0.5]}
        grid = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, cv=TimeSeriesSplit(3), n_jobs=-1)
        grid.fit(X_full, y_full)
        return grid.best_estimator_
    
    def _train_sgd(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'loss': trial.suggest_categorical('loss', ['squared_error', 'huber']),
                'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
                'random_state': seed
            }
            model = SGDRegressor(**params)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = SGDRegressor(**study.best_params, random_state=seed)
        model.fit(X_full, y_full)
        return model
    
    def _train_random_forest(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': seed
            }
            model = RandomForestRegressor(**params)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = RandomForestRegressor(**study.best_params, random_state=seed)
        model.fit(X_full, y_full)
        return model
    
    def _train_adaboost(self, n_trials, seed):
        X_full = np.vstack((self.X_train, self.X_val))
        y_full = np.concatenate((self.y_train, self.y_val))
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'random_state': seed
            }
            model = AdaBoostRegressor(**params)
            model.fit(self.X_train, self.y_train)
            return mean_absolute_error(self.y_val, model.predict(self.X_val))
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        model = AdaBoostRegressor(**study.best_params, random_state=seed)
        model.fit(X_full, y_full)
        return model

    
    def _calculate_all_feature_importances(self):
        """
        Расчёт важности признаков для всех обученных моделей.
        
        Использует:
        - feature_importances_ для tree-based моделей (CatBoost, XGB, LGB, RF, AdaBoost)
        - coef_ для линейных моделей (Ridge, Lasso, SGD)
        - SHAP values для моделей без встроенной важности (KNN, SVR)
        """
        for model_name, model in self.models.items():
            print(f"  {model_name}...", end=" ")
            try:
                importance = self._calculate_single_importance(model_name, model)
                if importance is not None:
                    self.feature_importances[model_name] = importance
                    print("✓")
                else:
                    print("пропущено")
            except Exception as e:
                print(f"ошибка: {e}")
    
    def _calculate_single_importance(self, model_name, model):
        """
        Расчёт важности для одной модели.
        
        Returns:
        --------
        dict с ключами:
            - 'values': np.array с важностями
            - 'method': str метод расчёта ('native', 'coef', 'shap')
            - 'normalized': np.array нормализованные значения (сумма = 100)
        """
        importance = None
        method = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            method = 'native'
    
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            method = 'coef'
        
        elif model_name in ['knn', 'svr']:
            importance = self._calculate_shap_importance(model)
            method = 'shap'
        
        if importance is None:
            return None
        
        total = np.sum(np.abs(importance))
        if total > 0:
            normalized = np.abs(importance) / total * 100
        else:
            normalized = np.zeros_like(importance)
        
        return {
            'values': importance,
            'method': method,
            'normalized': normalized,
            'features': self.feature_cols
        }
    
    def _calculate_shap_importance(self, model, n_samples=100):
        """
        Расчёт SHAP values для модели.
        
        Parameters:
        -----------
        model : sklearn model
            Обученная модель
        n_samples : int
            Количество сэмплов для расчёта (для ускорения)
        
        Returns:
        --------
        np.array с mean(|SHAP|) для каждого признака
        """
        
        X_sample = self.X_train
        if len(X_sample) > n_samples:
            idx = np.random.choice(len(X_sample), n_samples, replace=False)
            X_sample = X_sample[idx]
        
        try:
            explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            shap_values = explainer.shap_values(X_sample)
            importance = np.abs(shap_values).mean(axis=0)
            return importance
        except Exception as e:
            print(f"SHAP error: {e}")
            return None
    
    def get_feature_importance(self, model_name=None, top_n=None):
        """
        Получение важности признаков.
        
        Parameters:
        -----------
        model_name : str, optional
            Какую модель использовать. По умолчанию best_model
        top_n : int, optional
            Вернуть только top-N признаков
        
        Returns:
        --------
        pd.DataFrame с важностью признаков
        """
        model_name = model_name or self.best_model
        
        if model_name not in self.feature_importances:
            if model_name in self.models:
                imp = self._calculate_single_importance(model_name, self.models[model_name])
                if imp:
                    self.feature_importances[model_name] = imp
                else:
                    print(f"Не удалось рассчитать важность для {model_name}")
                    return None
            else:
                print(f"Модель {model_name} не найдена")
                return None
        
        imp_data = self.feature_importances[model_name]
        
        result = pd.DataFrame({
            'feature': imp_data['features'],
            'importance': imp_data['values'],
            'importance_pct': imp_data['normalized'],
            'method': imp_data['method']
        }).sort_values('importance_pct', ascending=False)
        
        result['rank'] = range(1, len(result) + 1)
        result = result[['rank', 'feature', 'importance', 'importance_pct', 'method']]
        
        if top_n:
            result = result.head(top_n)
        
        return result
    
    def get_all_feature_importances(self):
        """
        Получение важности признаков для всех моделей.
        
        Returns:
        --------
        pd.DataFrame с важностями по всем моделям
        """
        all_data = []
        
        for model_name in self.feature_importances:
            imp = self.get_feature_importance(model_name)
            if imp is not None:
                imp = imp.copy()
                imp['model'] = model_name
                all_data.append(imp)
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_aggregated_importance(self, method='mean'):
        """
        Агрегированная важность по всем моделям.
        
        Parameters:
        -----------
        method : str
            'mean' - среднее, 'median' - медиана, 'rank' - средний ранг
        
        Returns:
        --------
        pd.DataFrame с агрегированной важностью
        """
        all_imp = self.get_all_feature_importances()
        if all_imp is None:
            return None
        
        if method == 'mean':
            agg = all_imp.groupby('feature')['importance_pct'].mean()
        elif method == 'median':
            agg = all_imp.groupby('feature')['importance_pct'].median()
        elif method == 'rank':
            agg = all_imp.groupby('feature')['rank'].mean()
            # Инвертируем ранг (меньше = лучше)
            agg = agg.max() - agg + 1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = pd.DataFrame({
            'feature': agg.index,
            f'importance_{method}': agg.values
        }).sort_values(f'importance_{method}', ascending=False)
        
        result['rank'] = range(1, len(result) + 1)
        result = result[['rank', 'feature', f'importance_{method}']]
        
        return result
    
    def predict(self, X, model_name=None):
        """Прогноз на новых данных"""
        model_name = model_name or self.best_model
        X_scaled = self.scaler.transform(X) if not isinstance(X, np.ndarray) else X
        return self.models[model_name].predict(X_scaled)
    
    def save_results(self):
        """Сохранение результатов и моделей для последующего прогноза"""
        results_dir = self.config['results_dir']
        models_dir = self.config['models_dir']
        
        print("\nСохранение результатов...")
        
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = 'Model'
        metrics_df = metrics_df.sort_values('mae')
        metrics_df.to_excel(os.path.join(results_dir, 'ml_metrics.xlsx'))
        
        preds_df = pd.DataFrame({
            'Date': self.test_dates,
            'Actual': self.y_test
        })
        for model_name, res in self.results.items():
            preds_df[f'{model_name}_pred'] = res['test_pred']
        preds_df.to_excel(os.path.join(results_dir, 'ml_predictions.xlsx'), index=False)
        

        if self.feature_importances:
            all_imp = self.get_all_feature_importances()
            if all_imp is not None:
                all_imp.to_excel(os.path.join(results_dir, 'ml_feature_importance_detailed.xlsx'), index=False)
            
            pivot_data = []
            for model_name, imp_data in self.feature_importances.items():
                for feat, val in zip(imp_data['features'], imp_data['normalized']):
                    pivot_data.append({'feature': feat, 'model': model_name, 'importance_pct': val})
            
            if pivot_data:
                pivot_df = pd.DataFrame(pivot_data)
                pivot_table = pivot_df.pivot(index='feature', columns='model', values='importance_pct')
                pivot_table['mean'] = pivot_table.mean(axis=1)
                pivot_table = pivot_table.sort_values('mean', ascending=False)
                pivot_table.to_excel(os.path.join(results_dir, 'ml_feature_importance_pivot.xlsx'))
            
            agg_imp = self.get_aggregated_importance(method='mean')
            if agg_imp is not None:
                agg_imp.to_excel(os.path.join(results_dir, 'ml_feature_importance_aggregated.xlsx'), index=False)
            
            print(f"  Feature importance сохранён")
        
        for model_name, model in self.models.items():
            path = os.path.join(models_dir, f'ml_{model_name}.pkl')
            joblib.dump(model, path)
        
        joblib.dump(self.scaler, os.path.join(models_dir, 'ml_scaler.pkl'))

        meta = {
            'feature_cols': self.feature_cols,
            'config': self.config,
            'best_model': self.best_model,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances, 
            'base_features': self._get_base_features(),
            'created_at': datetime.now().isoformat()
        }
        joblib.dump(meta, os.path.join(models_dir, 'ml_meta.pkl'))
        
        print(f"  Результаты: {results_dir}")
        print(f"  Модели: {models_dir}")
        print(f"  Признаки для прогноза ({len(self.feature_cols)}): {self.feature_cols}")
        
        return self
    
    def _get_base_features(self):
        """Получить список базовых признаков (без лагов и MA)"""
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        if self.config['features']:
            return self.config['features']
        else:
            base = []
            for col in self.data.columns:
                if col in [date_col, target_col]:
                    continue
                if '_lag_' in col or '_ma_' in col:
                    continue
                base.append(col)
            return base
    
    def plot_results(self):
        """Визуализация результатов"""
        results_dir = self.config['results_dir']

        plt.figure(figsize=(14, 7))
        plt.plot(self.test_dates, self.y_test, 'b-', label='Факт', linewidth=2)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        for (name, res), color in zip(self.results.items(), colors):
            ls = '-' if name == self.best_model else '--'
            lw = 2.5 if name == self.best_model else 1.5
            plt.plot(self.test_dates, res['test_pred'], ls, label=name, 
                    color=color, linewidth=lw)
        
        plt.title('ML модели: сравнение прогнозов на test', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel(self.config['target_column'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ml_all_predictions.png'), dpi=150)
        plt.close()
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        models = list(self.metrics.keys())
        
        for ax, metric in zip(axes, ['mae', 'mape', 'r2']):
            values = [self.metrics[m][metric] for m in models]
            bars = ax.bar(models, values, alpha=0.7)
            best_idx = models.index(self.best_model)
            bars[best_idx].set_alpha(1.0)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_title(metric.upper())
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'ml_metrics_comparison.png'), dpi=150)
        plt.close()
        
        if self.feature_importances:
            self._plot_feature_importance()
        
        print(f"Графики сохранены в {results_dir}")
        return self
    
    def _plot_feature_importance(self):
        """Визуализация важности признаков"""
        results_dir = self.config['results_dir']
        
        best_imp = self.get_feature_importance(self.best_model)
        if best_imp is not None:
            top_n = min(15, len(best_imp))
            top_imp = best_imp.head(top_n)
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
            
            plt.barh(range(top_n), top_imp['importance_pct'].values[::-1], color=colors)
            plt.yticks(range(top_n), top_imp['feature'].values[::-1])
            plt.xlabel('Важность (%)')
            plt.title(f'Feature Importance: {self.best_model} (метод: {top_imp["method"].iloc[0]})', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            for i, v in enumerate(top_imp['importance_pct'].values[::-1]):
                plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'ml_feature_importance_{self.best_model}.png'), dpi=150)
            plt.close()
        
        if len(self.feature_importances) > 1:
            models = list(self.feature_importances.keys())
            features = self.feature_cols
            data = np.zeros((len(features), len(models)))
            for j, model_name in enumerate(models):
                imp_data = self.feature_importances[model_name]
                for i, feat in enumerate(features):
                    idx = list(imp_data['features']).index(feat)
                    data[i, j] = imp_data['normalized'][idx]
            
            mean_imp = data.mean(axis=1)
            sort_idx = np.argsort(mean_imp)[::-1]
        
            top_n = min(15, len(features))
            top_idx = sort_idx[:top_n]
            
            data_top = data[top_idx]
            features_top = [features[i] for i in top_idx]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(data_top, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_yticks(range(len(features_top)))
            ax.set_yticklabels(features_top)

            for i in range(len(features_top)):
                for j in range(len(models)):
                    text = ax.text(j, i, f'{data_top[i, j]:.1f}',
                                  ha='center', va='center', fontsize=8,
                                  color='white' if data_top[i, j] > 10 else 'black')
            
            plt.colorbar(im, label='Важность (%)')
            plt.title('Feature Importance: сравнение моделей (Top-15)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'ml_feature_importance_heatmap.png'), dpi=150)
            plt.close()
        
        agg_imp = self.get_aggregated_importance(method='mean')
        if agg_imp is not None:
            top_n = min(15, len(agg_imp))
            top_agg = agg_imp.head(top_n)
            
            plt.figure(figsize=(12, 8))
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, top_n))[::-1]
            
            plt.barh(range(top_n), top_agg['importance_mean'].values[::-1], color=colors)
            plt.yticks(range(top_n), top_agg['feature'].values[::-1])
            plt.xlabel('Средняя важность (%)')
            plt.title('Агрегированная важность признаков (среднее по всем моделям)', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            for i, v in enumerate(top_agg['importance_mean'].values[::-1]):
                plt.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'ml_feature_importance_aggregated.png'), dpi=150)
            plt.close()
    
    def summary(self):
        """Вывод итогов"""
        print("\n" + "="*60)
        print("ИТОГИ ML FORECASTING")
        print("="*60)
        
        print(f"\n{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
        print("-"*60)
        
        for model, m in sorted(self.metrics.items(), key=lambda x: x[1]['mae']):
            marker = '★' if model == self.best_model else ' '
            print(f"{marker}{model:<19} {m['mae']:<10.3f} {m['rmse']:<10.3f} "
                  f"{m['mape']:<10.2f} {m['r2']:<10.4f}")
        
        print("-"*60)
        print(f"\n🏆 Лучшая модель: {self.best_model}")
        print(f"   MAE: {self.metrics[self.best_model]['mae']:.3f}")
        
        return self
    
    
    @classmethod
    def load(cls, models_dir):
        """
        Загрузка сохраненной модели из директории.
        
        Parameters:
        -----------
        models_dir : str
            Путь к директории с сохраненными моделями
        
        Returns:
        --------
        MLForecaster с загруженными моделями
        
        Example:
        --------
        >>> forecaster = MLForecaster.load('saved_models/sugar')
        >>> predictions = forecaster.predict_from_file('data/future.xlsx')
        """
        instance = cls.__new__(cls)
        
        meta_path = os.path.join(models_dir, 'ml_meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Метаданные не найдены: {meta_path}")
        
        meta = joblib.load(meta_path)
        instance.config = meta['config']
        instance.feature_cols = meta['feature_cols']
        instance.best_model = meta['best_model']
        instance.metrics = meta.get('metrics', {})
        instance.base_features = meta.get('base_features', [])
        instance.feature_importances = meta.get('feature_importances', {})
        
        scaler_path = os.path.join(models_dir, 'ml_scaler.pkl')
        if os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
        else:
            instance.scaler = None
        
        instance.models = {}
        for model_name in cls.AVAILABLE_MODELS:
            model_path = os.path.join(models_dir, f'ml_{model_name}.pkl')
            if os.path.exists(model_path):
                instance.models[model_name] = joblib.load(model_path)
        
        instance.is_fitted = True
        instance.results = {}
        instance.data = None
        
        print(f"Модель загружена из {models_dir}")
        print(f"  Лучшая модель: {instance.best_model}")
        print(f"  Доступные модели: {list(instance.models.keys())}")
        print(f"  Признаки ({len(instance.feature_cols)}): {instance.feature_cols[:5]}...")
        
        return instance
    
    def load_model(self, models_dir=None):
        """
        Загрузка сохраненной модели в текущий экземпляр.
        
        Parameters:
        -----------
        models_dir : str, optional
            Путь к директории. Если не указан, берется из config['models_dir']
        """
        models_dir = models_dir or self.config['models_dir']

        meta_path = os.path.join(models_dir, 'ml_meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Метаданные не найдены: {meta_path}")
        
        meta = joblib.load(meta_path)
        self.feature_cols = meta['feature_cols']
        self.best_model = meta['best_model']
        self.metrics = meta.get('metrics', {})
        self.base_features = meta.get('base_features', [])
        self.feature_importances = meta.get('feature_importances', {})
        
        saved_config = meta['config']
        for key in ['lags', 'ma_windows', 'create_features', 'date_column', 'target_column']:
            if key in saved_config:
                self.config[key] = saved_config[key]
        
        scaler_path = os.path.join(models_dir, 'ml_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        self.models = {}
        for model_name in self.AVAILABLE_MODELS:
            model_path = os.path.join(models_dir, f'ml_{model_name}.pkl')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
        
        self.is_fitted = True
        
        print(f"Модель загружена из {models_dir}")
        print(f"  Лучшая модель: {self.best_model}")
        print(f"  Признаки: {len(self.feature_cols)}")
        
        return self
    
    def predict_from_file(self, filepath=None, model_name=None, save_results=True):
        """
        Прогноз на новых данных из файла.
        
        Parameters:
        -----------
        filepath : str, optional
            Путь к Excel файлу с признаками.
            Если не указан, берется из config['forecast_file']
        model_name : str, optional
            Какую модель использовать. По умолчанию best_model
        save_results : bool
            Сохранять ли результаты в Excel
        
        Returns:
        --------
        pd.DataFrame с прогнозами
        
        Example:
        --------
        >>> forecaster = MLForecaster.load('saved_models/sugar')
        >>> predictions = forecaster.predict_from_file('data/future_features.xlsx')
        >>> print(predictions)
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена и не загружена. Используйте fit() или load()")
        
        filepath = filepath or self.config.get('forecast_file')
        if not filepath:
            raise ValueError("Укажите путь к файлу с данными для прогноза")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        model_name = model_name or self.best_model
        if model_name not in self.models:
            raise ValueError(f"Модель '{model_name}' не найдена. Доступны: {list(self.models.keys())}")
        
        print(f"\n{'='*60}")
        print(f"ПРОГНОЗ НА НОВЫХ ДАННЫХ")
        print(f"{'='*60}")
        print(f"Файл: {filepath}")
        print(f"Модель: {model_name}")
        
        df = pd.read_excel(filepath)
        print(f"Загружено строк: {len(df)}")
        
        X_forecast, dates = self._prepare_forecast_features(df)
        model = self.models[model_name]
        predictions = model.predict(X_forecast)
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        result_df = pd.DataFrame({
            date_col: dates,
            f'{target_col}_predicted': predictions
        })

        for name, mdl in self.models.items():
            if name != model_name:
                result_df[f'{target_col}_pred_{name}'] = mdl.predict(X_forecast)
        
        print(f"\nПрогноз ({len(predictions)} значений):")
        print(result_df.head(10).to_string(index=False))
        
        if save_results:
            results_dir = self.config['results_dir']
            os.makedirs(results_dir, exist_ok=True)
            
            output_path = os.path.join(results_dir, 'ml_forecast_results.xlsx')
            result_df.to_excel(output_path, index=False)
            print(f"\nРезультаты сохранены: {output_path}")
            
            self._plot_forecast(result_df, model_name)
        
        return result_df
    
    def _prepare_forecast_features(self, df):
        """
        Подготовка признаков для прогноза.
        
        Генерирует лаги и MA если они были использованы при обучении.
        """
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            dates = df[date_col].values
        else:
            dates = np.arange(len(df))
        
        base_features = getattr(self, 'base_features', [])
        if not base_features:
            base_features = [col for col in self.feature_cols 
                           if '_lag_' not in col and '_ma_' not in col]
        
        missing = [f for f in base_features if f not in df.columns]
        if missing:
            raise ValueError(f"В файле отсутствуют признаки: {missing}")
        
        if self.config.get('create_features', False):
            print("Генерация лагов и скользящих средних...")
            
            for col in base_features:
                for lag in self.config.get('lags', []):
                    col_name = f"{col}_lag_{lag}"
                    if col_name in self.feature_cols:
                        df[col_name] = df[col].shift(lag)

                for window in self.config.get('ma_windows', []):
                    col_name = f"{col}_ma_{window}"
                    if col_name in self.feature_cols:
                        df[col_name] = df[col].rolling(window=window).mean()
        
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        if missing_features:
            print(f"ВНИМАНИЕ: Не все признаки найдены. Отсутствуют: {missing_features}")
            for f in missing_features:
                df[f] = 0
        
        df_clean = df.dropna(subset=[f for f in self.feature_cols if f in df.columns])
        
        if len(df_clean) == 0:
            raise ValueError("После удаления NaN не осталось данных. "
                           "Возможно, недостаточно исторических данных для генерации лагов.")
        
        if len(df_clean) < len(df):
            print(f"  Удалено строк с NaN: {len(df) - len(df_clean)}")
            dates = df_clean[date_col].values if date_col in df_clean.columns else dates[:len(df_clean)]
        
        X = df_clean[self.feature_cols].values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        print(f"  Подготовлено признаков: {X.shape[1]}")
        print(f"  Строк для прогноза: {X.shape[0]}")
        
        return X, dates
    
    def _plot_forecast(self, result_df, model_name):
        """График прогноза"""
        results_dir = self.config['results_dir']
        date_col = self.config['date_column']
        target_col = self.config['target_column']
        
        plt.figure(figsize=(14, 6))
    
        pred_col = f'{target_col}_predicted'
        plt.plot(result_df[date_col], result_df[pred_col], 
                'b-', linewidth=2, marker='o', markersize=4,
                label=f'Прогноз ({model_name})')
        
        plt.title(f'Прогноз {target_col} (модель: {model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(results_dir, 'ml_forecast_plot.png'), dpi=150)
        plt.close()
        
        print(f"График сохранен: {results_dir}/ml_forecast_plot.png")