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
        self.scaler = None
        self.feature_cols = None
        self.data = None
        self.is_fitted = False
        
        # Установка random seed
        self._set_seed()
        
        # Создание директорий
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
        
        # Значения по умолчанию
        defaults = {
            'input_file': 'data/input_data.xlsx',
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
        
        # Базовые признаки
        if self.config['features']:
            base_cols = self.config['features']
        else:
            base_cols = [col for col in self.data.columns 
                        if col not in [date_col, target_col]]
        
        print(f"Базовые признаки ({len(base_cols)}): {base_cols}")
        
        # Создание лагов и MA
        if self.config['create_features']:
            print(f"Создание лагов: {self.config['lags']}")
            for col in base_cols:
                for lag in self.config['lags']:
                    self.data[f"{col}_lag_{lag}"] = self.data[col].shift(lag)
            
            print(f"Создание MA: {self.config['ma_windows']}")
            for col in base_cols:
                for window in self.config['ma_windows']:
                    self.data[f"{col}_ma_{window}"] = self.data[col].rolling(window=window).mean()
        
        # Все признаки
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
        
        # Данные для отбора (без NaN)
        df_valid = self.data.dropna(subset=all_features + [target_col])
        X = df_valid[all_features]
        y = df_valid[target_col]
        
        if self.config['expert_features']:
            # Экспертный выбор
            print("Режим: ЭКСПЕРТНЫЙ ВЫБОР")
            missing = [f for f in self.config['expert_features'] if f not in all_features]
            if missing:
                raise ValueError(f"Признаки не найдены: {missing}")
            
            selected = self.config['expert_features'][:self.config['max_features']]
        else:
            # Автоматический отбор по корреляции
            print("Режим: АВТОМАТИЧЕСКИЙ ОТБОР")
            correlations = {}
            for col in all_features:
                valid_mask = X[col].notna() & y.notna()
                if valid_mask.sum() > 0:
                    correlations[col] = X.loc[valid_mask, col].corr(y.loc[valid_mask])
            
            corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
            corr_df['abs_correlation'] = corr_df['correlation'].abs()
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            
            # Отбор по порогу
            threshold = self.config['correlation_threshold']
            selected = corr_df[corr_df['abs_correlation'] >= threshold].index.tolist()
            
            # Ограничение количества
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
        
        # Удаляем строки с NaN в признаках
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
        
        # Нормализация
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
                
                # Предсказания
                train_pred = model.predict(self.X_train)
                val_pred = model.predict(self.X_val)
                test_pred = model.predict(self.X_test)
                
                self.results[model_name] = {
                    'train_pred': train_pred,
                    'val_pred': val_pred,
                    'test_pred': test_pred
                }
                
                # Метрики
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
        print(f"\n🏆 Лучшая модель: {self.best_model} (MAE = {self.metrics[self.best_model]['mae']:.4f})")
        
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
    
    def predict(self, X, model_name=None):
        """Прогноз на новых данных"""
        model_name = model_name or self.best_model
        X_scaled = self.scaler.transform(X) if not isinstance(X, np.ndarray) else X
        return self.models[model_name].predict(X_scaled)
    
    def save_results(self):
        """Сохранение результатов"""
        results_dir = self.config['results_dir']
        models_dir = self.config['models_dir']
        
        print("\nСохранение результатов...")
        
        # Метрики
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = 'Model'
        metrics_df = metrics_df.sort_values('mae')
        metrics_df.to_excel(os.path.join(results_dir, 'ml_metrics.xlsx'))
        
        # Прогнозы
        preds_df = pd.DataFrame({
            'Date': self.test_dates,
            'Actual': self.y_test
        })
        for model_name, res in self.results.items():
            preds_df[f'{model_name}_pred'] = res['test_pred']
        preds_df.to_excel(os.path.join(results_dir, 'ml_predictions.xlsx'), index=False)
        
        # Модели
        for model_name, model in self.models.items():
            path = os.path.join(models_dir, f'ml_{model_name}.pkl')
            joblib.dump(model, path)
        
        # Scaler и config
        joblib.dump(self.scaler, os.path.join(models_dir, 'ml_scaler.pkl'))
        joblib.dump({
            'feature_cols': self.feature_cols,
            'config': self.config,
            'best_model': self.best_model
        }, os.path.join(models_dir, 'ml_meta.pkl'))
        
        print(f"  Результаты: {results_dir}")
        print(f"  Модели: {models_dir}")
        
        return self
    
    def plot_results(self):
        """Визуализация результатов"""
        results_dir = self.config['results_dir']
        
        # 1. Все прогнозы
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
        
        # 2. Сравнение метрик
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
        
        print(f"Графики сохранены в {results_dir}")
        return self
    
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
