# Commodity Forecasting Library

Библиотека для прогнозирования цен на штз с использованием машинного обучения и фильтра Калмана.

## Возможности

### 1. ML Forecaster
Машинное обучение с автоматической оптимизацией гиперпараметров (Optuna):
- **CatBoost**, **XGBoost**, **LightGBM** - градиентный бустинг
- **Ridge**, **Lasso** - линейные модели с регуляризацией
- **KNN**, **SVR** - нелинейные модели
- **Random Forest**, **AdaBoost** - ансамбли

Автоматическое создание признаков:
- Лаги (по умолчанию: 3, 6, 9, 12, 18, 24)
- Скользящие средние (по умолчанию: 3, 6, 9, 12)
- Отбор признаков по корреляции или экспертный выбор

### 2. Kalman Forecaster
Time-Varying Kalman Filter с 7 методами прогнозирования коэффициентов β(t):
- `last_beta` - последний β постоянен
- `random_walk` - random walk с неопределенностью
- `linear_trend` - линейная экстраполяция
- `ar_beta` - AR модель для каждого β
- `damped_trend` - затухающий тренд (mean-reverting)
- `bayesian_shrinkage` - байесовское сжатие к историческому среднему
- `adaptive_ensemble` - адаптивный ансамбль

## Структура проекта

```
commodity_forecasting/
├── forecasting/              # Библиотека
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── ml_forecaster.py      # ML модели
│       └── kalman_forecaster.py  # Kalman Filter
├── configs/                  # Конфигурации для датасетов
│   ├── sugar.yaml
│   ├── wheat.yaml
│   ├── milk.yaml
│   ├── gold.yaml
│   └── soybean.yaml
├── data/                     # Данные (положить сюда xlsx файлы)
├── notebooks/                # Jupyter ноутбуки
├── results/                  # Результаты (создается автоматически)
├── saved_models/             # Сохраненные модели
├── run.py                    # Главный скрипт запуска
├── requirements.txt
├── setup.py
└── README.md
```

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/commodity-forecasting.git
cd commodity-forecasting

# Установка зависимостей
pip install -r requirements.txt

# Или установка как пакета
pip install -e .
```

## Использование

### Командная строка

```bash
# Запуск ML модели для сахара
python run.py --config configs/sugar.yaml --model ml

# Запуск Kalman модели для пшеницы
python run.py --commodity wheat --model kalman

# Запуск обеих моделей для сои
python run.py --commodity soybean --model all

# Запуск всех датасетов
python run.py --all
```

### Python API

```python
from forecasting import MLForecaster, KalmanForecaster

# === ML Модель ===
ml = MLForecaster(config='configs/sugar.yaml')
ml.fit()
ml.save_results()
ml.plot_results()
ml.summary()

print(f"Лучшая модель: {ml.best_model}")
print(f"MAE: {ml.metrics[ml.best_model]['mae']:.3f}")

# === Kalman Модель ===
kf = KalmanForecaster(config='configs/sugar.yaml')
kf.fit()
kf.save_results()
kf.plot_results()
kf.summary()

print(f"Лучший метод: {kf.best_method}")
print(f"MAE: {kf.metrics[kf.best_method]['mae']:.3f}")
```

### Конфигурация без YAML

```python
from forecasting import MLForecaster

ml = MLForecaster(
    input_file='data/my_data.xlsx',
    target_column='Price',
    date_column='Date',
    features=['Feature1', 'Feature2', 'Feature3'],
    models_to_train=['catboost', 'xgb', 'lgb'],
    test_size_ratio=0.1,
    n_optuna_trials=50
)
ml.fit()
```

## Формат данных

Входной Excel файл должен содержать:
- Колонку с датой (по дефольту название dt)
- Колонку с целевой переменной (цена, по дефолту называется price)
- Колонки с признаками (факторы-драйверы)

Пример:
| Date       | Price  | Brent  | DXY    | Stock-to-Use |
|------------|--------|--------|--------|--------------|
| 2020-01-01 | 100.5  | 65.2   | 97.3   | 0.25         |
| 2020-02-01 | 102.3  | 66.1   | 98.1   | 0.24         |
| ...        | ...    | ...    | ...    | ...          |

## Конфигурация (YAML)

```yaml
# === ДАННЫЕ ===
input_file: 'data/sugar_factors.xlsx'
date_column: 'Date'
target_column: 'Sugar Price'

# === ПРИЗНАКИ ===
features: []  # пусто = все колонки кроме даты и таргета
expert_features:  # экспертный выбор (приоритет над автоматическим)
  - 'Stock-to-Use'
  - 'Brent'
  - 'DXY'

# === ML МОДЕЛЬ ===
max_features: 10
correlation_threshold: 0.3
create_features: true
lags: [3, 6, 9, 12, 18, 24]
ma_windows: [3, 6, 9, 12]
models_to_train:
  - catboost
  - xgb
  - lgb
n_optuna_trials: 100

# === KALMAN МОДЕЛЬ ===
initial_state_covariance: 100.0
observation_covariance: 1.0
transition_covariance: 0.01
forecast_methods:
  - bayesian_shrinkage
  - adaptive_ensemble

# === РАЗДЕЛЕНИЕ ===
test_size_ratio: 0.1
val_size_ratio: 0.15

# === ПУТИ ===
results_dir: 'results/sugar'
models_dir: 'saved_models/sugar'
```

## Выходные файлы

### ML Forecaster
- `results/*/ml_metrics.xlsx` - метрики моделей
- `results/*/ml_predictions.xlsx` - прогнозы
- `results/*/ml_all_predictions.png` - график прогнозов
- `results/*/ml_metrics_comparison.png` - сравнение метрик
- `saved_models/*/ml_*.pkl` - сохраненные модели

### Kalman Forecaster
- `results/*/kalman_metrics.xlsx` - метрики методов
- `results/*/kalman_predictions.xlsx` - прогнозы
- `results/*/kalman_betas_train.xlsx` - коэффициенты β
- `results/*/kalman_all_forecasts.png` - график прогнозов
- `results/*/kalman_beta_evolution.png` - эволюция β(t)
- `saved_models/*/kalman_model.pkl` - модель

## Метрики

- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Square Error
- **MAPE** - Mean Absolute Percentage Error (%)
- **R²** - Coefficient of Determination

## Поддерживаемые датасеты

| Датасет | Конфиг | Описание |
|---------|--------|----------|
| Sugar   | `configs/sugar.yaml` | Цены на сахар |
| Wheat   | `configs/wheat.yaml` | Цены на пшеницу (Matif) |
| Milk    | `configs/milk.yaml` | Цены на молоко |
| Gold    | `configs/gold.yaml` | Цены на золото |
| Soybean | `configs/soybean.yaml` | Цены на сою |

## Добавление нового датасета

1. Создайте YAML конфиг в `configs/`:
```yaml
input_file: 'data/my_commodity.xlsx'
target_column: 'My Price'
# ... остальные параметры
```

2. Положите данные в `data/my_commodity.xlsx`

3. Запустите:
```bash
python run.py --config configs/my_commodity.yaml --model all
```

## Лицензия

MIT License

## Авторы

Aleksey Kirichenko, Anastasia Belotserkovets
