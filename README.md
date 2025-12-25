# Commodity Forecasting Library

Библиотека для прогнозирования цен на сырьевые товары с использованием машинного обучения и фильтра Калмана.

## Возможности

### 1. ML Forecaster

Машинное обучение с автоматической оптимизацией гиперпараметров (Optuna):

-   **CatBoost**, **XGBoost**, **LightGBM** - градиентный бустинг
-   **Ridge**, **Lasso** - линейные модели с регуляризацией
-   **KNN**, **SVR** - нелинейные модели
-   **Random Forest**, **AdaBoost** - ансамбли

Автоматическое создание признаков:

-   Лаги (по умолчанию: 3, 6, 9, 12, 18, 24)
-   Скользящие средние (по умолчанию: 3, 6, 9, 12)
-   Отбор признаков по корреляции или экспертный выбор

### 2. Kalman Forecaster

Time-Varying Kalman Filter с 7 методами прогнозирования коэффициентов β(t):

-   `last_beta` - последний β постоянен
-   `random_walk` - random walk с неопределенностью
-   `linear_trend` - линейная экстраполяция
-   `ar_beta` - AR модель для каждого β
-   `damped_trend` - затухающий тренд (mean-reverting)
-   `bayesian_shrinkage` - байесовское сжатие к историческому среднему
-   `adaptive_ensemble` - адаптивный ансамбль

## Структура проекта

```
commodity_forecasting/├── forecasting/              # Библиотека│   ├── __init__.py│   └── models/│       ├── __init__.py│       ├── ml_forecaster.py      # ML модели│       └── kalman_forecaster.py  # Kalman Filter├── configs/                  # Конфигурации для датасетов│   ├── sugar.yaml│   ├── wheat.yaml│   ├── milk.yaml│   ├── gold.yaml│   └── soybean.yaml├── data/                     # Данные (положить сюда xlsx файлы)├── notebooks/                # Jupyter ноутбуки├── results/                  # Результаты (создается автоматически)├── saved_models/             # Сохраненные модели├── run.py                    # Главный скрипт запуска├── requirements.txt├── setup.py└── README.md
```

## Установка

```bash
# Клонирование репозиторияgit clone https://github.com/your-username/commodity-forecasting.gitcd commodity-forecasting# Установка зависимостейpip install -r requirements.txt# Или установка как пакетаpip install -e .
```

## Использование

### Командная строка

```bash
# Запуск ML модели для сахараpython run.py --config configs/sugar.yaml --model ml# Запуск Kalman модели для пшеницыpython run.py --commodity wheat --model kalman# Запуск обеих моделей для соиpython run.py --commodity soybean --model all# Запуск всех датасетовpython run.py --all
```

### Прогноз на новых данных (Inference)

После обучения модели можно делать прогнозы на новых данных:

```bash
# === ML модель ===# Прогноз через конфигpython run.py --predict --config configs/sugar.yaml --forecast-file data/future_features.xlsx# Прогноз напрямую из директории с моделямиpython run.py --predict --models-dir saved_models/sugar -f data/future_features.xlsx# Использовать конкретную модель (не лучшую)python run.py --predict -c configs/sugar.yaml -f data/future.xlsx --model-name xgb# === Kalman модель ===# Прогноз через конфигpython run.py --predict --predict-model kalman --config configs/sugar.yaml -f data/future.xlsx# Прогноз из директорииpython run.py --predict --predict-model kalman --models-dir saved_models/sugar -f data/future.xlsx# Использовать конкретный метод (не лучший)python run.py --predict --predict-model kalman -c configs/sugar.yaml -f data/future.xlsx --model-name bayesian_shrinkage
```

### Python API

```python
from forecasting import MLForecaster, KalmanForecaster# === ML Модель - Обучение ===ml = MLForecaster(config='configs/sugar.yaml')ml.fit()ml.save_results()ml.plot_results()ml.summary()print(f"Лучшая модель: {ml.best_model}")print(f"MAE: {ml.metrics[ml.best_model]['mae']:.3f}")# === ML Модель - Прогноз на новых данных ===# Вариант 1: Загрузка модели из директорииforecaster = MLForecaster.load('saved_models/sugar')predictions = forecaster.predict_from_file('data/future_features.xlsx')# Вариант 2: Через конфигforecaster = MLForecaster(config='configs/sugar.yaml')forecaster.load_model()  # загружает из models_dir в конфигеpredictions = forecaster.predict_from_file('data/future_features.xlsx')# Вариант 3: Использовать конкретную модельpredictions = forecaster.predict_from_file('data/future.xlsx', model_name='catboost')# Результат - DataFrame с прогнозамиprint(predictions)# === Kalman Модель - Обучение ===kf = KalmanForecaster(config='configs/sugar.yaml')kf.fit()kf.save_results()kf.plot_results()kf.summary()print(f"Лучший метод: {kf.best_method}")print(f"MAE: {kf.metrics[kf.best_method]['mae']:.3f}")# === Kalman Модель - Прогноз на новых данных ===# Вариант 1: Загрузка модели из директорииforecaster = KalmanForecaster.load('saved_models/sugar')predictions = forecaster.predict_from_file('data/future_features.xlsx')# Вариант 2: Через конфигforecaster = KalmanForecaster(config='configs/sugar.yaml')forecaster.load_model()predictions = forecaster.predict_from_file('data/future_features.xlsx')# Вариант 3: Использовать конкретный методpredictions = forecaster.predict_from_file('data/future.xlsx', method='bayesian_shrinkage')# Результат включает прогноз цены и коэффициенты βprint(predictions)# Статистика по βbeta_stats = forecaster.get_beta_stats()print(beta_stats)
```

### Конфигурация без YAML

```python
from forecasting import MLForecasterml = MLForecaster(    input_file='data/my_data.xlsx',    target_column='Price',    date_column='Date',    features=['Feature1', 'Feature2', 'Feature3'],    models_to_train=['catboost', 'xgb', 'lgb'],    test_size_ratio=0.1,    n_optuna_trials=50)ml.fit()
```

## Формат данных

### Файл для обучения

Excel файл должен содержать:

-   Колонку с датой
-   Колонку с целевой переменной (цена)
-   Колонки с признаками (факторы-драйверы)

Пример:

Date

Price

Brent

DXY

Stock-to-Use

2020-01-01

100.5

65.2

97.3

0.25

2020-02-01

102.3

66.1

98.1

0.24

...

...

...

...

...

### Файл для прогноза (inference)

Excel файл с **теми же признаками**, что использовались при обучении, **без целевой переменной**:

Date

Brent

DXY

Stock-to-Use

2025-01-01

75.0

104.5

0.22

2025-02-01

76.2

105.1

0.21

...

...

...

...

**Важно:**

-   Признаки должны совпадать с теми, что были при обучении
-   Если модель использовала лаги/MA, они будут сгенерированы автоматически
-   Первые N строк могут быть пропущены из-за генерации лагов (где N = максимальный лаг)

## Конфигурация (YAML)

```yaml
# === ДАННЫЕ ===input_file: 'data/sugar_factors.xlsx'date_column: 'Date'target_column: 'Sugar Price'# === ФАЙЛ ДЛЯ ПРОГНОЗА ===# Путь к файлу с признаками для прогноза (опционально)forecast_file: 'data/sugar_forecast_features.xlsx'# === ПРИЗНАКИ ===features: []  # пусто = все колонки кроме даты и таргетаexpert_features:  # экспертный выбор (приоритет над автоматическим)  - 'Stock-to-Use'  - 'Brent'  - 'DXY'# === ML МОДЕЛЬ ===max_features: 10correlation_threshold: 0.3create_features: truelags: [3, 6, 9, 12, 18, 24]ma_windows: [3, 6, 9, 12]models_to_train:  - catboost  - xgb  - lgbn_optuna_trials: 100# === KALMAN МОДЕЛЬ ===initial_state_covariance: 100.0observation_covariance: 1.0transition_covariance: 0.01forecast_methods:  - bayesian_shrinkage  - adaptive_ensemble# === РАЗДЕЛЕНИЕ ===test_size_ratio: 0.1val_size_ratio: 0.15# === ПУТИ ===results_dir: 'results/sugar'models_dir: 'saved_models/sugar'
```

## Выходные файлы

### ML Forecaster

-   `results/*/ml_metrics.xlsx` - метрики моделей
-   `results/*/ml_predictions.xlsx` - прогнозы
-   `results/*/ml_all_predictions.png` - график прогнозов
-   `results/*/ml_metrics_comparison.png` - сравнение метрик
-   `saved_models/*/ml_*.pkl` - сохраненные модели

### Kalman Forecaster

-   `results/*/kalman_metrics.xlsx` - метрики методов
-   `results/*/kalman_predictions.xlsx` - прогнозы
-   `results/*/kalman_betas_train.xlsx` - коэффициенты β
-   `results/*/kalman_all_forecasts.png` - график прогнозов
-   `results/*/kalman_beta_evolution.png` - эволюция β(t)
-   `saved_models/*/kalman_model.pkl` - модель

## Метрики

-   **MAE** - Mean Absolute Error
-   **RMSE** - Root Mean Square Error
-   **MAPE** - Mean Absolute Percentage Error (%)
-   **R²** - Coefficient of Determination

## Поддерживаемые датасеты

Датасет

Конфиг

Описание

Sugar

`configs/sugar.yaml`

Цены на сахар

Wheat

`configs/wheat.yaml`

Цены на пшеницу (Matif)

Milk

`configs/milk.yaml`

Цены на молоко

Gold

`configs/gold.yaml`

Цены на золото

Soybean

`configs/soybean.yaml`

Цены на сою

## Добавление нового датасета

1.  Создайте YAML конфиг в `configs/`:

```yaml
input_file: 'data/my_commodity.xlsx'target_column: 'My Price'# ... остальные параметры
```

2.  Положите данные в `data/my_commodity.xlsx`
    
3.  Запустите:
    

```bash
python run.py --config configs/my_commodity.yaml --model all
```

## Логирование

Все запуски автоматически логируются в директорию `logs/`:

```bash
# Стандартный запуск - логи в logs/python run.py --all# Указать другую директориюpython run.py --all --log-dir my_logs# Отключить логирование в файлpython run.py --all --no-log
```

### Содержимое лог-файла

Лог-файл содержит:

-   Время запуска
-   Все выводимые сообщения
-   Время выполнения каждой задачи
-   Итоговую сводку по времени

Пример имени файла: `logs/all_commodities_20250101_143022.log`

### Использование логгера в Python

```python
from forecasting import Logger, get_logger# Создание логгераlogger = Logger(log_dir='logs', name='my_experiment')# Сообщенияlogger.info("Информация")logger.warning("Предупреждение")logger.error("Ошибка")logger.success("Успех")# Заголовкиlogger.section("ГЛАВНЫЙ РАЗДЕЛ")logger.subsection("Подраздел")# Замер времениwith logger.timer("Обучение модели"):    model.fit()# Сводка по времениlogger.summary()
```

## Лицензия

MIT License

## Авторы

Aleksey Kirichenko, Anastasia Belotserkovets