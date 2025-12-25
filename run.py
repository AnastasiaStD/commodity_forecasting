#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commodity Forecasting - Главный скрипт запуска
==============================================

Использование:
    # Обучение моделей
    python run.py --config configs/sugar.yaml --model ml
    python run.py --config configs/sugar.yaml --model kalman
    python run.py --config configs/sugar.yaml --model all
    python run.py --commodity sugar --model all
    python run.py --all  # запустить все датасеты
    
    # Прогноз на новых данных (ML)
    python run.py --predict --config configs/sugar.yaml --forecast-file data/future.xlsx
    python run.py --predict --models-dir saved_models/sugar --forecast-file data/future.xlsx
    
    # Прогноз на новых данных (Kalman)
    python run.py --predict --predict-model kalman --config configs/sugar.yaml -f data/future.xlsx
    python run.py --predict --predict-model kalman --models-dir saved_models/sugar -f data/future.xlsx

Примеры:
    python run.py -c configs/wheat.yaml -m ml
    python run.py --commodity soybean --model kalman
    python run.py --all --model ml
    python run.py --predict -c configs/sugar.yaml -f data/sugar_future.xlsx
    python run.py --predict --predict-model kalman -c configs/sugar.yaml -f data/future.xlsx
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from forecasting import MLForecaster, KalmanForecaster, Logger, get_logger


COMMODITIES = ['sugar', 'wheat', 'milk', 'soybean', 'palm_oil', 'sf_oil']


def run_ml(config_path):
    """Запуск ML модели"""
    logger = get_logger()
    
    logger.section("ML FORECASTER")
    
    with logger.timer(f"ML Training ({os.path.basename(config_path)})"):
        forecaster = MLForecaster(config=config_path)
        forecaster.fit()
        forecaster.save_results()
        forecaster.plot_results()
        forecaster.summary()
    
    return forecaster


def run_kalman(config_path):
    """Запуск Kalman модели"""
    logger = get_logger()
    
    logger.section("KALMAN FORECASTER")
    
    with logger.timer(f"Kalman Training ({os.path.basename(config_path)})"):
        forecaster = KalmanForecaster(config=config_path)
        forecaster.fit()
        forecaster.save_results()
        forecaster.plot_results()
        forecaster.summary()
    
    return forecaster


def run_predict(config_path=None, models_dir=None, forecast_file=None, model_name=None, model_type='ml'):
    """Запуск прогноза на новых данных"""
    logger = get_logger()
    
    logger.section(f"ПРОГНОЗ НА НОВЫХ ДАННЫХ ({model_type.upper()})")
    
    with logger.timer(f"Prediction ({model_type})"):
        if model_type == 'ml':
            if config_path:
                forecaster = MLForecaster(config=config_path)
                forecaster.load_model()
            elif models_dir:
                forecaster = MLForecaster.load(models_dir)
            else:
                logger.error("Укажите --config или --models-dir")
                return None
            
            if forecast_file:
                result = forecaster.predict_from_file(forecast_file, model_name=model_name)
            else:
                result = forecaster.predict_from_file(model_name=model_name)
        
        elif model_type == 'kalman':
            if config_path:
                forecaster = KalmanForecaster(config=config_path)
                forecaster.load_model()
            elif models_dir:
                forecaster = KalmanForecaster.load(models_dir)
            else:
                logger.error("Укажите --config или --models-dir")
                return None
            
            if forecast_file:
                result = forecaster.predict_from_file(forecast_file, method=model_name)
            else:
                result = forecaster.predict_from_file(method=model_name)
        
        else:
            logger.error(f"Неизвестный тип модели: {model_type}")
            return None
    
    return result


def run_commodity(commodity, model='all'):
    """Запуск для одного датасета"""
    logger = get_logger()
    config_path = f'configs/{commodity}.yaml'
    
    if not os.path.exists(config_path):
        logger.error(f"Конфиг не найден: {config_path}")
        return None, None
    
    logger.section(f"ДАТАСЕТ: {commodity.upper()}", char='#')
    
    ml_result = None
    kalman_result = None
    
    if model in ['ml', 'all']:
        try:
            ml_result = run_ml(config_path)
        except Exception as e:
            logger.error(f"ML ошибка: {e}")
    
    if model in ['kalman', 'all']:
        try:
            kalman_result = run_kalman(config_path)
        except Exception as e:
            logger.error(f"Kalman ошибка: {e}")
    
    return ml_result, kalman_result


def run_all(model='all'):
    """Запуск для всех датасетов"""
    logger = get_logger()
    results = {}
    
    logger.section("ЗАПУСК ВСЕХ ДАТАСЕТОВ", char='#')
    logger.print(f"Датасеты: {COMMODITIES}")
    logger.print(f"Модели: {model}")
    
    with logger.timer("Все датасеты"):
        for commodity in COMMODITIES:
            config_path = f'configs/{commodity}.yaml'
            if os.path.exists(config_path):
                ml, kalman = run_commodity(commodity, model)
                results[commodity] = {'ml': ml, 'kalman': kalman}
            else:
                logger.warning(f"Пропуск {commodity}: конфиг не найден")
    
    logger.section("СВОДКА ПО ВСЕМ ДАТАСЕТАМ")
    
    logger.print(f"\n{'Commodity':<15} {'ML Best':<15} {'ML MAE':<10} {'Kalman Best':<20} {'Kalman MAE':<10}")
    logger.print("-"*70)
    
    for commodity, res in results.items():
        ml_best = res['ml'].best_model if res['ml'] else '-'
        ml_mae = f"{res['ml'].metrics[res['ml'].best_model]['mae']:.3f}" if res['ml'] else '-'
        kf_best = res['kalman'].best_method if res['kalman'] else '-'
        kf_mae = f"{res['kalman'].metrics[res['kalman'].best_method]['mae']:.3f}" if res['kalman'] else '-'
        
        logger.print(f"{commodity:<15} {ml_best:<15} {ml_mae:<10} {kf_best:<20} {kf_mae:<10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Commodity Forecasting - Прогнозирование цен на сырьевые товары',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Примеры:
                # Обучение
                python run.py --config configs/sugar.yaml --model ml
                python run.py --commodity wheat --model kalman
                python run.py --all --model all
                
                # Прогноз на новых данных
                python run.py --predict --config configs/sugar.yaml --forecast-file data/future.xlsx
                python run.py --predict --models-dir saved_models/sugar -f data/future.xlsx
                
                # С логированием в другую директорию
                python run.py --all --log-dir my_logs
        """
    )
    
    parser.add_argument('-c', '--config', type=str, help='Путь к YAML конфигу')
    parser.add_argument('--commodity', type=str, choices=COMMODITIES, 
                       help='Название датасета (sugar, wheat, milk, gold, soybean)')
    parser.add_argument('-m', '--model', type=str, default='all',
                       choices=['ml', 'kalman', 'all'],
                       help='Тип модели: ml, kalman, all (default: all)')
    parser.add_argument('--all', action='store_true', 
                       help='Запустить все датасеты')
    
    parser.add_argument('--predict', action='store_true',
                       help='Режим прогноза на новых данных')
    parser.add_argument('-f', '--forecast-file', type=str,
                       help='Путь к файлу с признаками для прогноза')
    parser.add_argument('--models-dir', type=str,
                       help='Директория с сохраненными моделями (для --predict)')
    parser.add_argument('--model-name', type=str,
                       help='Название модели/метода для прогноза (по умолчанию лучшая)')
    parser.add_argument('--predict-model', type=str, default='ml',
                       choices=['ml', 'kalman'],
                       help='Тип модели для прогноза: ml или kalman (default: ml)')
    
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Директория для лог-файлов (default: logs)')
    parser.add_argument('--no-log', action='store_true',
                       help='Отключить запись в лог-файл')
    
    args = parser.parse_args()
    
    if args.predict:
        if not args.config and not args.models_dir:
            parser.print_help()
            print("\nОШИБКА: Для прогноза укажите --config или --models-dir")
            sys.exit(1)
    elif not any([args.config, args.commodity, args.all]):
        parser.print_help()
        print("\nОШИБКА: Укажите --config, --commodity, --all или --predict")
        sys.exit(1)
    
    if args.all:
        log_name = 'all_commodities'
    elif args.commodity:
        log_name = args.commodity
    elif args.config:
        log_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        log_name = 'run'
    
    if args.predict:
        log_name = f'predict_{log_name}'
    
    logger = Logger(
        log_dir=args.log_dir,
        name=log_name,
        console=True,
        file=not args.no_log
    )
    
    logger.section("COMMODITY FORECASTING LIBRARY")
    logger.print(f"Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.print(f"Аргументы: {' '.join(sys.argv[1:])}")
    
    try:
        if args.predict:
            run_predict(
                config_path=args.config,
                models_dir=args.models_dir,
                forecast_file=args.forecast_file,
                model_name=args.model_name,
                model_type=args.predict_model
            )
        elif args.all:
            run_all(args.model)
        elif args.config:
            if args.model == 'ml':
                run_ml(args.config)
            elif args.model == 'kalman':
                run_kalman(args.config)
            else:
                run_ml(args.config)
                run_kalman(args.config)
        elif args.commodity:
            run_commodity(args.commodity, args.model)
        
        logger.summary()
        logger.success("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    finally:
        logger.close()


if __name__ == '__main__':
    main()
