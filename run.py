#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commodity Forecasting - Главный скрипт запуска
==============================================

Использование:
    python run.py --config configs/sugar.yaml --model ml
    python run.py --config configs/sugar.yaml --model kalman
    python run.py --config configs/sugar.yaml --model all
    python run.py --commodity sugar --model all
    python run.py --all  # запустить все датасеты

Примеры:
    python run.py -c configs/wheat.yaml -m ml
    python run.py --commodity soybean --model kalman
    python run.py --all --model ml
"""

import argparse
import os
import sys
from pathlib import Path

# Добавляем путь к библиотеке
sys.path.insert(0, str(Path(__file__).parent))

from forecasting import MLForecaster, KalmanForecaster


# Доступные датасеты
COMMODITIES = ['sugar', 'wheat', 'milk', 'palm_oil', 'sf_oil', 'soybean']


def run_ml(config_path):
    """Запуск ML модели"""
    print("\n" + "="*70)
    print("ML FORECASTER")
    print("="*70)
    
    forecaster = MLForecaster(config=config_path)
    forecaster.fit()
    forecaster.save_results()
    forecaster.plot_results()
    forecaster.summary()
    
    return forecaster


def run_kalman(config_path):
    """Запуск Kalman модели"""
    print("\n" + "="*70)
    print("KALMAN FORECASTER")
    print("="*70)
    
    forecaster = KalmanForecaster(config=config_path)
    forecaster.fit()
    forecaster.save_results()
    forecaster.plot_results()
    forecaster.summary()
    
    return forecaster


def run_commodity(commodity, model='all'):
    """Запуск для одного датасета"""
    config_path = f'configs/{commodity}.yaml'
    
    if not os.path.exists(config_path):
        print(f"ОШИБКА: Конфиг не найден: {config_path}")
        return None, None
    
    print("\n" + "#"*70)
    print(f"# ДАТАСЕТ: {commodity.upper()}")
    print("#"*70)
    
    ml_result = None
    kalman_result = None
    
    if model in ['ml', 'all']:
        try:
            ml_result = run_ml(config_path)
        except Exception as e:
            print(f"Ошибка ML: {e}")
    
    if model in ['kalman', 'all']:
        try:
            kalman_result = run_kalman(config_path)
        except Exception as e:
            print(f"Ошибка Kalman: {e}")
    
    return ml_result, kalman_result


def run_all(model='all'):
    """Запуск для всех датасетов"""
    results = {}
    
    for commodity in COMMODITIES:
        config_path = f'configs/{commodity}.yaml'
        if os.path.exists(config_path):
            ml, kalman = run_commodity(commodity, model)
            results[commodity] = {'ml': ml, 'kalman': kalman}
        else:
            print(f"Пропуск {commodity}: конфиг не найден")
    
    # Сводка
    print("\n" + "="*70)
    print("СВОДКА ПО ВСЕМ ДАТАСЕТАМ")
    print("="*70)
    
    print(f"\n{'Commodity':<15} {'ML Best':<15} {'ML MAE':<10} {'Kalman Best':<20} {'Kalman MAE':<10}")
    print("-"*70)
    
    for commodity, res in results.items():
        ml_best = res['ml'].best_model if res['ml'] else '-'
        ml_mae = f"{res['ml'].metrics[res['ml'].best_model]['mae']:.3f}" if res['ml'] else '-'
        kf_best = res['kalman'].best_method if res['kalman'] else '-'
        kf_mae = f"{res['kalman'].metrics[res['kalman'].best_method]['mae']:.3f}" if res['kalman'] else '-'
        
        print(f"{commodity:<15} {ml_best:<15} {ml_mae:<10} {kf_best:<20} {kf_mae:<10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Commodity Forecasting - Прогнозирование цен на сырьевые товары',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run.py --config configs/sugar.yaml --model ml
  python run.py --commodity wheat --model kalman
  python run.py --all --model all
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
    
    args = parser.parse_args()
    
    # Проверка аргументов
    if not any([args.config, args.commodity, args.all]):
        parser.print_help()
        print("\nОШИБКА: Укажите --config, --commodity или --all")
        sys.exit(1)
    
    print("="*70)
    print("COMMODITY FORECASTING LIBRARY")
    print("="*70)
    
    if args.all:
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
    
    print("\n" + "="*70)
    print("✅ ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == '__main__':
    main()
