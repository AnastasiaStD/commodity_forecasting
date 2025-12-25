"""
Logger - Модуль логирования для Commodity Forecasting
======================================================

Обеспечивает:
- Вывод в консоль И в файл одновременно
- Временные метки для каждого сообщения
- Измерение времени выполнения
- Автоматическое создание лог-файлов с датой/временем

Использование:
    from forecasting.utils.logger import Logger, get_logger
    
    # Инициализация (один раз в начале)
    logger = Logger(log_dir='logs', name='my_run')
    
    # Использование везде в коде
    logger = get_logger()
    logger.info("Сообщение")
    logger.section("ЗАГОЛОВОК РАЗДЕЛА")
    
    # Замер времени
    with logger.timer("Обучение модели"):
        model.fit()
"""

import os
import sys
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path


_global_logger = None


def get_logger():
    """Получить глобальный логгер"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def set_logger(logger):
    """Установить глобальный логгер"""
    global _global_logger
    _global_logger = logger


class Logger:
    """
    Логгер с выводом в консоль и файл.
    
    Parameters:
    -----------
    log_dir : str
        Директория для лог-файлов (default: 'logs')
    name : str
        Имя лог-файла (default: 'run')
    console : bool
        Выводить в консоль (default: True)
    file : bool
        Записывать в файл (default: True)
    timestamp_format : str
        Формат временных меток
    """
    
    def __init__(self, log_dir='logs', name='run', console=True, file=True,
                 timestamp_format='%Y-%m-%d %H:%M:%S'):
        self.console = console
        self.file = file
        self.timestamp_format = timestamp_format
        self.start_time = datetime.now()
        self.timers = {}
        
        if file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            self.log_path = os.path.join(log_dir, f'{name}_{timestamp}.log')
     
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*70}\n")
                f.write(f"COMMODITY FORECASTING LOG\n")
                f.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*70}\n\n")
        else:
            self.log_path = None
        

        set_logger(self)
    
    def _get_timestamp(self):
        """Получить текущую временную метку"""
        return datetime.now().strftime(self.timestamp_format)
    
    def _write(self, message, prefix=''):
        """Записать сообщение"""
        timestamp = self._get_timestamp()
        
        if prefix:
            formatted = f"[{timestamp}] {prefix}: {message}"
        else:
            formatted = f"[{timestamp}] {message}"
        
        if self.console:
            print(message) 
        
        if self.file and self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
    
    def info(self, message):
        """Информационное сообщение"""
        self._write(message, 'INFO')
    
    def debug(self, message):
        """Отладочное сообщение (только в файл)"""
        if self.file and self.log_path:
            timestamp = self._get_timestamp()
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] DEBUG: {message}\n")
    
    def warning(self, message):
        """Предупреждение"""
        self._write(f"⚠️  {message}", 'WARNING')
    
    def error(self, message):
        """Ошибка"""
        self._write(f"❌ {message}", 'ERROR')
    
    def success(self, message):
        """Успех"""
        self._write(f"✅ {message}", 'SUCCESS')
    
    def print(self, message=''):
        """Простой вывод (как print)"""
        if self.console:
            print(message)
        if self.file and self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def section(self, title, char='=', width=70):
        """Заголовок раздела"""
        line = char * width
        self.print('')
        self.print(line)
        self.print(title)
        self.print(line)
    
    def subsection(self, title, char='-', width=50):
        """Подзаголовок"""
        line = char * width
        self.print('')
        self.print(line)
        self.print(title)
        self.print(line)
    
    def table_row(self, *args, widths=None):
        """Строка таблицы"""
        if widths is None:
            row = '  '.join(str(a) for a in args)
        else:
            row = ''.join(f"{str(a):<{w}}" for a, w in zip(args, widths))
        self.print(row)
    
    @contextmanager
    def timer(self, name):
        """
        Контекстный менеджер для измерения времени.
        
        Использование:
            with logger.timer("Обучение модели"):
                model.fit()
        """
        start = time.time()
        self.info(f"Начало: {name}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timers[name] = elapsed
            
            if elapsed < 60:
                time_str = f"{elapsed:.2f} сек"
            elif elapsed < 3600:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes} мин {seconds:.1f} сек"
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                time_str = f"{hours} ч {minutes} мин"
            
            self.success(f"Завершено: {name} ({time_str})")
    
    def start_timer(self, name):
        """Запустить таймер вручную"""
        self.timers[f"_start_{name}"] = time.time()
        self.info(f"Начало: {name}")
    
    def stop_timer(self, name):
        """Остановить таймер вручную"""
        start_key = f"_start_{name}"
        if start_key in self.timers:
            elapsed = time.time() - self.timers[start_key]
            self.timers[name] = elapsed
            del self.timers[start_key]
            
            if elapsed < 60:
                time_str = f"{elapsed:.2f} сек"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes} мин {seconds:.1f} сек"
            
            self.success(f"Завершено: {name} ({time_str})")
            return elapsed
        return None
    
    def get_elapsed(self, name):
        """Получить время выполнения задачи"""
        return self.timers.get(name)
    
    def summary(self):
        """Вывод итоговой сводки по времени"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        self.section("СВОДКА ПО ВРЕМЕНИ ВЫПОЛНЕНИЯ")
        
        completed = {k: v for k, v in self.timers.items() if not k.startswith('_start_')}
        
        if completed:
            self.print(f"\n{'Задача':<40} {'Время':<15}")
            self.print("-" * 55)
            
            for name, elapsed in completed.items():
                if elapsed < 60:
                    time_str = f"{elapsed:.2f} сек"
                elif elapsed < 3600:
                    minutes = int(elapsed // 60)
                    seconds = elapsed % 60
                    time_str = f"{minutes} мин {seconds:.1f} сек"
                else:
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    time_str = f"{hours} ч {minutes} мин"
                
                self.print(f"{name:<40} {time_str:<15}")
        
        self.print("-" * 55)
        
        if total_time < 60:
            total_str = f"{total_time:.2f} сек"
        elif total_time < 3600:
            minutes = int(total_time // 60)
            seconds = total_time % 60
            total_str = f"{minutes} мин {seconds:.1f} сек"
        else:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            total_str = f"{hours} ч {minutes} мин"
        
        self.print(f"{'ОБЩЕЕ ВРЕМЯ:':<40} {total_str:<15}")
        self.print('')
        
        if self.log_path:
            self.print(f"Лог сохранен: {self.log_path}")
    
    def close(self):
        """Закрыть логгер и записать финальную информацию"""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        if self.file and self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"{'='*70}\n")


class DualOutput:
    """
    Перенаправление stdout в консоль и файл одновременно.
    
    Использование:
        dual = DualOutput('output.log')
        sys.stdout = dual
        print("Это пойдет и в консоль, и в файл")
        dual.close()
        sys.stdout = sys.__stdout__
    """
    
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()
