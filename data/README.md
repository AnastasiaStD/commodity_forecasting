# Data Directory

Поместите сюда ваши Excel файлы с данными.

## Ожидаемые файлы

| Файл | Конфиг |
|------|--------|
| `sugar_factors.xlsx` | `configs/sugar.yaml` |
| `wheat_factors.xlsx` | `configs/wheat.yaml` |
| `milk_factors.xlsx` | `configs/milk.yaml` |
| `sfoil_factors.xlsx` | `configs/sf_oil.yaml` |
| `po_factors.xlsx` | `configs/palm_oil.yaml` |
| `soybean_factors.xlsx` | `configs/soybean.yaml` |

## Формат данных

Excel файл должен содержать:
- Колонку с датой (например, `Date` или `dt`)
- Колонку с целевой переменной (например, `Price`, `Sugar Price`)
- Колонки с признаками (факторы-драйверы)

Пример:
| Date | Price | Feature1 | Feature2 | Feature3 |
|------|-------|----------|----------|----------|
| 2020-01-01 | 100.5 | 65.2 | 97.3 | 0.25 |
| 2020-02-01 | 102.3 | 66.1 | 98.1 | 0.24 |
| ... | ... | ... | ... | ... |
