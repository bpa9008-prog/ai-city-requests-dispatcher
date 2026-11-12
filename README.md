# Итоговый проект: AI-диспетчер обращений граждан

Битковский П. А., Р4155. Отчёт: `Битковский_Р4155_Итоговый_проект.pdf`.

## Структура
- `data/raw_requests.csv` — исходный синтетический набор (2730 × 13)
- `data/clean_requests.csv` — очищенный (2601 × 26); `eval_set.csv` — 56 размеченных обращений для проверки агента;
  `sla_reference.csv` — справочник сроков; `category_mapping.csv` — Label Encoding
- `output/` — аудит качества, до/после, статистики, графики, результаты агента, экономика
- `scripts/` — весь код (запускать по порядку)

## Запуск
```
pip install -r requirements.txt
python scripts/01_generate_data.py   # синтетические данные (seed 4155)
python scripts/02_eda_clean.py       # анализ, очистка, подготовка
$env:LLM_API_KEY = "<ключ Groq>"     # без ключа агент работает в offline-режиме
python scripts/03_run_agent.py --pause 2
python scripts/04_economics.py
```
