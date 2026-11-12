"""Этап 2. Синтетический журнал обращений граждан в городскую диспетчерскую службу.

Данные СИНТЕТИЧЕСКИЕ. Схема продолжает набор городских заявок из ЛР3–ЛР4
(district, category, priority, response_hours, satisfaction_score, resolved, operator)
и добавляет то, что нужно для AI-задачи: текст обращения, канал, первичную и итоговую
категорию (по ним видно, сколько заявок маршрутизировано ошибочно) и время ручной
сортировки оператором.

Ошибки качества (форматы дат, регистр и синонимы категорий, пропуски, дубликаты,
выбросы, значения вне диапазона) внесены намеренно, чтобы повторить типичный
выгрузочный «мусор» CRM. seed фиксирован -> результат воспроизводим.

Запуск: python scripts/01_generate_data.py  ->  data/raw_requests.csv
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw_requests.csv"
rng = np.random.default_rng(4155)

N = 2600
START = datetime(2026, 3, 1)
DAYS = 184  # 1 марта – 31 августа 2026

DISTRICTS = ["Центральный", "Северный", "Южный", "Западный", "Восточный"]
DISTRICT_W = [0.26, 0.21, 0.20, 0.18, 0.15]
STREETS = {
    "Центральный": ["Ленина", "Садовая", "Невская", "Пушкинская", "Театральная"],
    "Северный": ["Северная", "Полярная", "Лесная", "Озёрная", "Берёзовая"],
    "Южный": ["Южная", "Степная", "Вокзальная", "Заводская", "Рабочая"],
    "Западный": ["Западная", "Морская", "Портовая", "Речная", "Набережная"],
    "Восточный": ["Восточная", "Солнечная", "Новая", "Молодёжная", "Парковая"],
}
CHANNELS = ["portal", "phone", "email", "telegram"]
CHANNEL_W = [0.38, 0.30, 0.12, 0.20]
OPERATORS = ["Иванова", "Петров", "Сидоров", "Кузнецова", "Смирнов", "Попова", "Волков", "Орлова"]

# категория: (вес, медиана часов до решения, доля высокого приоритета по сути)
CATS = {
    "road":     (0.20, 48, 0.15),
    "lighting": (0.14, 20, 0.10),
    "water":    (0.15, 10, 0.35),
    "heating":  (0.10, 14, 0.30),
    "garbage":  (0.19, 12, 0.05),
    "yard":     (0.14, 70, 0.05),
    "other":    (0.08, 30, 0.05),
}

PROBLEMS = {
    "road": ["глубокая яма на проезжей части", "провал асфальта у пешеходного перехода",
             "разбит тротуар, ямы по всей длине", "после ремонта не восстановили асфальт",
             "открытый люк на дороге", "яма во дворе, машины бьют колёса"],
    "lighting": ["не горят фонари на всей улице", "мигает уличный фонарь", "темно во дворе, не работает освещение",
                 "фонарь горит днём и не выключается", "оборван провод уличного освещения"],
    "water": ["нет холодной воды с утра", "из крана течёт ржавая вода", "прорыв трубы, вода бьёт из-под асфальта",
              "очень слабый напор воды", "течёт вода в подвале дома", "нет горячей воды без объявления"],
    "heating": ["в квартире холодно, батареи еле тёплые", "не включили отопление в доме",
                "течёт батарея в подъезде", "в детском саду холодные батареи"],
    "garbage": ["не вывозят мусор уже неделю", "контейнерная площадка переполнена",
                "свалка строительного мусора у дома", "мусоровоз приезжает ночью и шумит",
                "разбросан мусор вокруг контейнеров"],
    "yard": ["сломаны качели на детской площадке", "не скошена трава во дворе",
             "аварийное дерево может упасть", "нет скамеек у подъезда", "разрушена ограда сквера"],
    "other": ["шумят соседи по ночам", "бродячие собаки во дворе", "не работает лифт",
              "незаконная торговля у метро", "граффити на фасаде дома"],
}
URGENT = ["Опасно для детей!", "Срочно, пожалуйста!", "Пожилые люди не могут выйти.",
          "Уже был несчастный случай.", "Есть угроза аварии."]
OPENERS = ["Здравствуйте!", "Добрый день.", "Прошу помочь.", "", "", "Уважаемые сотрудники!"]
CLOSERS = ["Спасибо.", "Прошу принять меры.", "Жду ответа.", "Обращаюсь уже третий раз!", "", ""]
# «смешанные» обращения: основная проблема + упоминание соседней категории
MIXED_TAIL = {
    "road": "Заодно и фонарь рядом не горит.",
    "lighting": "Ещё и мусор не вывозят.",
    "water": "Из-за воды размыло асфальт.",
    "heating": "И вода горячая еле идёт.",
    "garbage": "Контейнеры стоят прямо на дороге, объезжать неудобно.",
    "yard": "Возле площадки ещё и мусор валяется.",
    "other": "И во дворе темно, освещения нет.",
}
CONFUSABLE = {"road": "yard", "lighting": "other", "water": "heating", "heating": "water",
              "garbage": "yard", "yard": "garbage", "other": "yard"}
CAT_RU = {"road": "дороги", "lighting": "освещение", "water": "водоснабжение", "heating": "отопление",
          "garbage": "мусор", "yard": "благоустройство", "other": "прочее"}


def typo(s: str) -> str:
    """Случайная опечатка: пропуск буквы — как в реальных обращениях с телефона."""
    if len(s) < 20 or rng.random() > 0.15:
        return s
    i = int(rng.integers(5, len(s) - 5))
    return s[:i] + s[i + 1:]


rows = []
cat_names = list(CATS)
cat_w = np.array([CATS[c][0] for c in cat_names])
for k in range(N):
    created = START + timedelta(days=float(rng.uniform(0, DAYS)))
    month = created.month
    w = cat_w.copy()
    if month >= 5:  # отопительный сезон закончился: жалоб на отопление почти нет
        w[cat_names.index("heating")] *= 0.12
    if month >= 6:  # летом больше благоустройства и мусора
        w[cat_names.index("yard")] *= 1.6
        w[cat_names.index("garbage")] *= 1.2
    cat = rng.choice(cat_names, p=w / w.sum())
    district = rng.choice(DISTRICTS, p=DISTRICT_W)
    street = rng.choice(STREETS[district])
    house = int(rng.integers(1, 120))
    urgent = rng.random() < CATS[cat][2]
    mixed = rng.random() < 0.12

    parts = [rng.choice(OPENERS),
             f"{rng.choice(PROBLEMS[cat]).capitalize()} по адресу ул. {street}, д. {house}."]
    if urgent:
        parts.append(rng.choice(URGENT))
    if mixed:
        parts.append(MIXED_TAIL[cat])
    parts.append(rng.choice(CLOSERS))
    text = typo(" ".join(p for p in parts if p))

    # Оператор сортирует вручную: ошибается чаще на смешанных текстах и в часы пик
    p_err = 0.38 if mixed else 0.09
    initial = CONFUSABLE[cat] if rng.random() < p_err else cat
    rerouted = initial != cat

    # Приоритет ставит оператор: срочность по тексту иногда пропускается
    if urgent:
        priority = "high" if rng.random() < 0.7 else "medium"
    else:
        priority = rng.choice(["low", "medium", "high"], p=[0.45, 0.47, 0.08])

    triage_min = float(rng.lognormal(np.log(5.5), 0.45)) + (3 if mixed else 0)
    base_h = CATS[cat][1] * (0.5 if priority == "high" else 1.0)
    resp_h = float(rng.lognormal(np.log(base_h), 0.55)) + (float(rng.uniform(18, 60)) if rerouted else 0)
    resolved = rng.random() < (0.80 if rerouted else 0.92)
    sat = np.clip(rng.normal(4.3 - 0.9 * rerouted - 0.008 * resp_h, 0.55), 1, 5) if resolved else np.nan

    rows.append({
        "request_id": 50000 + k,
        "created_at": created,
        "channel": rng.choice(CHANNELS, p=CHANNEL_W),
        "district": district,
        "text": text,
        "category_initial": initial,
        "category_final": cat if resolved else np.nan,
        "priority": priority,
        "operator": rng.choice(OPERATORS),
        "triage_minutes": round(triage_min, 1),
        "response_hours": round(resp_h, 1),
        "resolved": "yes" if resolved else "no",
        "satisfaction_score": round(float(sat), 1) if resolved else np.nan,
        "_true_category": cat,  # служебное: удаляется перед сохранением
    })

df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
df = df.drop(columns="_true_category")


# ---------- внесение ошибок качества ----------
def pick(frac):
    return rng.choice(df.index, size=int(len(df) * frac), replace=False)


# даты: три разных формата из разных каналов выгрузки
fmt = rng.choice(["iso", "ru", "slash"], size=len(df), p=[0.75, 0.17, 0.08])
df["created_at"] = [d.strftime("%Y-%m-%d %H:%M") if f == "iso" else
                    d.strftime("%d.%m.%Y %H:%M") if f == "ru" else d.strftime("%Y/%m/%d")
                    for d, f in zip(df["created_at"], fmt)]
# категории: регистр, пробелы, русские синонимы
for i in pick(0.10):
    df.at[i, "category_initial"] = rng.choice([df.at[i, "category_initial"].title(),
                                               df.at[i, "category_initial"] + " ",
                                               CAT_RU[df.at[i, "category_initial"]]])
for i in pick(0.05):
    df.at[i, "category_initial"] = np.nan
# районы: пропуски и варианты написания
df.loc[pick(0.06), "district"] = np.nan
for i in pick(0.05):
    if isinstance(df.at[i, "district"], str):
        df.at[i, "district"] = rng.choice([df.at[i, "district"].lower(), df.at[i, "district"] + " р-н"])
# канал: регистр
for i in pick(0.08):
    df.at[i, "channel"] = df.at[i, "channel"].upper()
# resolved: разные кодировки
for i in pick(0.06):
    df.at[i, "resolved"] = {"yes": rng.choice(["да", "1", "Yes"]), "no": rng.choice(["нет", "0", "No"])}[df.at[i, "resolved"]]
# числа: десятичная запятая (строки), выбросы и невозможные значения
df["response_hours"] = df["response_hours"].astype(object)
for i in pick(0.04):
    df.at[i, "response_hours"] = str(df.at[i, "response_hours"]).replace(".", ",")
for i in pick(0.006):
    df.at[i, "response_hours"] = rng.choice([-5.0, 0.0, 9999.0])
for i in pick(0.008):
    df.at[i, "triage_minutes"] = rng.choice([0.0, 480.0, 999.0])
for i in pick(0.01):
    if not np.isnan(df.at[i, "satisfaction_score"]):
        df.at[i, "satisfaction_score"] = rng.choice([0.0, 6.0, 10.0, -1.0])
# пустые и бессодержательные тексты
for i in pick(0.012):
    df.at[i, "text"] = rng.choice(["", "   ", "???", "тест", "см. вложение"])
# точные дубликаты строк (повторная выгрузка) и повторные обращения граждан
dups = df.loc[pick(0.03)]
repeats = df.loc[pick(0.02)].copy()
repeats["request_id"] = np.arange(60000, 60000 + len(repeats))
df = pd.concat([df, dups, repeats]).sample(frac=1, random_state=7).reset_index(drop=True)

OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"Сохранено {OUT}: {df.shape[0]} строк x {df.shape[1]} столбцов")
