"""Этапы 2–3. Анализ исходных данных, аудит качества, очистка и подготовка для ИИ.

Вход:  data/raw_requests.csv
Выход: data/clean_requests.csv      — очищенный журнал (все обращения)
       data/eval_set.csv            — размеченная выборка для проверки агента
       data/sla_reference.csv       — справочник сроков для инструмента агента
       data/category_mapping.csv    — Label Encoding категорий
       output/*.csv, output/fig_*.png — таблицы и графики для отчёта
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4df"
C1, C2 = "#2a78d6", "#eb6834"  # categorical slot 1 (blue) и slot 2 (orange)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

CATS = ["road", "lighting", "water", "heating", "garbage", "yard", "other"]
CAT_RU = {"road": "дороги", "lighting": "освещение", "water": "водоснабжение", "heating": "отопление",
          "garbage": "мусор", "yard": "благоустройство", "other": "прочее"}
DISTRICTS = ["Центральный", "Северный", "Южный", "Западный", "Восточный"]
STREET_TO_DISTRICT = {
    **dict.fromkeys(["Ленина", "Садовая", "Невская", "Пушкинская", "Театральная"], "Центральный"),
    **dict.fromkeys(["Северная", "Полярная", "Лесная", "Озёрная", "Берёзовая"], "Северный"),
    **dict.fromkeys(["Южная", "Степная", "Вокзальная", "Заводская", "Рабочая"], "Южный"),
    **dict.fromkeys(["Западная", "Морская", "Портовая", "Речная", "Набережная"], "Западный"),
    **dict.fromkeys(["Восточная", "Солнечная", "Новая", "Молодёжная", "Парковая"], "Восточный"),
}
URGENT_RE = re.compile(r"опасно|срочно|несчастн|угроза|не могут выйти|прорыв|открытый люк|аварийн", re.I)
ADDR_RE = re.compile(r"ул\.\s*([А-ЯЁ][а-яё]+),\s*д\.\s*(\d+)")


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x" if ax.get_ylabel() == "" and ax.get_xlabel() != "" else "y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =====================================================================
# 1. Загрузка «как есть» — все столбцы строками, чтобы увидеть реальные значения
# =====================================================================
raw = pd.read_csv(DATA / "raw_requests.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
raw = raw.replace({"": np.nan})
print("Исходный набор:", raw.shape)

profile = pd.DataFrame({
    "тип в CSV": "строка",
    "уникальных": raw.nunique(),
    "пропусков": raw.isna().sum(),
    "пример": raw.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
})
profile.to_csv(OUT / "raw_profile.csv", encoding="utf-8-sig")


def to_num(s):
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


# =====================================================================
# 2. Аудит качества: считаем каждую проблему до изменений
# =====================================================================
issues = []


def issue(name, n, action):
    issues.append({"проблема": name, "строк": int(n), "что сделано": action})


df = raw.copy()
n_exact = df.duplicated().sum()
issue("Полные дубликаты строк (повторная выгрузка)", n_exact, "удалены")
df = df.drop_duplicates().reset_index(drop=True)

text_norm = df["text"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
rep_mask = df.assign(t=text_norm).duplicated(subset=["t", "created_at"], keep="first") & (text_norm.str.len() >= 15)
issue("Повторные обращения: тот же текст и время, другой request_id", rep_mask.sum(),
      "удалены, оставлено первое (иначе завышают объём и нагрузку)")
df = df[~rep_mask].reset_index(drop=True)
df["text"] = text_norm[~rep_mask].reset_index(drop=True)

dt_iso = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M", errors="coerce")
dt_ru = pd.to_datetime(df["created_at"], format="%d.%m.%Y %H:%M", errors="coerce")
dt_sl = pd.to_datetime(df["created_at"], format="%Y/%m/%d", errors="coerce")
issue("Дата в формате ДД.ММ.ГГГГ ЧЧ:ММ", dt_ru.notna().sum(), "приведена к datetime")
issue("Дата в формате ГГГГ/ММ/ДД без времени", dt_sl.notna().sum(),
      "приведена к datetime, время 00:00 (час не используется для этих строк)")
df["created_at"] = dt_iso.fillna(dt_ru).fillna(dt_sl)
df["time_known"] = dt_sl.isna().astype("int8")
assert df["created_at"].notna().all()

ch = df["channel"].str.strip().str.lower()
issue("Канал в разном регистре (PORTAL/portal)", (ch != df["channel"]).sum(), "приведён к нижнему регистру")
df["channel"] = ch

d = df["district"].str.replace(" р-н", "", regex=False).str.strip().str.capitalize()
issue("Район в другом написании (строчные, «р-н»)", (d.notna() & (d != df["district"])).sum(), "нормализован")
df["district"] = d

addr = df["text"].str.extract(ADDR_RE)
df["street"], df["house"] = addr[0], pd.to_numeric(addr[1], errors="coerce").astype("Int64")
miss_d = df["district"].isna()
recovered = miss_d & df["street"].map(STREET_TO_DISTRICT).notna()
issue("Пропуск района", miss_d.sum(),
      f"{recovered.sum()} восстановлено по улице из текста, остальные -> «Не указан»")
df.loc[recovered, "district"] = df.loc[recovered, "street"].map(STREET_TO_DISTRICT)
df["district"] = df["district"].fillna("Не указан")

ru2en = {v: k for k, v in CAT_RU.items()}
ci = df["category_initial"].str.strip().str.lower().replace(ru2en)
issue("Категория: регистр, пробелы, русские синонимы", (ci.notna() & (ci != df["category_initial"])).sum(),
      "приведена к 7 кодам (road, lighting, ...)")
issue("Пропуск первичной категории", ci.isna().sum(),
      "оставлен NaN: неизвестно, куда отправили; в расчёт ошибок маршрутизации не входит")
df["category_initial"] = ci
assert set(df["category_initial"].dropna()) <= set(CATS)
issue("Пропуск итоговой категории", df["category_final"].isna().sum(),
      "структурный пропуск: обращение не закрыто; оставлен NaN")

res = df["resolved"].str.strip().str.lower().map({"yes": 1, "да": 1, "1": 1, "no": 0, "нет": 0, "0": 0})
issue("resolved в разных кодировках (да/1/Yes)", (~df["resolved"].isin(["yes", "no"])).sum(), "-> 1/0 (int8)")
df["resolved"] = res.astype("int8")

rh_comma = df["response_hours"].str.contains(",", na=False)
issue("response_hours с десятичной запятой (строка)", rh_comma.sum(), "запятая -> точка, тип float")
rh = to_num(df["response_hours"])
bad_rh = (rh <= 0) | (rh > 720)
issue("response_hours <= 0 или > 720 ч (-5, 0, 9999 — заглушки)", bad_rh.sum(),
      "-> NaN: строка остаётся, значение не используется")
df["response_hours"] = rh.mask(bad_rh)

tm = to_num(df["triage_minutes"])
bad_tm = (tm <= 0) | (tm > 120)
issue("triage_minutes = 0 или > 120 мин (480, 999)", bad_tm.sum(),
      "-> NaN: на сортировку одного обращения не уходит 8–16 часов")
df["triage_minutes"] = tm.mask(bad_tm)

sat = to_num(df["satisfaction_score"])
bad_sat = sat.notna() & ~sat.between(1, 5)
issue("satisfaction_score вне шкалы 1–5 (0, 6, 10, -1)", bad_sat.sum(), "-> NaN")
df["satisfaction_score"] = sat.mask(bad_sat)

df["text_valid"] = (df["text"].str.len() >= 25).astype("int8")
issue("Пустой или бессодержательный текст («???», «тест»)", (df["text_valid"] == 0).sum(),
      "строка оставлена с флагом text_valid=0, исключена из выборки для агента")

df["request_id"] = df["request_id"].astype("int64")
assert df["request_id"].is_unique
pd.DataFrame(issues).to_csv(OUT / "quality_report.csv", index=False, encoding="utf-8-sig")

# =====================================================================
# 3. Производные признаки и кодирование
# =====================================================================
df["rerouted"] = np.where(df["category_initial"].isna() | df["category_final"].isna(), np.nan,
                          (df["category_initial"] != df["category_final"]).astype(float))
df["month"] = df["created_at"].dt.month.astype("int8")
df["weekday"] = df["created_at"].dt.dayofweek.astype("int8")
df["hour"] = df["created_at"].dt.hour.where(df["time_known"] == 1).astype("Int8")
df["text_len"] = df["text"].str.len().astype("int16")
df["urgent_marker"] = df["text"].str.contains(URGENT_RE).astype("int8")
df["priority"] = pd.Categorical(df["priority"], categories=["low", "medium", "high"], ordered=True)
df["priority_ord"] = df["priority"].cat.codes.astype("int8")
cat_code = {c: i for i, c in enumerate(CATS)}  # фиксированный порядок, а не алфавит
df["category_initial_code"] = df["category_initial"].map(cat_code).astype("Int8")
df["category_final_code"] = df["category_final"].map(cat_code).astype("Int8")
pd.DataFrame({"code": range(len(CATS)), "category": CATS, "category_ru": [CAT_RU[c] for c in CATS]}
             ).to_csv(DATA / "category_mapping.csv", index=False, encoding="utf-8-sig")

COLS = ["request_id", "created_at", "time_known", "month", "weekday", "hour", "channel", "district",
        "street", "house", "text", "text_len", "text_valid", "urgent_marker",
        "category_initial", "category_initial_code", "category_final", "category_final_code", "rerouted",
        "priority", "priority_ord", "operator", "triage_minutes", "response_hours", "resolved",
        "satisfaction_score"]
clean = df[COLS].sort_values("created_at").reset_index(drop=True)
clean.to_csv(DATA / "clean_requests.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")

# =====================================================================
# 4. Сравнение «до / после»
# =====================================================================
cmp_rows = [
    ("Строк", len(raw), len(clean)),
    ("Столбцов", raw.shape[1], clean.shape[1]),
    ("Уникальных значений category_initial", raw["category_initial"].nunique(), clean["category_initial"].nunique()),
    ("Уникальных значений district", raw["district"].nunique(), clean["district"].nunique()),
    ("Уникальных значений channel", raw["channel"].nunique(), clean["channel"].nunique()),
    ("Уникальных значений resolved", raw["resolved"].nunique(), clean["resolved"].nunique()),
    ("Форматов даты", 3, 1),
    ("Пропусков district", raw["district"].isna().sum(), (clean["district"] == "Не указан").sum()),
    ("Числовых столбцов (тип float/int)", 0, int(clean.select_dtypes("number").shape[1])),
    ("max response_hours", to_num(raw["response_hours"]).max(), clean["response_hours"].max()),
    ("max triage_minutes", to_num(raw["triage_minutes"]).max(), clean["triage_minutes"].max()),
    ("Диапазон satisfaction_score",
     f"{to_num(raw['satisfaction_score']).min():g}..{to_num(raw['satisfaction_score']).max():g}",
     f"{clean['satisfaction_score'].min():g}..{clean['satisfaction_score'].max():g}"),
]
before_after = pd.DataFrame(cmp_rows, columns=["показатель", "исходный", "очищенный"])
before_after.to_csv(OUT / "before_after.csv", index=False, encoding="utf-8-sig")
print(before_after.to_string(index=False))

# =====================================================================
# 5. Анализ (на очищенных данных)
# =====================================================================
stats = {}
num = clean[["triage_minutes", "response_hours", "satisfaction_score", "text_len"]]
desc = num.describe().T
desc["skew"] = num.skew()
desc.round(2).to_csv(OUT / "numeric_stats.csv", encoding="utf-8-sig")

closed = clean[clean["rerouted"].notna()]
stats["n_clean"] = len(clean)
stats["period"] = f"{clean['created_at'].min():%d.%m.%Y} – {clean['created_at'].max():%d.%m.%Y}"
stats["per_month"] = round(len(clean) / 6, 0)
stats["reroute_rate"] = round(closed["rerouted"].mean(), 4)
stats["n_closed_known"] = len(closed)
stats["triage_median"] = round(clean["triage_minutes"].median(), 2)
stats["triage_mean"] = round(clean["triage_minutes"].mean(), 2)
stats["resp_median_ok"] = round(closed.loc[closed["rerouted"] == 0, "response_hours"].median(), 1)
stats["resp_median_rerouted"] = round(closed.loc[closed["rerouted"] == 1, "response_hours"].median(), 1)
stats["sat_ok"] = round(closed.loc[closed["rerouted"] == 0, "satisfaction_score"].mean(), 2)
stats["sat_rerouted"] = round(closed.loc[closed["rerouted"] == 1, "satisfaction_score"].mean(), 2)
stats["resolved_rate"] = round(clean["resolved"].mean(), 4)
urg = clean[clean["urgent_marker"] == 1]
stats["urgent_n"] = len(urg)
stats["urgent_not_high"] = round((urg["priority"] != "high").mean(), 4)
stats["corr_resp_sat"] = round(clean[["response_hours", "satisfaction_score"]].corr().iloc[0, 1], 3)
stats["text_len_median"] = int(clean["text_len"].median())
stats["channels"] = clean["channel"].value_counts(normalize=True).round(3).to_dict()

by_cat = (closed.groupby("category_final")
          .agg(обращений=("request_id", "size"), доля_ошибок_маршрутизации=("rerouted", "mean"),
               медиана_часов=("response_hours", "median"), удовлетворённость=("satisfaction_score", "mean"))
          .reindex(CATS).round(3))
by_cat.index = [CAT_RU[c] for c in by_cat.index]
by_cat.to_csv(OUT / "by_category.csv", encoding="utf-8-sig")

reroute_pairs = (closed[closed["rerouted"] == 1].groupby(["category_final", "category_initial"]).size()
                 .sort_values(ascending=False).head(6))
stats["top_confusions"] = [f"{CAT_RU[a]} -> ошибочно {CAT_RU[b]}: {n}" for (a, b), n in reroute_pairs.items()]

monthly = clean.groupby(["month", "category_final"]).size().unstack(fill_value=0).reindex(columns=CATS)
monthly.to_csv(OUT / "monthly_by_category.csv", encoding="utf-8-sig")
(OUT / "eda_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2, default=float))

# --- графики ---
fig, ax = plt.subplots(figsize=(7.5, 3.2))
counts = clean["category_final"].value_counts().reindex(CATS)
labels = [CAT_RU[c] for c in CATS]
ax.barh(labels[::-1], counts.values[::-1], color=C1, height=0.6)
for y, v in enumerate(counts.values[::-1]):
    ax.text(v + 8, y, f"{v}", va="center", color=MUTED, fontsize=9)
ax.set_xlabel("закрытых обращений")
style(ax, "Обращения по итоговой категории, март–август 2026")
ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig_categories.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
rr = by_cat["доля_ошибок_маршрутизации"] * 100
axes[0].bar(rr.index, rr.values, color=C1, width=0.6)
for x, v in enumerate(rr.values):
    axes[0].text(x, v + 0.5, f"{v:.0f}%", ha="center", color=MUTED, fontsize=8)
axes[0].tick_params(axis="x", rotation=35, labelsize=8)
axes[0].set_ylabel("% переадресованных")
style(axes[0], "Доля ошибок первичной маршрутизации")
g = closed.groupby(["category_final", "rerouted"])["response_hours"].median().unstack().reindex(CATS)
x = np.arange(len(CATS))
axes[1].bar(x - 0.2, g[0.0], 0.38, color=C1, label="направлено верно")
axes[1].bar(x + 0.2, g[1.0], 0.38, color=C2, label="переадресовано")
axes[1].set_xticks(x, labels, rotation=35, fontsize=8)
axes[1].set_ylabel("медиана, часов до решения")
axes[1].legend(frameon=False, fontsize=8)
style(axes[1], "Срок решения: верно vs переадресовано")
fig.tight_layout(); fig.savefig(OUT / "fig_rerouting.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
axes[0].hist(clean["triage_minutes"].dropna(), bins=30, color=C1, edgecolor="white", linewidth=0.8)
axes[0].axvline(stats["triage_median"], color=INK, lw=1.5, ls="--")
axes[0].text(stats["triage_median"] + 0.5, axes[0].get_ylim()[1] * 0.9,
             f"медиана {stats['triage_median']:.1f} мин", color=INK, fontsize=8)
axes[0].set_xlabel("минут на сортировку одного обращения")
style(axes[0], "Ручная сортировка (triage_minutes)")
mtot = clean.groupby("month").size()
heat = clean[clean["category_final"] == "heating"].groupby("month").size().reindex(mtot.index, fill_value=0)
mnames = ["мар", "апр", "май", "июн", "июл", "авг"]
axes[1].bar(mnames, mtot.values, color=C1, width=0.6, label="все обращения")
axes[1].bar(mnames, heat.values, color=C2, width=0.6, label="из них отопление")
axes[1].legend(frameon=False, fontsize=8)
style(axes[1], "Объём по месяцам: сезонность")
fig.tight_layout(); fig.savefig(OUT / "fig_triage_monthly.png", dpi=160); plt.close(fig)

q = pd.DataFrame(issues).sort_values("строк")
q = q[q["строк"] > 0]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.barh([s if len(s) < 62 else s[:60] + "…" for s in q["проблема"]], q["строк"], color=C1, height=0.6)
for y, v in enumerate(q["строк"]):
    ax.text(v + 5, y, str(v), va="center", color=MUTED, fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.set_xlabel("затронуто строк")
style(ax, "Проблемы качества в исходной выгрузке")
ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig_quality.png", dpi=160); plt.close(fig)

# =====================================================================
# 6. Наборы для агента
# =====================================================================
# Выборка для проверки: только закрытые обращения с валидным текстом, 8 на категорию,
# с перевесом «трудных» случаев (где оператор ошибся) — именно они показывают пользу ИИ.
pool = clean[(clean["text_valid"] == 1) & clean["category_final"].notna() & clean["rerouted"].notna()]
parts = []
for c in CATS:
    p = pool[pool["category_final"] == c]
    hard = p[p["rerouted"] == 1].sample(min(3, (p["rerouted"] == 1).sum()), random_state=1)
    easy = p[p["rerouted"] == 0].sample(8 - len(hard), random_state=1)
    parts += [hard, easy]
ev = pd.concat(parts).sample(frac=1, random_state=2)
ev[["request_id", "created_at", "channel", "district", "text", "category_initial", "category_final",
    "priority", "rerouted"]].to_csv(DATA / "eval_set.csv", index=False, encoding="utf-8-sig",
                                     date_format="%Y-%m-%d %H:%M")
print("Выборка для агента:", len(ev), "обращений, из них переадресованных:", int(ev["rerouted"].sum()))

# Справочник сроков: медиана и 80-й перцентиль по категории и приоритету, только верно направленные
ok = closed[closed["rerouted"] == 0]
sla = (ok.groupby(["category_final", "priority"], observed=True)["response_hours"]
       .agg(n="size", median_h="median", p80_h=lambda s: s.quantile(0.8)).round(1).reset_index())
sla.to_csv(DATA / "sla_reference.csv", index=False, encoding="utf-8-sig")
print("Готово.")
