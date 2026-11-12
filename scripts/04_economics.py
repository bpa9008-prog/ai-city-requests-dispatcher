"""Этап 6. Экономическая модель: стоимость вызова, затраты, эффект, окупаемость.

Все исходные значения — в словаре A (с источником). Результаты: output/economics_*.csv,
output/economics.json, output/fig_payback.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
import pandas as pd
import tiktoken
from langchain_core.utils.function_calling import convert_to_openai_tool

sys.path.insert(0, str(Path(__file__).parent))
import agent as AG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
eda = json.loads((OUT / "eda_stats.json").read_text(encoding="utf-8"))
ags = json.loads((OUT / "agent_summary.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- токены одного обращения
enc = tiktoken.get_encoding("o200k_base")   # токенизатор семейства gpt-oss
tok = lambda s: len(enc.encode(s))
SYS = tok(AG.SYSTEM_PROMPT)
TOOLS = tok(json.dumps([convert_to_openai_tool(t) for t in AG.TOOLS], ensure_ascii=False))
ev = pd.read_csv(ROOT / "data" / "eval_set.csv", encoding="utf-8-sig")
USER = round(ev["text"].map(tok).mean() + 30)   # текст + служебная строка «Обращение №…»
demo = json.loads((OUT / "agent_demo.json").read_text(encoding="utf-8"))
CARD = round(sum(tok(json.dumps(d["card"], ensure_ascii=False)) for d in demo if d["card"]) / len(demo))
TOOL_RES, SMALL_CALL, REASONING = 70, 40, 200   # ответ инструмента, короткий вызов, «рассуждение» на шаг

# Цикл агента = 4 вызова модели; каждый раз заново отправляется вся история.
base = SYS + TOOLS + USER
calls = [
    (base,                                               REASONING + SMALL_CALL),   # -> lookup_address
    (base + SMALL_CALL + TOOL_RES,                       REASONING + SMALL_CALL),   # -> get_routing_and_sla
    (base + 2 * (SMALL_CALL + TOOL_RES),                 REASONING + CARD),         # -> submit_triage_card
    (base + 2 * (SMALL_CALL + TOOL_RES) + CARD + 30,     50),                       # -> «Готово»
]
tok_in_est = sum(c[0] for c in calls)
tok_out_est = sum(c[1] for c in calls)
measured = ags["mode"] == "llm" and ags["avg_input_tokens"] > 0
TOK_IN = ags["avg_input_tokens"] if measured else tok_in_est
TOK_OUT = ags["avg_output_tokens"] if measured else tok_out_est

# ---------------------------------------------------------------- допущения
A = {
    "price_in_usd_per_1m":  (0.15, "Groq, openai/gpt-oss-120b, прайс на 10.2026"),
    "price_out_usd_per_1m": (0.60, "Groq, openai/gpt-oss-120b, прайс на 10.2026"),
    "usd_rub":              (90.0, "допущение"),
    "volume_per_month":     (eda["per_month"], "данные: 2601 обращение за 6 мес."),
    "triage_min_now":       (eda["triage_mean"], "данные: среднее triage_minutes"),
    "auto_share":           (ags["auto_route_share"], f"прогон агента ({ags['mode']}), доля auto_route"),
    "min_auto":             (1.0, "допущение: оператор просматривает карточку и утверждает ответ"),
    "min_review":           (3.5, "допущение: карточка заполнена, оператор проверяет и правит"),
    "reroute_now":          (eda["reroute_rate"], "данные: доля переадресованных"),
    "reroute_with_ai":      (0.04, "допущение, проверяется на пилоте"),
    "min_per_reroute":      (15.0, "допущение: повторная обработка вторым оператором и службой"),
    "operator_salary_gross": (55000, "допущение: оклад оператора, ₽/мес"),
    "payroll_tax":          (0.302, "страховые взносы 30,2%"),
    "work_hours_month":     (164, "норма часов при 40-часовой неделе"),
    "hosting_month":        (1500, "VPS для сервиса агента, ₽/мес"),
    "support_hours_month":  (6, "допущение: инженер на сопровождение, ч/мес"),
    "engineer_rate":        (2000, "допущение, ₽/ч"),
    "qa_hours_month":       (4, "старший оператор выборочно проверяет 5% карточек, ч/мес"),
}
v = {k: x[0] for k, x in A.items()}
pd.DataFrame([(k, x[0], x[1]) for k, x in A.items()], columns=["параметр", "значение", "источник"]
             ).to_csv(OUT / "economics_assumptions.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- стоимость вызова
call_usd = (TOK_IN * v["price_in_usd_per_1m"] + TOK_OUT * v["price_out_usd_per_1m"]) / 1_000_000
call_rub = call_usd * v["usd_rub"]
per_call_tbl = pd.DataFrame(
    [(f"вызов {i + 1}", ti, to, round((ti * v['price_in_usd_per_1m'] + to * v['price_out_usd_per_1m']) / 1e6, 6))
     for i, (ti, to) in enumerate(calls)],
    columns=["шаг", "входные токены", "выходные токены", "стоимость, $"])
per_call_tbl.to_csv(OUT / "economics_tokens.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- разовые затраты (суперкоманда)
DEV = pd.DataFrame([
    ("Владелец процесса / аналитик (от службы)", 40, 1500),
    ("LLM-инженер: промпт, инструменты, оценка качества", 80, 2500),
    ("Backend-разработчик: интеграция с CRM, очередь карточек", 60, 2200),
    ("Разметчик/QA: эталонная выборка 300 обращений", 30, 1000),
    ("Юрист/ИБ: 152-ФЗ, обезличивание, договор с провайдером", 10, 2500),
    ("Обучение 8 операторов по 2 ч", 16, round(v["operator_salary_gross"] * (1 + v["payroll_tax"]) / v["work_hours_month"])),
], columns=["статья", "часы", "ставка, ₽/ч"])
DEV["сумма, ₽"] = DEV["часы"] * DEV["ставка, ₽/ч"]
DEV.to_csv(OUT / "economics_capex.csv", index=False, encoding="utf-8-sig")
CAPEX = int(DEV["сумма, ₽"].sum())

# ---------------------------------------------------------------- эффект на одно обращение
rub_per_min = v["operator_salary_gross"] * (1 + v["payroll_tax"]) / v["work_hours_month"] / 60
min_with_ai = v["auto_share"] * v["min_auto"] + (1 - v["auto_share"]) * v["min_review"]
saved_triage = v["triage_min_now"] - min_with_ai
saved_reroute = (v["reroute_now"] - v["reroute_with_ai"]) * v["min_per_reroute"]
saved_min = saved_triage + saved_reroute
saving_rub = saved_min * rub_per_min
net_per_task = saving_rub - call_rub
STATIC = v["hosting_month"] + v["support_hours_month"] * v["engineer_rate"] + v["qa_hours_month"] * rub_per_min * 60


def scenario(name, volume, capex=CAPEX, static=STATIC, net=net_per_task):
    monthly = volume * net - static
    months = capex / monthly if monthly > 0 else float("inf")
    return {"сценарий": name, "обращений в месяц": int(volume), "разовые затраты, ₽": int(capex),
            "постоянные затраты, ₽/мес": int(static), "экономия за вычетом LLM, ₽/мес": int(volume * net),
            "чистый эффект, ₽/мес": int(monthly),
            "окупаемость, мес": round(months, 1) if months != float("inf") else "не окупается",
            "задач до окупаемости": int(round(months * volume)) if months != float("inf") else "—"}


breakeven_volume = STATIC / net_per_task
scen = pd.DataFrame([
    scenario("A. Текущий поток службы", v["volume_per_month"]),
    scenario("B. Пессимистичный: auto 40%, переадресаций 8%, цена LLM x20", v["volume_per_month"],
             net=((v["triage_min_now"] - (0.4 * v["min_auto"] + 0.6 * v["min_review"]))
                  + (v["reroute_now"] - 0.08) * v["min_per_reroute"]) * rub_per_min - 20 * call_rub),
    scenario("C. Городской контакт-центр (5 таких служб)", 5 * v["volume_per_month"]),
    scenario("D. MVP силами службы (n8n, без интеграции с CRM)", v["volume_per_month"],
             capex=int(DEV["сумма, ₽"].iloc[[0, 1, 3, 4, 5]].sum() * 0.6), static=v["hosting_month"] + 2 * v["engineer_rate"]),
])
scen.to_csv(OUT / "economics_scenarios.csv", index=False, encoding="utf-8-sig")

res = {
    "tokens_measured": measured, "sys_tokens": SYS, "tool_schema_tokens": TOOLS, "user_tokens": USER,
    "card_tokens": CARD, "tok_in": round(TOK_IN), "tok_out": round(TOK_OUT),
    "call_usd": round(call_usd, 6), "call_rub": round(call_rub, 3),
    "llm_month_rub": round(call_rub * v["volume_per_month"], 1),
    "rub_per_min": round(rub_per_min, 2), "min_with_ai": round(min_with_ai, 2),
    "saved_triage_min": round(saved_triage, 2), "saved_reroute_min": round(saved_reroute, 2),
    "saved_min": round(saved_min, 2), "saving_rub": round(saving_rub, 2), "net_per_task": round(net_per_task, 2),
    "capex": CAPEX, "static_month": round(STATIC), "breakeven_volume": round(breakeven_volume),
    "tasks_to_cover_capex_only": round(CAPEX / net_per_task),
    "hours_saved_month": round(saved_min * v["volume_per_month"] / 60, 1),
}
(OUT / "economics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
print(scen.to_string(index=False))

# ---------------------------------------------------------------- график окупаемости
fig, ax = plt.subplots(figsize=(8, 3.4))
months = range(0, 37)
colors = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#6250d6"}
for _, r in scen.iterrows():
    key = r["сценарий"][0]
    ys = [(-r["разовые затраты, ₽"] + m * r["чистый эффект, ₽/мес"]) / 1000 for m in months]
    ax.plot(months, ys, lw=2, color=colors[key], label=r["сценарий"].split(":")[0].split(" (")[0])
ax.axhline(0, color="#52514e", lw=1)
ax.set_xlabel("месяцев после запуска"); ax.set_ylabel("накопленный эффект, тыс. ₽")
ax.set_title("Накопленный чистый эффект по сценариям", loc="left", fontsize=11, color="#0b0b0b")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e5e4df"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_payback.png", dpi=160); plt.close(fig)
