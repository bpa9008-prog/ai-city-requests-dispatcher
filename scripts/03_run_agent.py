"""Этап 5. Демонстрация и проверка агента.

  python scripts/03_run_agent.py                  # режим выбирается сам: llm, если задан LLM_API_KEY
  python scripts/03_run_agent.py --offline        # принудительно без LLM
  python scripts/03_run_agent.py --limit 10       # проверить только 10 обращений из eval_set
  python scripts/03_run_agent.py --interactive    # оператор утверждает карточки вручную (y/n)

Результаты: output/agent_demo.json, agent_eval.csv, agent_summary.json (в офлайн-режиме — с суффиксом _offline)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import agent as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--interactive", action="store_true")
ap.add_argument("--pause", type=float, default=0.0, help="пауза между обращениями (лимиты бесплатного тарифа)")
args = ap.parse_args()

MODE = "llm" if os.environ.get("LLM_API_KEY") and not args.offline else "offline"
agent = A.build_llm_agent() if MODE == "llm" else None
SUF = "" if MODE == "llm" else "_offline"   # офлайн-базовая линия пишется в отдельные файлы
MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b") if MODE == "llm" else "keyword rules (без LLM)"
print(f"Режим: {MODE} | модель: {MODEL}\n")


def show(req, res):
    c = res["card"]
    print(f"--- №{req['request_id']}: «{req['text']}»")
    print("    вызовы:", " -> ".join(res["calls"]) or "—")
    if c is None:
        print("    КАРТОЧКА НЕ СОЗДАНА (агент не вызвал submit_triage_card) -> ручная обработка\n")
        return
    print(f"    {c['category']} | {c['priority']} | {c['district']} | ув. {c['confidence']} | {c['department']}")
    print(f"    суть: {c['summary']}")
    print(f"    черновик ответа: {c['reply_draft']}")
    print(f"    СТАТУС: {c['status']} — {c['review_reason']}\n")


def operator_review(card):
    """Контрольная точка: без решения человека ни маршрут, ни ответ гражданину не уходят."""
    if not args.interactive:
        return "ожидает оператора"
    ans = input(f"Утвердить карточку №{card['request_id']} ({card['category']}, {card['priority']})? [y/n] ")
    return "утверждено" if ans.strip().lower() == "y" else "отклонено, ручная обработка"


# ---------- 1. Демонстрация на трёх характерных примерах ----------
DEMO = [
    {"request_id": 90001, "created_at": "2026-08-20 08:15", "channel": "telegram",
     "text": "Добрый день, на Садовой 12 со вчерашнего вечера из крана вообще ничего не идёт, "
             "в доме пожилые люди, управляйка трубку не берёт. Сделайте что-нибудь!"},
    {"request_id": 90002, "created_at": "2026-08-20 09:40", "channel": "portal",
     "text": "Здравствуйте! Контейнерная площадка переполнена по адресу ул. Полярная, д. 96. "
             "Возле площадки ещё и мусор валяется, дети играют рядом на качелях."},
    {"request_id": 90003, "created_at": "2026-08-20 10:05", "channel": "email", "text": "см. вложение"},
]
demo_out = []
for req in DEMO:
    res = A.process(req, agent)
    show(req, res)
    if res["card"]:
        res["card"]["operator_decision"] = operator_review(res["card"])
    demo_out.append({"request": req, **res})
(A.OUT / f"agent_demo{SUF}.json").write_text(json.dumps(demo_out, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")

# ---------- 2. Проверка на размеченной выборке ----------
ev = pd.read_csv(A.DATA / "eval_set.csv", encoding="utf-8-sig")
if args.limit:
    ev = ev.head(args.limit)
rows, t0 = [], time.time()
for i, req in enumerate(ev.to_dict("records"), 1):
    try:
        res = A.process(req, agent)
    except Exception as e:  # сбой API не должен ронять весь прогон: обращение уходит человеку
        res = {"card": None, "calls": [], "input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "error": str(e)[:200]}
    c = res["card"] or {}
    rows.append({"request_id": req["request_id"], "text": req["text"], "true_category": req["category_final"],
                 "operator_initial": req["category_initial"], "agent_category": c.get("category"),
                 "agent_priority": c.get("priority"), "operator_priority": req["priority"],
                 "confidence": c.get("confidence"), "status": c.get("status", "no_card"),
                 "review_reason": c.get("review_reason", res.get("error", "")),
                 "summary": c.get("summary"), "calls": " -> ".join(res["calls"]),
                 "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                 "llm_calls": res["llm_calls"]})
    print(f"[{i}/{len(ev)}] №{req['request_id']}: {req['category_final']} -> {c.get('category')} ({c.get('status', 'no_card')})")
    if args.pause:
        time.sleep(args.pause)
res_df = pd.DataFrame(rows)
res_df.to_csv(A.OUT / f"agent_eval{SUF}.csv", index=False, encoding="utf-8-sig")

ok_agent = res_df["agent_category"] == res_df["true_category"]
ok_oper = res_df["operator_initial"] == res_df["true_category"]
auto = res_df["status"] == "auto_route"
summary = {
    "mode": MODE, "model": MODEL, "n": len(res_df),
    "agent_accuracy": round(ok_agent.mean(), 3),
    "operator_initial_accuracy": round(ok_oper.mean(), 3),
    "agent_accuracy_on_operator_errors": round(ok_agent[~ok_oper].mean(), 3),
    "auto_route_share": round(auto.mean(), 3),
    "auto_route_accuracy": round(ok_agent[auto].mean(), 3) if auto.any() else None,
    "needs_review_share": round((res_df["status"] == "needs_review").mean(), 3),
    "no_card_share": round((res_df["status"] == "no_card").mean(), 3),
    "agent_high_share": round((res_df["agent_priority"] == "high").mean(), 3),
    "operator_high_share": round((res_df["operator_priority"] == "high").mean(), 3),
    "avg_input_tokens": round(res_df["input_tokens"].mean(), 1),
    "avg_output_tokens": round(res_df["output_tokens"].mean(), 1),
    "avg_llm_calls": round(res_df["llm_calls"].mean(), 2),
    "seconds_total": round(time.time() - t0, 1),
}
errors = res_df[~ok_agent][["request_id", "true_category", "agent_category", "text"]]
summary["errors"] = errors.head(10).to_dict("records")
(A.OUT / f"agent_summary{SUF}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False, indent=2))
