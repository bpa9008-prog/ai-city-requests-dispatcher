"""Этап 5. AI-агент первичной обработки обращений граждан («диспетчер-сортировщик»).

Что делает агент с одним обращением:
  1. читает свободный текст, определяет категорию, приоритет и адрес;
  2. lookup_address  — по улице находит район и ищет похожие обращения по тому же адресу
                       за 14 дней (возможный дубль);
  3. get_routing_and_sla — по категории и приоритету получает ответственную службу и
                       типичный срок решения из исторических данных (data/sla_reference.csv);
  4. submit_triage_card — сохраняет карточку: категория, приоритет, район, краткое содержание,
                       уверенность и ЧЕРНОВИК ответа гражданину.
Карточка ничего не отправляет сама: она попадает в очередь со статусом, который определяет
правило human-in-the-loop (см. decide_status). Ответ гражданину всегда утверждает оператор.

Режимы:
  llm     — LangChain create_agent + openai/gpt-oss-120b (Groq, OpenAI-совместимый API),
            нужен ключ в переменной окружения LLM_API_KEY;
  offline — детерминированная замена LLM (ключевые слова) для проверки инструментов без ключа.
"""
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from langchain.tools import tool

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"

JOURNAL = pd.read_csv(DATA / "clean_requests.csv", parse_dates=["created_at"], encoding="utf-8-sig")
SLA = pd.read_csv(DATA / "sla_reference.csv", encoding="utf-8-sig")

CATS = ["road", "lighting", "water", "heating", "garbage", "yard", "other"]
DEPARTMENTS = {
    "road": "Дорожная служба (ГБУ «Автодороги»)",
    "lighting": "Служба наружного освещения",
    "water": "Водоканал, аварийно-диспетчерская служба",
    "heating": "Теплосеть / управляющая компания",
    "garbage": "Региональный оператор по вывозу ТКО",
    "yard": "Отдел благоустройства района",
    "other": "Администрация района (общий отдел)",
}
STREET_TO_DISTRICT = {
    **dict.fromkeys(["Ленина", "Садовая", "Невская", "Пушкинская", "Театральная"], "Центральный"),
    **dict.fromkeys(["Северная", "Полярная", "Лесная", "Озёрная", "Берёзовая"], "Северный"),
    **dict.fromkeys(["Южная", "Степная", "Вокзальная", "Заводская", "Рабочая"], "Южный"),
    **dict.fromkeys(["Западная", "Морская", "Портовая", "Речная", "Набережная"], "Западный"),
    **dict.fromkeys(["Восточная", "Солнечная", "Новая", "Молодёжная", "Парковая"], "Восточный"),
}
CARDS: dict[int, dict] = {}   # очередь карточек: request_id -> карточка
CURRENT: dict = {}            # контекст обрабатываемого обращения (id, дата)


# ---------------------------------------------------------------- инструменты
@tool(parse_docstring=True)
def lookup_address(street: str, house: int, category: str) -> str:
    """Находит район по улице и ищет похожие обращения по тому же адресу за последние 14 дней.

    Используй ОДИН раз для каждого обращения, после того как определил улицу, дом и категорию.
    Если адреса в тексте нет, передай street="" и house=0.

    Args:
        street: название улицы без «ул.», например «Ленина».
        house: номер дома числом; 0, если не указан.
        category: предполагаемая категория: road, lighting, water, heating, garbage, yard или other.
    """
    street = street.replace("ул.", "").strip()
    district = STREET_TO_DISTRICT.get(street, "Не указан")
    now = CURRENT.get("created_at", datetime.now())
    win = JOURNAL[(JOURNAL["street"] == street) & (JOURNAL["house"] == house)
                  & (JOURNAL["created_at"] < now) & (JOURNAL["created_at"] >= now - timedelta(days=14))
                  & (JOURNAL["request_id"] != CURRENT.get("request_id"))]
    same = win[win["category_final"].fillna(win["category_initial"]) == category]
    CURRENT["duplicate_ids"] = same["request_id"].tolist()   # для правила human-in-the-loop
    return json.dumps({
        "district": district,
        "similar_last_14d": int(len(same)),
        "similar_ids": same["request_id"].tolist()[:5],
        "other_requests_same_address": int(len(win) - len(same)),
    }, ensure_ascii=False)


@tool(parse_docstring=True)
def get_routing_and_sla(category: str, priority: str) -> str:
    """Возвращает ответственную службу и типичный срок решения по историческим данным.

    Используй ОДИН раз, когда категория и приоритет уже определены.

    Args:
        category: road, lighting, water, heating, garbage, yard или other.
        priority: low, medium или high.
    """
    if category not in DEPARTMENTS:
        return f"Ошибка: неизвестная категория {category!r}. Допустимо: {', '.join(CATS)}."
    row = SLA[(SLA["category_final"] == category) & (SLA["priority"] == priority)]
    if row.empty:
        row = SLA[SLA["category_final"] == category]
    return json.dumps({
        "department": DEPARTMENTS[category],
        "typical_hours_median": float(row["median_h"].median()),
        "hours_p80": float(row["p80_h"].median()),
        "based_on_requests": int(row["n"].sum()),
    }, ensure_ascii=False)


@tool(parse_docstring=True)
def submit_triage_card(category: str, priority: str, district: str, street: str, house: int,
                       summary: str, priority_reason: str, confidence: float, reply_draft: str) -> str:
    """Сохраняет итоговую карточку обращения в очередь на проверку оператором. Ничего не отправляет.

    Вызывай ПОСЛЕДНИМ шагом, ровно один раз, после lookup_address и get_routing_and_sla.

    Args:
        category: road, lighting, water, heating, garbage, yard или other.
        priority: low, medium или high.
        district: район из ответа lookup_address.
        street: улица без «ул.»; пустая строка, если нет.
        house: номер дома; 0, если нет.
        summary: суть проблемы одной фразой до 15 слов, без эмоций.
        priority_reason: почему выбран такой приоритет, 5–15 слов.
        confidence: уверенность в категории от 0 до 1.
        reply_draft: вежливый черновик ответа гражданину, 2–3 предложения, со сроком из get_routing_and_sla.
    """
    rid = CURRENT.get("request_id")
    card = {"request_id": rid, "category": category, "priority": priority, "district": district,
            "street": street, "house": house, "summary": summary, "priority_reason": priority_reason,
            "confidence": round(float(confidence), 2), "reply_draft": reply_draft,
            "department": DEPARTMENTS.get(category, "?")}
    card["status"], card["review_reason"] = decide_status(card, CURRENT.get("duplicate_ids", []))
    CARDS[rid] = card
    return f"Карточка сохранена, статус: {card['status']}. Отправка только после решения оператора."


TOOLS = [lookup_address, get_routing_and_sla, submit_triage_card]


def decide_status(card: dict, duplicate_ids: list) -> tuple[str, str]:
    """Правило human-in-the-loop. Решает КОД, а не модель: модель не может себе разрешить отправку."""
    reasons = []
    if card["category"] not in CATS:
        reasons.append("категория вне справочника")
    if card["confidence"] < 0.8:
        reasons.append(f"низкая уверенность {card['confidence']}")
    if card["priority"] == "high":
        reasons.append("высокий приоритет — диспетчер подтверждает выезд")
    if card["district"] == "Не указан":
        reasons.append("адрес не распознан")
    if duplicate_ids:
        reasons.append(f"возможный дубль {duplicate_ids[:3]}")
    if reasons:
        return "needs_review", "; ".join(reasons)
    return "auto_route", "маршрут в службу автоматически; ответ гражданину — после утверждения оператором"


# ---------------------------------------------------------------- режим LLM
SYSTEM_PROMPT = """
Ты — диспетчер-сортировщик городской службы обращений граждан. На вход — текст одного обращения.
Твоя задача — подготовить карточку для оператора, а не отвечать в чате.

Порядок работы (строго):
1. Определи ОСНОВНУЮ проблему (то, о чём обращение в первую очередь; попутные жалобы не меняют категорию)
   и её категорию:
   road — дороги, ямы, тротуары, люки на проезжей части;
   lighting — уличное освещение, фонари, провода освещения;
   water — холодная/горячая вода, трубы, протечки водопровода, вода в подвале;
   heating — отопление, батареи;
   garbage — вывоз мусора, контейнеры, свалки;
   yard — благоустройство двора: площадки, трава, деревья, скамейки, ограды;
   other — всё остальное (шум, животные, лифт, торговля, граффити).
2. Определи приоритет:
   high — угроза жизни и здоровью, дети в опасности, несчастный случай, прорыв трубы, открытый люк,
          аварийное дерево, нет воды или отопления в доме;
   medium — неисправность, мешающая многим жителям, без прямой угрозы;
   low — неудобство, эстетика, плановые работы.
3. Вызови lookup_address (улица, дом, категория).
4. Вызови get_routing_and_sla (категория, приоритет).
5. Вызови submit_triage_card. В reply_draft не обещай сроков короче, чем typical_hours_median,
   не выдумывай фактов, не упоминай внутренние коды категорий.
Если текст пустой или непонятный — category="other", confidence не выше 0.3.
После submit_triage_card ответь одной строкой: «Готово».
"""


def build_llm_agent():
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["LLM_API_KEY"],
                     model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
                     temperature=0, max_retries=6)
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def run_llm(agent, req: dict) -> dict:
    msg = (f"Обращение №{req['request_id']}, получено {req['created_at']}, канал {req['channel']}.\n"
           f"Текст: «{req['text']}»")
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config={"recursion_limit": 12})
    calls, tin, tout = [], 0, 0
    for m in result["messages"]:
        for c in getattr(m, "tool_calls", []) or []:
            calls.append(c["name"])
        u = getattr(m, "usage_metadata", None)
        if u:
            tin += u.get("input_tokens", 0)
            tout += u.get("output_tokens", 0)
    return {"calls": calls, "input_tokens": tin, "output_tokens": tout, "llm_calls": sum(
        1 for m in result["messages"] if m.__class__.__name__ == "AIMessage")}


# ---------------------------------------------------------------- режим offline
KEYWORDS = {
    "road": ["яма", "ямы", "асфальт", "тротуар", "проезж", "люк"],
    "lighting": ["фонар", "освещен"],
    "water": ["вод", "труб", "напор", "кран"],
    "heating": ["батаре", "отоплен", "холодно"],
    "garbage": ["мусор", "контейнер", "свалк"],
    "yard": ["качел", "площадк", "трава", "дерев", "скамейк", "оград"],
    "other": ["сосед", "собак", "лифт", "торгов", "граффити"],
}
HIGH_RE = re.compile(r"опасно|несчастн|угроза|прорыв|открытый люк|аварийн|нет холодной|нет горячей|не включили", re.I)


def offline_classify(text: str) -> tuple[str, float]:
    first = re.split(r"(?<=\d)\.", text, maxsplit=1)[0].lower()   # основная проблема — до адреса
    scores = {c: sum(first.count(k) for k in kws) for c, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other", 0.3
    return best, 0.9 if sorted(scores.values())[-2] == 0 else 0.6


def run_offline(req: dict) -> dict:
    text = req["text"]
    cat, conf = offline_classify(text)
    pr = "high" if HIGH_RE.search(text) else "medium" if re.search(r"не работа|не горят|слабый|течёт", text, re.I) else "low"
    m = re.search(r"ул\.\s*([А-ЯЁ][а-яё]+),\s*д\.\s*(\d+)", text)
    street, house = (m.group(1), int(m.group(2))) if m else ("", 0)
    addr = json.loads(lookup_address.invoke({"street": street, "house": house, "category": cat}))
    sla = json.loads(get_routing_and_sla.invoke({"category": cat, "priority": pr}))
    days = max(1, round(sla["typical_hours_median"] / 24))
    summary = re.sub(r"^(Здравствуйте!|Добрый день\.|Прошу помочь\.|Уважаемые сотрудники!)\s*", "", text).split(" по адресу")[0]
    submit_triage_card.invoke({
        "category": cat, "priority": pr, "district": addr["district"], "street": street, "house": house,
        "summary": summary[:120], "priority_reason": "найдены признаки угрозы" if pr == "high" else "нет признаков угрозы",
        "confidence": conf,
        "reply_draft": (f"Здравствуйте! Ваше обращение зарегистрировано и передано: {sla['department']}. "
                        f"Обычно такие вопросы решаются примерно за {days} дн. Мы сообщим о результате."),
    })
    return {"calls": ["lookup_address", "get_routing_and_sla", "submit_triage_card"],
            "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}


def process(req: dict, agent=None) -> dict:
    CURRENT.clear()
    CURRENT.update(request_id=int(req["request_id"]), created_at=pd.Timestamp(req["created_at"]).to_pydatetime())
    meta = run_offline(req) if agent is None else run_llm(agent, req)
    return {"card": CARDS.get(int(req["request_id"])), **meta}
