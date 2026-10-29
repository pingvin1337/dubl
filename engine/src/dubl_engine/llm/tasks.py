"""Задачи ИИ: перевод, укладка по длине, имена персонажей. Плюс оценка длительности речи без ИИ."""
from __future__ import annotations

import re

from ..models import Line, SceneLine, Speaker
from ..util import EngineError
from . import prompts
from .providers import LLMProvider, extract_json

# ─────────────────────────── оценка длительности ───────────────────────────
_VOWELS_RU = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_VOWELS_LAT = set("aeiouyAEIOUY")
_VOWELS_OTHER = set("àáâãäåæèéêëìíîïòóôõöøùúûüýœÀÁÂÃÄÅÆÈÉÊËÌÍÎÏÒÓÔÕÖØÙÚÛÜÝŒіїєІЇЄ")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NUM_RE = re.compile(r"\d+")

# Секунд на слог при живой разговорной озвучке (≈5 слогов/с). Сверено на «Кто трогал мою колбасу?» ≈ 1.85 с.
SYLLABLE_S: dict[str, float] = {"ru": 0.20, "en": 0.19}
_PAUSE_SOFT = set(",;:—–")
_PAUSE_HARD = set(".!?…")
PAUSE_SOFT_S = 0.15
PAUSE_HARD_S = 0.25


def _en_syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ie", "ye")):
        n -= 1                                      # немое конечное e: «make», «sausage»
    if w.endswith("ed") and n > 1 and not w.endswith(("ted", "ded")):
        n -= 1
    return max(n, 1)


def guess_lang(text: str) -> str:
    cyr = sum(1 for c in text if "а" <= c.lower() <= "я" or c in "ёЁіїєІЇЄ")
    lat = sum(1 for c in text if "a" <= c.lower() <= "z")
    return "ru" if cyr >= lat else "en"


def count_syllables(text: str, lang: str = "ru") -> int:
    """Слоги ≈ гласные. Для английского — группы гласных с поправкой на немое e."""
    lang = (lang or "ru").lower().split("-")[0]
    total = 0
    for word in _WORD_RE.findall(text):
        if lang == "en":
            total += _en_syllables(word)
        else:
            total += sum(1 for ch in word if ch in _VOWELS_RU or ch in _VOWELS_LAT or ch in _VOWELS_OTHER)
    total += 3 * len(_NUM_RE.findall(text))       # число читается словами, ~3 слога
    return total


def estimate_duration(text: str, lang: str = "ru") -> float:
    """Сколько секунд займёт произнести текст. Без ИИ: слоги × темп + паузы на знаках препинания."""
    lang = (lang or "ru").lower().split("-")[0]
    rate = SYLLABLE_S.get(lang, SYLLABLE_S["ru"])
    t = count_syllables(text, lang) * rate
    s = text.replace("...", "…")
    for ch in s:
        if ch in _PAUSE_HARD:
            t += PAUSE_HARD_S
        elif ch in _PAUSE_SOFT:
            t += PAUSE_SOFT_S
    return round(t, 3)


def line_duration(line: SceneLine) -> float:
    return max(0.0, line.end_rel - line.start_rel)


# ─────────────────────────── перевод ───────────────────────────
BATCH_SIZE = 25
CONTEXT_LINES = 3


def _items_for(lines: list[SceneLine], base: int) -> list[dict]:
    return [
        {"id": base + i, "speaker": ln.speaker_id or "", "duration": round(line_duration(ln), 1),
         "text": ln.original or ln.text}
        for i, ln in enumerate(lines)
    ]


def _parse_translations(raw: str) -> dict[int, str]:
    data = extract_json(raw)
    items = data.get("translations") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise EngineError("Модель вернула перевод в неожиданном формате.")
    out: dict[int, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("id"))
        except (TypeError, ValueError):
            continue
        text = it.get("text")
        if isinstance(text, str) and text.strip():
            out[idx] = text.strip()
    return out


async def translate_lines(lines: list[SceneLine], target_lang: str, provider: LLMProvider) -> list[str]:
    """Перевести реплики на target_lang. Порядок и количество сохраняются; результат того же размера, что вход."""
    if not lines:
        return []
    result: list[str | None] = [None] * len(lines)
    for start in range(0, len(lines), BATCH_SIZE):
        chunk = lines[start:start + BATCH_SIZE]
        context = _items_for(lines[max(0, start - CONTEXT_LINES):start], max(0, start - CONTEXT_LINES)) if start else None
        pending = list(range(len(chunk)))
        for _attempt in range(2):                       # повторяем один раз только недостающие
            items = [_items_for([chunk[i]], start + i)[0] for i in pending]
            system, user = prompts.translate_prompt(items, target_lang, context)
            raw = await provider.complete(system, user, json_schema=prompts.TRANSLATE_SCHEMA,
                                          max_tokens=min(8000, 400 + 120 * len(items)), temperature=0.3)
            got = _parse_translations(raw)
            for i in list(pending):
                text = got.get(start + i)
                if text is not None:
                    result[start + i] = text
                    pending.remove(i)
            if not pending:
                break
        if pending:
            raise EngineError(
                f"Модель перевела не все реплики (пропущено {len(pending)} из {len(chunk)}). Нажми «Перевести» ещё раз.")
    return [r or "" for r in result]


# ─────────────────────────── укладка по длине ───────────────────────────
FIT_TOLERANCE = 1.1


async def fit_line(line: SceneLine, provider: LLMProvider, *, n: int = 3) -> list[str]:
    """До n вариантов текста реплики, которые влезают в её длительность (оценка — estimate_duration)."""
    duration = line_duration(line)
    if duration <= 0 or not line.text.strip():
        return []
    lang = guess_lang(line.text)
    limit = duration * FIT_TOLERANCE
    rate = SYLLABLE_S.get(lang, 0.2)
    syllables = max(1, int((duration - 0.25) / rate))
    chars = int(syllables * 2.3)
    ask = n + 2                                          # запас: часть вариантов отсеется
    good: list[str] = []
    too_long: list[str] = []
    seen: set[str] = set()
    for _attempt in range(2):
        system, user = prompts.fit_prompt(line.text, line.original, duration, ask, syllables, chars,
                                          too_long=too_long or None)
        raw = await provider.complete(system, user, json_schema=prompts.FIT_SCHEMA, max_tokens=600, temperature=0.7)
        data = extract_json(raw)
        variants = data.get("variants") if isinstance(data, dict) else data
        if not isinstance(variants, list):
            raise EngineError("Модель вернула варианты в неожиданном формате.")
        too_long = []
        for v in variants:
            if not isinstance(v, str):
                continue
            v = v.strip().strip("«»\"")
            key = v.casefold()
            if not v or key in seen:
                continue
            seen.add(key)
            if estimate_duration(v, lang) <= limit:
                good.append(v)
            else:
                too_long.append(v)
        if good:
            break
    good.sort(key=lambda v: abs(estimate_duration(v, lang) - duration))
    return good[:n]


# ─────────────────────────── имена персонажей ───────────────────────────
MIN_NAME_CONFIDENCE = 0.7
MAX_DIALOGUE_LINES = 400


def _name_in_dialogue(name: str, dialogue_cf: str) -> bool:
    cf = name.casefold()
    stem = cf[: max(3, len(cf) - 2)] if len(cf) > 4 else cf   # «Барсику» ~ «Барсик»
    return stem in dialogue_cf


async def name_speakers(lines: list[Line], speakers: list[Speaker], provider: LLMProvider) -> dict[str, str]:
    """Угадать имена по обращениям в диалогах. Возвращает только уверенные {speaker_id: имя}."""
    todo = {s.id: s for s in speakers if not s.name}
    if not todo or not lines:
        return {}
    ordered = sorted(lines, key=lambda ln: ln.start)[:MAX_DIALOGUE_LINES]
    dialogue = "\n".join(f"[{ln.speaker_id or '?'}] {ln.text.strip()[:240]}" for ln in ordered if ln.text.strip())
    if not dialogue:
        return {}
    desc = "\n".join(
        f"{s.id} — {s.label}" + (f" (имя уже известно: {s.name}, не включай)" if s.name else "")
        for s in speakers)
    system, user = prompts.names_prompt(dialogue, desc)
    raw = await provider.complete(system, user, json_schema=prompts.NAMES_SCHEMA, max_tokens=800, temperature=0.1)
    data = extract_json(raw)
    items = data.get("names") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise EngineError("Модель вернула имена в неожиданном формате.")
    dialogue_cf = dialogue.casefold()
    found: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sid, name = it.get("speaker_id"), it.get("name")
        try:
            conf = float(it.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        if sid not in todo or not isinstance(name, str):
            continue
        name = name.strip()
        if not name or len(name) > 40 or conf < MIN_NAME_CONFIDENCE or not _name_in_dialogue(name, dialogue_cf):
            continue
        found[sid] = name
    counts: dict[str, int] = {}
    for nm in found.values():
        counts[nm.casefold()] = counts.get(nm.casefold(), 0) + 1
    return {sid: nm for sid, nm in found.items() if counts[nm.casefold()] == 1}
