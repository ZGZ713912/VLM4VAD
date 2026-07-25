"""Prompt normalization for common anomaly descriptions."""

from __future__ import annotations

from typing import List

NORMAL_PROMPT = "normal"

_ANOMALY_ALIASES = {
    "abuse": "abuse",
    "arrest": "arrest",
    "arson": "arson",
    "assault": "assault",
    "bomb": "explosion",
    "car accident": "car accident",
    "car crash": "car accident",
    "crime": "crime",
    "explosion": "explosion",
    "fight": "fighting",
    "fighting": "fighting",
    "robbery": "robbery",
    "riot": "riot",
    "shooting": "shooting",
    "stealing": "stealing",
    "vandalism": "vandalism",
    "打架": "fighting",
    "斗殴": "fighting",
    "暴乱": "riot",
    "虐待": "abuse",
    "车祸": "car accident",
    "爆炸": "explosion",
    "纵火": "arson",
    "抢劫": "robbery",
    "偷窃": "stealing",
    "盗窃": "stealing",
    "破坏公物": "vandalism",
    "逮捕": "arrest",
    "枪击": "shooting",
}

_PROMPT_TEMPLATES = (
    "{text}",
    "a video of {text}",
    "an abnormal video of {text}",
)


def normalize_anomaly_text(text: str) -> str:
    cleaned = " ".join(text.strip().lower().split())
    return _ANOMALY_ALIASES.get(cleaned, cleaned)


def build_anomaly_prompt_variants(text: str) -> List[str]:
    canonical = normalize_anomaly_text(text)
    variants = [template.format(text=canonical) for template in _PROMPT_TEMPLATES]
    ordered: List[str] = []
    for variant in variants:
        if variant not in ordered:
            ordered.append(variant)
    return ordered


def build_prompt_pairs(text: str) -> List[List[str]]:
    return [[NORMAL_PROMPT, variant] for variant in build_anomaly_prompt_variants(text)]
