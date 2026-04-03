import re
from collections import Counter

import re
from rapidfuzz.fuzz import ratio
from rapidfuzz.fuzz import token_set_ratio

# ---------- TEXT NORMALIZATION ----------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    #text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_score(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0

    pred_n = normalize(pred)
    gt_n = normalize(gt)

    base = token_set_ratio(pred_n, gt_n) / 100.0

    # 🔥 Boost if one contains the other (handles verbosity)
    if gt_n in pred_n or pred_n in gt_n:
        base = max(base, 0.85)

    return base


# ---------- EXACT MATCH ----------
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


# ---------- TOKEN F1 ----------
def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# ---------- CONTEXT PRECISION ----------
def context_precision(pred_ctx: list[str], gold_ctx: list[str]) -> float:
    if not pred_ctx:
        return 0.0

    pred_set = {normalize(c) for c in pred_ctx}
    gold_set = {normalize(c) for c in gold_ctx}

    if not pred_set:
        return 0.0

    overlap = len(pred_set & gold_set)
    return overlap / len(pred_set)


# ---------- CONTEXT RECALL ----------
def context_recall(pred_ctx: list[str], gold_ctx: list[str]) -> float:
    if not gold_ctx:
        return 0.0

    pred_set = {normalize(c) for c in pred_ctx}
    gold_set = {normalize(c) for c in gold_ctx}

    if not gold_set:
        return 0.0

    overlap = len(pred_set & gold_set)
    return overlap / len(gold_set)

def ragas_score(pred: str, gt: str, pred_ctx: list[str], gt_ctx: list[str]) -> float:
    f1 = f1_score(pred, gt)
    fuzzy = fuzzy_score(pred, gt)
    recall = context_recall(pred_ctx, gt_ctx)

    # ✅ Smart detection
    has_context = len(pred_ctx) > 0 and len(gt_ctx) > 0

    if has_context:
        score = 0.4 * f1 + 0.4 * fuzzy + 0.2 * recall
    else:
        score = 0.3 * f1 + 0.8 * fuzzy

    # 🔥Length-aware adjustment
    pred_len = len(pred.split())
    gt_len = len(gt.split())

    if pred_len > 0 and gt_len > 0:
        length_ratio = min(pred_len, gt_len) / max(pred_len, gt_len)
        score = score * (0.8 + 0.2 * length_ratio)

    return round(score, 4)