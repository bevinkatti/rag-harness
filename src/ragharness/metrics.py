import re
from collections import Counter


# ---------- TEXT NORMALIZATION ----------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

def ragas_score(pred: str, gold: str, pred_ctx: list[str], gold_ctx: list[str]) -> float:
    """
    Lightweight RAGAS-style score
    Combines:
    - Answer F1
    - Context Recall
    """

    f1 = f1_score(pred, gold)
    cr = context_recall(pred_ctx, gold_ctx)

    return round((f1 * 0.6 + cr * 0.4), 4)