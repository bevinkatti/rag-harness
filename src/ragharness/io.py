import json
import csv
from pathlib import Path

from .models import Example, Prediction


def _parse_contexts(value):
    if not value:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value]

    if isinstance(value, str):
        try:
            # Try JSON list
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except:
            pass

        # fallback: split by ||
        return [x.strip() for x in value.split("||")]

    return []


def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_dataset(path: Path) -> list[Example]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = load_jsonl(path)
    elif suffix == ".json":
        rows = json.load(open(path))
    elif suffix == ".csv":
        rows = list(csv.DictReader(open(path)))
    else:
        raise ValueError("Unsupported format")

    return [
        Example(
            id=str(r["id"]),
            question=r["question"],
            answer=r["answer"],
            contexts=_parse_contexts(r.get("contexts")),
        )
        for r in rows
    ]


def load_predictions(path: Path) -> list[Prediction]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = load_jsonl(path)
    elif suffix == ".json":
        rows = json.load(open(path))
    elif suffix == ".csv":
        rows = list(csv.DictReader(open(path)))
    else:
        raise ValueError("Unsupported format")

    return [
        Prediction(
            id=str(r["id"]),
            answer=r["answer"],
            contexts=_parse_contexts(r.get("contexts")),
        )
        for r in rows
    ]