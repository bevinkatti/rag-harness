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


def detect_format(row: dict):
    if "result" in row and "source_documents" in row:
        return "langchain"

    if "response" in row and "contexts" in row:
        return "llamaindex"

    if "answer" in row:
        return "standard"

    return "generic"


def convert_row(row: dict, idx: int):
    fmt = detect_format(row)

    id_val = str(row.get("id", idx))

    # LangChain
    if fmt == "langchain":
        return {
            "id": id_val,
            "answer": row.get("result", ""),
            "contexts": [
                doc.get("page_content", "")
                for doc in row.get("source_documents", [])
                if isinstance(doc, dict)
            ],
        }

    # LlamaIndex
    if fmt == "llamaindex":
        return {
            "id": id_val,
            "answer": row.get("response", ""),
            "contexts": row.get("contexts", []),
        }

    # Standard
    if fmt == "standard":
        return {
            "id": id_val,
            "answer": row.get("answer", ""),
            "contexts": row.get("contexts", []),
        }

    # Generic fallback (🔥 important)
    return {
        "id": id_val,
        "answer": row.get("answer") or row.get("result") or row.get("response", ""),
        "contexts": row.get("contexts") or row.get("documents") or [],
    }


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

    predictions = []

    for idx, row in enumerate(rows):
        converted = convert_row(row, idx)

        predictions.append(
            Prediction(
                id=converted["id"],
                answer=converted["answer"],
                contexts=converted["contexts"],
            )
        )

    return predictions