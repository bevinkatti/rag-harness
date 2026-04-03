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
    id_val = str(row.get("id") or row.get("question_id") or idx)

    # 🔥 Detect answer
    answer = (
        row.get("answer")
        or row.get("generated_answer")
        or row.get("result")
        or row.get("response")
        or ""
    )

    # 🔥 Detect ground truth
    ground_truth = (
        row.get("ground_truth")
        or row.get("answer_gt")
        or row.get("expected_answer")
        or ""
    )

    # 🔥 Detect contexts
    context_candidates = [
        "contexts",
        "context",
        "documents",
        "retrieved_docs",
        "retrieved_documents",
        "source_documents",
        "chunks",
        "sources",
        "docs",
    ]

    contexts = []

    for key in context_candidates:
        if key in row and row[key]:
            val = row[key]

            if isinstance(val, list):
                for v in val:
                    if isinstance(v, str):
                        contexts.append(v.strip())
                    elif isinstance(v, dict):
                        contexts.append(
                            v.get("page_content")
                            or v.get("text")
                            or str(v)
                        )

            elif isinstance(val, str):
                contexts.extend([x.strip() for x in val.split("||")])

    # 🔥 Clean contexts
    contexts = [c for c in contexts if c]
    contexts = list(dict.fromkeys(contexts))

    return {
        "id": id_val,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def load_predictions(path: Path) -> list[Prediction]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = load_jsonl(path)
        
    elif suffix == ".json":
        data = json.load(open(path))

        # 🔥 Handle benchmark_report.json format
        if isinstance(data, dict) and "questions" in data:
            rows = data["questions"]

        # Normal list of dicts
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("Unsupported JSON format")
    
        
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
                ground_truth=converted.get("ground_truth", ""),
            )
        )

    return predictions