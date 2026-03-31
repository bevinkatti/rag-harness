from pathlib import Path

from .io import load_dataset, load_predictions
from .metrics import exact_match, f1_score, context_precision, context_recall, ragas_score
from .models import ExampleScore, AggregateScore


def evaluate(dataset_path: Path, predictions_path: Path):
    dataset = load_dataset(dataset_path)
    predictions = load_predictions(predictions_path)

    pred_map = {p.id: p for p in predictions}

    rows = []

    for ex in dataset:
        pred = pred_map.get(ex.id)

        if pred is None:
            rows.append(
                ExampleScore(
                    id=ex.id,
                    exact_match=0.0,
                    f1=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                    ragas_score=0.0,
                    missing=True,
                )
            )
            continue

        ragas = ragas_score(pred.answer, ex.answer, pred.contexts, ex.contexts)
        
        rows.append(
            ExampleScore(
                id=ex.id,
                exact_match=exact_match(pred.answer, ex.answer),
                f1=f1_score(pred.answer, ex.answer),
                context_precision=context_precision(pred.contexts, ex.contexts),
                context_recall=context_recall(pred.contexts, ex.contexts),
                ragas_score=ragas,
                missing=False,
            )
        )

    total = len(rows)
    matched = sum(1 for r in rows if not r.missing)
    missing = total - matched

    def avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    aggregate = AggregateScore(
        total=total,
        matched=matched,
        missing=missing,
        exact_match=avg([r.exact_match for r in rows]),
        f1=avg([r.f1 for r in rows]),
        context_precision=avg([r.context_precision for r in rows]),
        context_recall=avg([r.context_recall for r in rows]),
        ragas_score=avg([r.ragas_score for r in rows]),
    )

    return rows, aggregate