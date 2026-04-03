from pathlib import Path

from .io import load_dataset, load_predictions
from .metrics import exact_match, f1_score, context_precision, context_recall, ragas_score, fuzzy_score
from .models import Example, ExampleScore, AggregateScore


def evaluate(dataset_path: Path | None, predictions_path: Path):
    predictions = load_predictions(predictions_path)
    if dataset_path is not None:
        dataset = load_dataset(dataset_path)
    else:
        dataset = []
        
        if predictions and predictions[0].ground_truth:
            
            dataset = [
                
                Example(
                    id=p.id,
                    question="",
                    answer=p.ground_truth,
                    contexts=[]
                )

                for p in predictions
            ]

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

        fuzzy = fuzzy_score(pred.answer, ex.answer)
        #print("FUZZY DEBUG:", fuzzy) 
        ragas = ragas_score(pred.answer, ex.answer, pred.contexts, ex.contexts)
        
        rows.append(
            ExampleScore(
                id=ex.id,
                exact_match=exact_match(pred.answer, ex.answer),
                f1=f1_score(pred.answer, ex.answer),
                context_precision=context_precision(pred.contexts, ex.contexts),
                context_recall=context_recall(pred.contexts, ex.contexts),
                ragas_score=ragas,
                fuzzy=fuzzy,
                missing=False,
            )
        )
    if not dataset:
        return [], AggregateScore(
            total=len(predictions),
            matched=0,
            missing=0,
            exact_match=0.0,
            f1=0.0,
            context_precision=0.0,
            context_recall=0.0,
            ragas_score=0.0,
            fuzzy=0.0,
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
        fuzzy=avg([r.fuzzy for r in rows]),
    )

    return rows, aggregate