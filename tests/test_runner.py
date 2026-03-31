from pathlib import Path
from ragharness.runner import evaluate


def test_evaluate():
    dataset = Path("examples/dataset.jsonl")
    preds = Path("examples/predictions_a.jsonl")

    rows, agg = evaluate(dataset, preds)

    assert agg.total == 2
    assert agg.matched == 2
    assert agg.f1 > 0