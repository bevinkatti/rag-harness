from ragharness.metrics import exact_match, f1_score, context_precision, context_recall


def test_exact_match():
    assert exact_match("Paris", "Paris") == 1.0
    assert exact_match("The Paris", "Paris") == 1.0
    assert exact_match("London", "Paris") == 0.0


def test_f1():
    assert f1_score("William Shakespeare", "Shakespeare") > 0.5
    assert f1_score("abc", "xyz") == 0.0


def test_context():
    pred = ["Paris is capital of France"]
    gold = ["Paris is capital of France"]

    assert context_precision(pred, gold) == 1.0
    assert context_recall(pred, gold) == 1.0