from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    id: str
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    id: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    ground_truth: str = ""

@dataclass
class ExampleScore:
    id: str

    # Original data
    question: str = ""
    expected_answer: str = ""
    predicted_answer: str = ""

    # Evaluation metrics
    exact_match: float = 0.0
    f1: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    ragas_score: float = 0.0
    fuzzy: float = 0.0
    
    # Status
    missing: bool = False 


@dataclass
class AggregateScore:
    total: int
    matched: int
    missing: int
    exact_match: float
    f1: float
    context_precision: float
    context_recall: float
    ragas_score: float
    fuzzy: float = 0.0