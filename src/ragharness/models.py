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


@dataclass
class ExampleScore:
    id: str
    exact_match: float
    f1: float
    context_precision: float
    context_recall: float
    missing: bool = False
    ragas_score: float = 0.0


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