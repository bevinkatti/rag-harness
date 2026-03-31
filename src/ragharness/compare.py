from pathlib import Path
from rich.table import Table
from rich.console import Console

from .runner import evaluate

console = Console()


def compare_models(dataset: Path, pred_a: Path, pred_b: Path):
    rows_a, agg_a = evaluate(dataset, pred_a)
    rows_b, agg_b = evaluate(dataset, pred_b)

    table = Table(title="⚔️ RAG Systems Comparison", show_lines=True)

    table.add_column("Metric", style="bold magenta")
    table.add_column("System A", style="green")
    table.add_column("System B", style="cyan")

    def row(name, a, b):
        table.add_row(name, f"{a:.4f}", f"{b:.4f}")

    row("Exact Match", agg_a.exact_match, agg_b.exact_match)
    row("F1 Score", agg_a.f1, agg_b.f1)
    row("Context Precision", agg_a.context_precision, agg_b.context_precision)
    row("Context Recall", agg_a.context_recall, agg_b.context_recall)
    row("RAGAS Score", agg_a.ragas_score, agg_b.ragas_score)

    console.print(table)

    # 🏆 Winner logic
    score_a = agg_a.f1 + agg_a.ragas_score
    score_b = agg_b.f1 + agg_b.ragas_score

    if score_a > score_b:
        console.print("\n🏆 [bold green]System A wins[/bold green]")
    elif score_b > score_a:
        console.print("\n🏆 [bold cyan]System B wins[/bold cyan]")
    else:
        console.print("\n🤝 It's a tie")