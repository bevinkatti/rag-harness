import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .compare import compare_models
from .runner import evaluate as run_evaluate

app = typer.Typer(help="⚡ RAG Harness CLI", invoke_without_command=True)
console = Console()


@app.callback()
def main():
    """RAG Harness CLI"""
    pass


# -------------------------------
# EVALUATE COMMAND
# -------------------------------
@app.command()
def evaluate(
    predictions: Path = typer.Argument(
        ...,
        exists=True,
        help="Predictions file"
    ),
    dataset: Path = typer.Option(
        None,
        exists=True,
        help="Optional dataset file"
    ),
):
    """Evaluate a RAG system"""

    rows, agg = run_evaluate(dataset, predictions)
    if dataset is None:
        console.print("[yellow] No dataset → limited metric Evaluation [/yellow]\n")

    # 🔹 Main Evaluation Table
    table = Table(title="📊 RAG Evaluation Summary", show_lines=True)

    table.add_column("Metric", style="bold magenta")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total", str(agg.total))
    table.add_row("Matched", str(agg.matched))
    table.add_row("Missing", str(agg.missing))
    table.add_row("Exact Match", f"{agg.exact_match:.4f}")
    table.add_row("F1 Score", f"{agg.f1:.4f}")
    table.add_row("Context Precision", f"{agg.context_precision:.4f}")
    table.add_row("Context Recall", f"{agg.context_recall:.4f}")

    console.print(table)

    # 🔹 Separate RAGAS Highlight Table (nice UX)
    ragas_table = Table(title= None, show_lines=True)
    ragas_table.add_column("Metric", style="bold cyan")
    ragas_table.add_column("Value", justify="right", style="yellow")
    ragas_table.add_row("RAGAS Score", f"{agg.ragas_score:.4f}")

    console.print(ragas_table)



# -------------------------------
# COMPARE COMMAND (ADD HERE)
# -------------------------------
@app.command()
def compare(
    dataset: Path = typer.Argument(..., exists=True),
    pred_a: Path = typer.Argument(..., help="Predictions A"),
    pred_b: Path = typer.Argument(..., help="Predictions B"),
):
    """Compare two RAG systems"""

    compare_models(dataset, pred_a, pred_b)


if __name__ == "__main__":
    app()