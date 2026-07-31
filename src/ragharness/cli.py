import typer

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from .io import load_predictions
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show per-example evaluation results.",
    ),
    max_examples: int = typer.Option(
        5,
        "--max-examples",
        min=1,
        help="Maximum failed examples to display in verbose mode.",
    ),
):
    """Evaluate a RAG system"""

    rows, agg = run_evaluate(dataset, predictions)
    predictions_data = load_predictions(predictions)
    
    # ✅ Case 1: user provided dataset
    if dataset is not None:
        console.print("[green]dataset Present → running full evaluation[/green]\n")
        pass  # full evaluation

    # ✅ Case 2: auto ground truth detected
    elif any(getattr(p, "ground_truth", "") for p in predictions_data):
        console.print("[cyan]Auto-detected ground truth → running full evaluation[/cyan]\n")

    # ⚠️ Case 3: no ground truth at all
    elif all(not getattr(p, "ground_truth", "") for p in predictions_data):
        console.print(
            "[cyan]⚠ No ground truth found → cannot evaluate answers[/cyan]\n"
            "[yellow]👉 Provide --dataset or include 'ground_truth' in your file[/yellow]\n"
        )

    # ⚠️ Case 4: fallback
    else:
        console.print("[yellow]⚠ Running limited evaluation[/yellow]\n")

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
    table.add_row("Fuzzy Score", f"{agg.fuzzy:.4f}")

    console.print(table)

    # 🔹 Separate RAGAS Highlight Table (nice UX)
    ragas_table = Table(title= None, show_lines=True)
    ragas_table.add_column("Metric", style="bold cyan")
    ragas_table.add_column("Value", justify="right", style="yellow")
    ragas_table.add_row("RAGAS Score", f"{agg.ragas_score:.4f}")

    console.print(ragas_table)
    console.print()

    label, color = performance_label(agg.f1)

    console.print("[bold]Evaluation Summary[/bold]")

    console.print(
        f"Matched predictions : [green]{agg.matched}/{agg.total}[/green]"
    )

    console.print(
        f"Overall quality     : [{color}]{label}[/{color}]"
    )

    if agg.context_precision < 0.70:
        console.print(
            "[yellow]⚠ Context precision could be improved.[/yellow]"
        )

    if agg.context_recall < 0.70:
        console.print(
            "[yellow]⚠ Context recall is relatively low.[/yellow]"
        )

    if agg.exact_match == 1.0:
        console.print(
            "[green]✓ Perfect exact match.[/green]"
        )
        
    if verbose:
        for row in rows:
            if row.missing or row.exact_match < 1.0:
                #new
                details = Text()

                details.append("❓ Question\n", style="bold cyan")
                details.append(f"{row.question}\n\n")

                details.append("🎯 Reference Answer\n", style="bold green")
                details.append(f"{row.expected_answer}\n\n")

                details.append("🤖 Model Answer\n", style="bold yellow")
                details.append(
                    f"{row.predicted_answer or '<missing prediction>'}"
                )

                console.print(
                    Panel(
                        details,
                        title=f"Failed Example (ID: {row.id})",
                        border_style="red",
                        expand=False,
                    )
                )
                #new-end

                console.print()

                console.print(f"Exact Match      : {row.exact_match:.2f}")
                console.print(f"F1 Score         : {row.f1:.2f}")
                console.print(f"Fuzzy Similarity : {row.fuzzy * 100:.0f}%")
                console.print(f"Context Precision: {row.context_precision:.2f}")
                console.print(f"Context Recall   : {row.context_recall:.2f}")
                console.print(f"RAGAS Score      : {row.ragas_score:.2f}")
                console.print()
                
                if row.exact_match == 1:
                    verdict = "[bold green]🟢 Correct[/bold green]"
                elif row.f1 >= 0.6:
                    verdict = "[bold yellow]🟡 Mostly Correct[/bold yellow]"
                else:
                    verdict = "[bold red]🔴 Incorrect[/bold red]"

                console.print(f"[bold]Overall Verdict:[/bold] {verdict}")
                console.print()
                console.print("[bold cyan]💡 Analysis[/bold cyan]")
                for message in analyze_failure(row):
                    console.print(f"• {message}")

                break


def performance_label(score: float) -> tuple[str, str]:
    if score >= 0.90:
        return "Excellent", "green"
    elif score >= 0.75:
        return "Good", "cyan"
    elif score >= 0.60:
        return "Fair", "yellow"
    return "Needs Improvement", "red"

def analyze_failure(row) -> list[str]:
    """
    Generate human-readable insights from evaluation metrics.
    Uses simple heuristics (no LLM/API required).
    """

    analysis = []

    # Exact match failed but answer is still quite similar
    if row.exact_match == 0 and row.f1 >= 0.6:
        analysis.append(
            "⚠ Exact wording differs, but the answer is mostly correct."
        )

    # Completely different answer
    if row.f1 == 0 and row.fuzzy < 0.30:
        analysis.append(
            "❌ Prediction is very different from the expected answer."
        )

    # High lexical similarity
    if row.fuzzy >= 0.90:
        analysis.append(
            "✅ High lexical similarity to the expected answer."
        )

    # Context quality
    if row.context_precision >= 0.80:
        analysis.append(
            "✅ Retrieved context appears relevant."
        )
    elif row.context_precision < 0.50:
        analysis.append(
            "⚠ Retrieved context contains irrelevant information."
        )

    # Context completeness
    if row.context_recall >= 0.80:
        analysis.append(
            "✅ Retrieved context appears sufficiently complete."
        )
    elif row.context_recall < 0.50:
        analysis.append(
            "⚠ Important context may be missing."
        )

    # Overall RAG pipeline quality
    if row.ragas_score >= 0.80:
        analysis.append(
            "🎯 Overall retrieval and answer quality are strong."
        )
    elif row.ragas_score < 0.40:
        analysis.append(
            "⚠ Overall RAG pipeline quality is poor for this example."
        )

    # Fallback
    if not analysis:
        analysis.append(
            "ℹ No obvious issues detected from the available metrics."
        )

    return analysis
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