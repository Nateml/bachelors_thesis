from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

def create_progress_bar(dataloader, desc='Training'):
    return tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)

def update_progress_bar(bar, loss):
    bar.set_postfix({
        "loss": f"{loss:.4f}"
    })
    bar.update(1)

def log_epoch_summary(epoch, train_metrics, val_metrics):
    table = Table(title=f"[bold cyan]Epoch {epoch + 1} Results[/bold cyan]", show_lines=True)

    table.add_column("Split", justify="center", style="bold white")
    for metric in train_metrics.keys():
        table.add_column(metric.capitalize(), justify="right")

    def row_from_metrics(name, metrics):
        return [name] + [f"{value:.4f}" if isinstance(value, (int, float)) else str(value) for value in metrics.values()]

    table.add_row(*row_from_metrics("Train", train_metrics))
    table.add_row(*row_from_metrics("Val", val_metrics))

    console.print("\n")
    console.print(table)
    console.print("\n")
