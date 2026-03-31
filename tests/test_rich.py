from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Test Table")
table.add_column("Col1", style="cyan")
table.add_column("Col2", style="magenta")
table.add_row("A", "1")
table.add_row("B", "2")
console.print(table)
