__all__ = ["SystemPath"]
from pathlib import Path

SystemPath: Path = Path(__file__).absolute().parent.parent.parent