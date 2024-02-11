__all__ = ["unzip_file"]
import zipfile
from pathlib import Path
from typing import Optional


def unzip_file(path: Path, dest: Optional[Path] = None, dry_run: bool = False) -> Path:
    if not path.exists():
        raise RuntimeError(f"'{path}' does not exist.")
    if dest is None:
        dest = path.parent / f"{path.stem}"
    elif dest.suffix not in [""]:
        raise RuntimeError(f"'{dest}' is not a directory.")

    if not dry_run:
        dest.mkdir(exist_ok=True, parents=True)

        with zipfile.ZipFile(str(path), 'r') as zip_ref:
            zip_ref.extractall(str(dest))

        print(f"Successfully unzipped '{path}' to '{dest}'.")
    return dest
