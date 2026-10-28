"""Brand rule sets are stored as one JSON file per brand so an admin can review and diff them."""

import os
import re
from pathlib import Path

from .models import Brand

DEFAULT_DIR = Path(os.environ.get("VIDEOQA_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "brands"))


def _path(name: str, directory: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError("Brand name must contain letters or digits")
    return directory / f"{slug}.json"


def save_brand(brand: Brand, directory: Path = DEFAULT_DIR) -> Path:
    path = _path(brand.name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brand.model_dump_json(indent=2) + "\n")
    return path


def load_brand(name: str, directory: Path = DEFAULT_DIR) -> Brand:
    return Brand.model_validate_json(_path(name, directory).read_text())


def list_brands(directory: Path = DEFAULT_DIR) -> list[str]:
    if not directory.exists():
        return []
    return sorted(Brand.model_validate_json(p.read_text()).name for p in directory.glob("*.json"))
