"""Brand rule sets are stored as one JSON file per brand so an admin can review and diff them."""

import os
import re
from pathlib import Path

from .models import Brand

def data_dir() -> Path:
    return Path(os.environ.get("VIDEOQA_DATA_DIR", Path(__file__).resolve().parent.parent / "data" / "brands"))


def _path(name: str, directory: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError("Brand name must contain letters or digits")
    return directory / f"{slug}.json"


def assets_dir(name: str, directory: Path | None = None) -> Path:
    path = _path(name, directory or data_dir())
    return path.parent / f"{path.stem}_assets"


def save_logo(brand: Brand, filename: str, content: bytes, directory: Path | None = None) -> None:
    """Stores a logo image next to the brand's rules and records it on the brand (call save_brand afterwards)."""
    target = assets_dir(brand.name, directory)
    target.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    (target / safe).write_bytes(content)
    if safe not in brand.logo_files:
        brand.logo_files.append(safe)


def save_font(brand: Brand, filename: str, content: bytes, directory: Path | None = None) -> None:
    """Stores a brand font file next to the brand's rules (call save_brand afterwards)."""
    target = assets_dir(brand.name, directory)
    target.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    (target / safe).write_bytes(content)
    if safe not in brand.font_files:
        brand.font_files.append(safe)


def font_paths(brand: Brand, directory: Path | None = None) -> list[Path]:
    base = assets_dir(brand.name, directory)
    return [Path(f) if Path(f).is_absolute() else base / f for f in brand.font_files]


def logo_paths(brand: Brand, directory: Path | None = None) -> list[Path]:
    """Absolute paths of the brand's logos; entries that are already absolute (the bundled samples) are kept."""
    base = assets_dir(brand.name, directory)
    return [Path(f) if Path(f).is_absolute() else base / f for f in brand.logo_files]


def save_brand(brand: Brand, directory: Path | None = None) -> Path:
    path = _path(brand.name, directory or data_dir())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brand.model_dump_json(indent=2) + "\n")
    return path


def load_brand(name: str, directory: Path | None = None) -> Brand:
    return Brand.model_validate_json(_path(name, directory or data_dir()).read_text())


def list_brands(directory: Path | None = None) -> list[str]:
    directory = directory or data_dir()
    if not directory.exists():
        return []
    return sorted(Brand.model_validate_json(p.read_text()).name for p in directory.glob("*.json"))
