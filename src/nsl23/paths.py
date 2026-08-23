"""Portable path helpers for manifests that are stored relative to the repository."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a manifest path without depending on the machine that created it."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def portable_path(value: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Write paths inside the repository as POSIX-style repository-relative paths."""
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
