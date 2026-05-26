"""Shared helpers for NC50 ablation scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


NC50_SUFFIX = "_50_nc.json"
PLAN_TYPE_ORDER = (
    "Mi1",
    "Mi10",
    "Mi4",
    "Mi9",
    "Tm1",
    "Tm2",
    "Tm20",
    "Tm3",
    "Tm5a",
    "Tm5b",
    "Tm5c",
    "Tm9",
    "TmY5a",
)

TARGET_DSI_CELL_TYPES = (
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
)


def repo_root() -> Path:
    """Return the repository root for this script directory."""
    return Path(__file__).resolve().parents[1]


def default_connectome_dir() -> Path:
    """Return the package connectome directory."""
    return repo_root() / "flyvis" / "connectome"


def _fallback_sort_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def sort_nc50_types(types: Iterable[str]) -> list[str]:
    """Sort known NC50 types in the documented order, then unknowns naturally."""
    seen = set(types)
    ordered = [cell_type for cell_type in PLAN_TYPE_ORDER if cell_type in seen]
    ordered.extend(sorted(seen.difference(PLAN_TYPE_ORDER), key=_fallback_sort_key))
    return ordered


def discover_nc50_types(connectome_dir: Path | str | None = None) -> list[str]:
    """Discover ablation types from ``*_50_nc.json`` files."""
    connectome_dir = Path(connectome_dir or default_connectome_dir())
    return sort_nc50_types(
        path.name[: -len(NC50_SUFFIX)]
        for path in connectome_dir.glob(f"*{NC50_SUFFIX}")
        if path.is_file()
    )


def parse_type_selection(
    selected: Iterable[str] | None,
    connectome_dir: Path | str | None = None,
) -> list[str]:
    """Resolve a CLI type selection, where ``all`` expands to NC50 files."""
    if selected is None:
        return discover_nc50_types(connectome_dir)

    selected = list(selected)
    if not selected or any(value.lower() == "all" for value in selected):
        return discover_nc50_types(connectome_dir)
    return sort_nc50_types(selected)


def load_connectome_spec(path: Path | str) -> dict:
    """Load a connectome JSON specification."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_node_stride(spec: dict, cell_type: str) -> tuple[int, int]:
    """Extract the stride pattern for a cell type from a connectome spec."""
    for node in spec.get("nodes", []):
        if node.get("name") != cell_type:
            continue

        pattern = node.get("pattern")
        if (
            not isinstance(pattern, list)
            or len(pattern) != 2
            or pattern[0] != "stride"
            or len(pattern[1]) != 2
        ):
            raise ValueError(f"{cell_type} does not use a stride pattern: {pattern}")

        return int(pattern[1][0]), int(pattern[1][1])

    raise ValueError(f"{cell_type} not found in connectome spec")


def extract_nc50_stride(
    cell_type: str,
    connectome_dir: Path | str | None = None,
) -> tuple[int, int]:
    """Extract the NC50 stride for a cell type from its perturbation JSON."""
    connectome_dir = Path(connectome_dir or default_connectome_dir())
    spec_path = connectome_dir / f"{cell_type}{NC50_SUFFIX}"
    if not spec_path.exists():
        raise FileNotFoundError(f"NC50 connectome not found: {spec_path}")
    return extract_node_stride(load_connectome_spec(spec_path), cell_type)


def ablation_ensemble_path(base: Path | str, cell_type: str) -> Path:
    """Return the conventional ablated ensemble path for ``base`` and type."""
    base = Path(base)
    return base.parent / f"{base.name}_{cell_type}"


def iter_network_dirs(ensemble_path: Path | str) -> list[Path]:
    """Return sorted network directories within an ensemble directory."""
    ensemble_path = Path(ensemble_path)
    return sorted(
        path
        for path in ensemble_path.iterdir()
        if path.is_dir() and (path / "_meta.yaml").exists()
    )


def model_id_from_path(path: Path | str) -> str:
    """Return the model ID from a network directory path."""
    return Path(path).name


def target_intensity(cell_type: str) -> int:
    """Return the intensity used for T4/T5 DSI comparisons."""
    if cell_type.startswith("T4"):
        return 1
    if cell_type.startswith("T5"):
        return 0
    raise ValueError(f"No DSI target intensity configured for {cell_type}")


def rank_network_records(records: Iterable[dict], top_n: int | None = None) -> list[dict]:
    """Rank records by ascending loss and mark the selected top rows."""
    ranked = []
    for rank, record in enumerate(
        sorted(records, key=lambda item: (float(item["loss"]), str(item["model_id"]))),
        start=1,
    ):
        row = dict(record)
        row["rank"] = rank
        row["selected"] = top_n is None or rank <= top_n
        ranked.append(row)
    return ranked
