#!/usr/bin/env python
"""Recompute task error for NC50 ablated ensembles."""

from __future__ import annotations

import argparse
from pathlib import Path

from nc50_utils import (
    ablation_ensemble_path,
    iter_network_dirs,
    parse_type_selection,
)


def first_checkpoint(path: Path | str, **_) -> Path:
    """Select the first checkpoint without consulting validation files."""
    checkpoint_dir = Path(path) / "chkpts"
    checkpoints = sorted(
        candidate
        for candidate in checkpoint_dir.glob("chkpt_*")
        if candidate.is_file() and not candidate.name.endswith(".success")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[0]


def validation_loss_exists(
    network_dir: Path,
    validation_subdir: str,
    loss_file_name: str = "epe",
) -> bool:
    """Return whether the expected validation loss file already exists."""
    return (network_dir / validation_subdir / f"{loss_file_name}.h5").exists()


def selected_network_dirs(
    ensemble_path: Path,
    network_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Return network directories filtered by optional IDs and limit."""
    network_dirs = iter_network_dirs(ensemble_path)
    if network_ids:
        wanted = set(network_ids)
        network_dirs = [path for path in network_dirs if path.name in wanted]
    if limit is not None:
        network_dirs = network_dirs[:limit]
    return network_dirs


def validate_network(
    network_dir: Path,
    validation_subdir: str,
    dt: float,
    t_pre: float,
) -> None:
    """Run validation for a single ablated network."""
    from flyvis import NetworkView
    from flyvis.analysis.validation import validate_all_checkpoints

    network_view = NetworkView(network_dir, best_checkpoint_fn=first_checkpoint)
    validate_all_checkpoints(
        network_view,
        dt=dt,
        t_pre=t_pre,
        validation_subdir=validation_subdir,
    )


def validate_ablation_type(
    base: Path,
    cell_type: str,
    validation_subdir: str,
    dt: float,
    t_pre: float,
    skip_existing: bool,
    network_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Validate all requested networks for one ablation type."""
    ensemble_path = ablation_ensemble_path(base, cell_type)
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Ablated ensemble for {cell_type} does not exist: {ensemble_path}"
        )

    network_dirs = selected_network_dirs(ensemble_path, network_ids, limit)
    if not network_dirs:
        raise FileNotFoundError(f"No network directories found in {ensemble_path}")

    completed = 0
    skipped = 0
    for network_dir in network_dirs:
        if skip_existing and validation_loss_exists(network_dir, validation_subdir):
            print(f"{cell_type}/{network_dir.name}: validation exists, skipping")
            skipped += 1
            continue

        print(f"{cell_type}/{network_dir.name}: validating into {validation_subdir}")
        validate_network(network_dir, validation_subdir, dt=dt, t_pre=t_pre)
        completed += 1

    return completed, skipped


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Recompute validation task error for NC50 ablated ensembles."
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Base WT ensemble path, e.g. data/results/flow/0000.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["all"],
        help="Ablation types to validate, or 'all' for *_50_nc.json files.",
    )
    parser.add_argument(
        "--validation-subdir",
        default="validation",
        help="Validation subdirectory to write inside each ablated network.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1 / 50,
        help="Validation timestep passed to validate_all_checkpoints.",
    )
    parser.add_argument(
        "--t-pre",
        type=float,
        default=0.5,
        help="Pre-stimulus duration passed to validate_all_checkpoints.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip networks that already have <validation-subdir>/epe.h5.",
    )
    parser.add_argument(
        "--network-ids",
        nargs="+",
        help="Optional model IDs to validate, e.g. 000 001.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit after sorting network directories.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    cell_types = parse_type_selection(args.types)

    total_completed = 0
    total_skipped = 0
    for cell_type in cell_types:
        completed, skipped = validate_ablation_type(
            args.base,
            cell_type,
            validation_subdir=args.validation_subdir,
            dt=args.dt,
            t_pre=args.t_pre,
            skip_existing=args.skip_existing,
            network_ids=args.network_ids,
            limit=args.limit,
        )
        total_completed += completed
        total_skipped += skipped

    print(
        "Done. "
        f"Validated {total_completed} network(s); skipped {total_skipped} existing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
