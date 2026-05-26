#!/usr/bin/env python
"""Assess WT and NC50 ablation DSI after independent task-error ranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from nc50_utils import (
    TARGET_DSI_CELL_TYPES,
    ablation_ensemble_path,
    iter_network_dirs,
    model_id_from_path,
    parse_type_selection,
    rank_network_records,
    target_intensity,
)


def read_h5_array(path: Path):
    """Read a datamate-style HDF5 array."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Reading validation losses requires h5py and numpy in the active "
            "Python environment."
        ) from exc

    datasets = []
    with h5py.File(path, "r") as handle:
        if "data" in handle:
            return np.asarray(handle["data"][()])

        def collect_dataset(_, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(np.asarray(obj[()]))

        handle.visititems(collect_dataset)

    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset in {path}, found {len(datasets)}")
    return datasets[0]


def min_validation_loss(
    network_dir: Path,
    validation_subdir: str,
    loss_file_name: str,
) -> float:
    """Read the minimum validation loss for a network."""
    import numpy as np

    loss_path = network_dir / validation_subdir / f"{loss_file_name}.h5"
    if not loss_path.exists():
        raise FileNotFoundError(f"Validation loss file not found: {loss_path}")

    values = np.asarray(read_h5_array(loss_path), dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Validation loss file is empty: {loss_path}")
    return float(np.nanmin(values))


def load_validation_records(
    ensemble_path: Path,
    validation_subdir: str,
    loss_file_name: str,
) -> list[dict]:
    """Load per-network validation loss records for an ensemble."""
    records = []
    for network_dir in iter_network_dirs(ensemble_path):
        network_dir = network_dir.resolve()
        records.append(
            {
                "model_id": model_id_from_path(network_dir),
                "network_path": str(network_dir),
                "loss": min_validation_loss(
                    network_dir,
                    validation_subdir=validation_subdir,
                    loss_file_name=loss_file_name,
                ),
            }
        )
    if not records:
        raise FileNotFoundError(f"No network directories found in {ensemble_path}")
    return records


def ranking_frame_for_group(
    group: str,
    ablation_type: str,
    ensemble_path: Path,
    validation_subdir: str,
    loss_file_name: str,
    top_n: int,
):
    """Build a ranked DataFrame for one ensemble."""
    import pandas as pd

    records = load_validation_records(
        ensemble_path,
        validation_subdir=validation_subdir,
        loss_file_name=loss_file_name,
    )
    ranked = rank_network_records(records, top_n=top_n)
    frame = pd.DataFrame(ranked)
    frame.insert(0, "group", group)
    frame.insert(1, "ablation_type", ablation_type)
    frame.insert(2, "ensemble_path", str(ensemble_path.resolve()))
    frame["validation_subdir"] = validation_subdir
    frame["loss_file_name"] = loss_file_name
    return frame


def build_rankings(
    base: Path,
    ablation_types: Iterable[str],
    top_n: int,
    wt_validation_subdir: str,
    ablation_validation_subdir: str,
    loss_file_name: str,
):
    """Build WT and ablated ranking tables."""
    import pandas as pd

    frames = [
        ranking_frame_for_group(
            "WT",
            "WT",
            base,
            validation_subdir=wt_validation_subdir,
            loss_file_name=loss_file_name,
            top_n=top_n,
        )
    ]
    for ablation_type in ablation_types:
        frames.append(
            ranking_frame_for_group(
                ablation_type,
                ablation_type,
                ablation_ensemble_path(base, ablation_type),
                validation_subdir=ablation_validation_subdir,
                loss_file_name=loss_file_name,
                top_n=top_n,
            )
        )
    return pd.concat(frames, ignore_index=True)


def moving_edge_dataset(speeds: Iterable[float]):
    """Create the default MovingEdge dataset used for DSI assessment."""
    import numpy as np
    from flyvis.datasets.moving_bar import MovingEdge

    return MovingEdge(
        offsets=(-10, 11),
        intensities=[0, 1],
        speeds=list(speeds),
        height=80,
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        dt=1 / 200,
        angles=list(np.arange(0, 360, 30)),
    )


def _ensure_network_dim(dsis):
    """Ensure DSI arrays keep a network dimension for top_n=1."""
    if "network_id" in dsis.dims:
        return dsis
    if "network_id" in dsis.coords:
        dsis = dsis.drop_vars("network_id")
    return dsis.expand_dims(network_id=[0])


def extract_dsi_records(dsis, selected_rankings) -> list[dict]:
    """Extract one DSI row per selected model and target T4/T5 cell type."""
    dsis = _ensure_network_dim(dsis)
    selected_rankings = selected_rankings.sort_values("rank").reset_index(drop=True)

    records = []
    for network_pos, rank_row in selected_rankings.iterrows():
        network_dsi = dsis.isel(network_id=network_pos)
        for cell_type in TARGET_DSI_CELL_TYPES:
            intensity = target_intensity(cell_type)
            cell_dsi = network_dsi.where(
                network_dsi["cell_type"] == cell_type,
                drop=True,
            )
            if cell_dsi.sizes.get("neuron", 0) == 0:
                continue

            value = cell_dsi.sel(intensity=intensity).mean(dim="neuron")
            records.append(
                {
                    "group": rank_row["group"],
                    "ablation_type": rank_row["ablation_type"],
                    "ensemble_path": rank_row["ensemble_path"],
                    "model_id": rank_row["model_id"],
                    "network_path": rank_row["network_path"],
                    "rank": int(rank_row["rank"]),
                    "loss": float(rank_row["loss"]),
                    "cell_type": cell_type,
                    "intensity": intensity,
                    "dsi": float(value),
                }
            )
    return records


def compute_group_dsi_records(
    selected_rankings,
    validation_subdir: str,
    loss_file_name: str,
    dataset,
    batch_size: int,
) -> list[dict]:
    """Compute DSI records for one selected, ranked ensemble group."""
    from flyvis.analysis.moving_bar_responses import direction_selectivity_index
    from flyvis.network import Ensemble

    selected_rankings = selected_rankings.sort_values("rank").reset_index(drop=True)
    paths = [Path(path) for path in selected_rankings["network_path"]]
    ensemble = Ensemble(
        paths,
        best_checkpoint_fn_kwargs={
            "validation_subdir": validation_subdir,
            "loss_file_name": loss_file_name,
        },
    )
    responses = ensemble.moving_edge_responses(dataset=dataset, batch_size=batch_size)
    dsis = direction_selectivity_index(responses)
    return extract_dsi_records(dsis, selected_rankings)


def compute_all_dsi_records(
    rankings,
    groups: Iterable[str],
    wt_validation_subdir: str,
    ablation_validation_subdir: str,
    loss_file_name: str,
    speeds: Iterable[float],
    batch_size: int,
):
    """Compute DSI for every selected WT and ablated group."""
    import pandas as pd

    dataset = moving_edge_dataset(speeds)
    records = []
    for group in groups:
        selected = rankings[(rankings["group"] == group) & rankings["selected"]]
        validation_subdir = (
            wt_validation_subdir if group == "WT" else ablation_validation_subdir
        )
        print(f"{group}: computing DSI for {len(selected)} selected model(s)")
        records.extend(
            compute_group_dsi_records(
                selected,
                validation_subdir=validation_subdir,
                loss_file_name=loss_file_name,
                dataset=dataset,
                batch_size=batch_size,
            )
        )
    return pd.DataFrame(records)


def filter_target_dsi_table(dsi_table):
    """Keep only T4 ON and T5 OFF rows used by the assessment."""
    def is_target_row(row) -> bool:
        if row["cell_type"] not in TARGET_DSI_CELL_TYPES:
            return False
        return int(row["intensity"]) == target_intensity(row["cell_type"])

    mask = dsi_table.apply(is_target_row, axis=1)
    return dsi_table[mask].copy()


def bh_fdr(p_values: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction preserving NaN positions."""
    import math
    import numpy as np

    p_values = list(p_values)
    finite = [
        (idx, float(value))
        for idx, value in enumerate(p_values)
        if not math.isnan(value)
    ]
    q_values = [math.nan] * len(p_values)
    if not finite:
        return q_values

    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = np.empty(len(ordered), dtype=float)
    running = 1.0
    total = len(ordered)
    for pos in range(len(ordered) - 1, -1, -1):
        rank = pos + 1
        running = min(running, ordered[pos][1] * total / rank)
        adjusted[pos] = min(running, 1.0)

    for (original_idx, _), value in zip(ordered, adjusted):
        q_values[original_idx] = float(value)
    return q_values


def _safe_mannwhitneyu(wt_values, ablation_values) -> tuple[float, float]:
    from scipy.stats import mannwhitneyu

    if len(wt_values) == 0 or len(ablation_values) == 0:
        return float("nan"), float("nan")
    try:
        result = mannwhitneyu(wt_values, ablation_values, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def primary_stats(dsi_table, ablation_types: Iterable[str]):
    """Compute independent WT vs ablation Mann-Whitney U statistics."""
    import numpy as np
    import pandas as pd

    wt_table = dsi_table[dsi_table["group"] == "WT"]
    rows = []
    for ablation_type in ablation_types:
        ablation_table = dsi_table[dsi_table["group"] == ablation_type]
        for cell_type in TARGET_DSI_CELL_TYPES:
            wt_values = wt_table.loc[wt_table["cell_type"] == cell_type, "dsi"].dropna()
            ablation_values = ablation_table.loc[
                ablation_table["cell_type"] == cell_type,
                "dsi",
            ].dropna()
            statistic, p_value = _safe_mannwhitneyu(wt_values, ablation_values)
            rows.append(
                {
                    "ablation_type": ablation_type,
                    "cell_type": cell_type,
                    "intensity": target_intensity(cell_type),
                    "n_wt": int(len(wt_values)),
                    "n_ablation": int(len(ablation_values)),
                    "wt_median": float(np.nanmedian(wt_values))
                    if len(wt_values)
                    else np.nan,
                    "ablation_median": float(np.nanmedian(ablation_values))
                    if len(ablation_values)
                    else np.nan,
                    "median_delta": float(
                        np.nanmedian(ablation_values) - np.nanmedian(wt_values)
                    )
                    if len(wt_values) and len(ablation_values)
                    else np.nan,
                    "wt_mean": float(np.nanmean(wt_values)) if len(wt_values) else np.nan,
                    "ablation_mean": float(np.nanmean(ablation_values))
                    if len(ablation_values)
                    else np.nan,
                    "mean_delta": float(
                        np.nanmean(ablation_values) - np.nanmean(wt_values)
                    )
                    if len(wt_values) and len(ablation_values)
                    else np.nan,
                    "mannwhitney_u": statistic,
                    "p_value": p_value,
                }
            )

    stats = pd.DataFrame(rows)
    stats["q_value"] = bh_fdr(stats["p_value"])
    return stats


def _safe_wilcoxon(deltas) -> tuple[float, float]:
    from scipy.stats import wilcoxon

    if len(deltas) < 3:
        return float("nan"), float("nan")
    try:
        result = wilcoxon(deltas, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def same_id_stats(dsi_table, ablation_types: Iterable[str]):
    """Compute supplemental paired same-model deltas where top-N IDs overlap."""
    import numpy as np
    import pandas as pd

    wt_table = dsi_table[dsi_table["group"] == "WT"]
    rows = []
    for ablation_type in ablation_types:
        ablation_table = dsi_table[dsi_table["group"] == ablation_type]
        for cell_type in TARGET_DSI_CELL_TYPES:
            wt_values = wt_table.loc[
                wt_table["cell_type"] == cell_type,
                ["model_id", "dsi"],
            ].rename(columns={"dsi": "wt_dsi"})
            ablation_values = ablation_table.loc[
                ablation_table["cell_type"] == cell_type,
                ["model_id", "dsi"],
            ].rename(columns={"dsi": "ablation_dsi"})
            paired = wt_values.merge(ablation_values, on="model_id", how="inner")
            paired["delta"] = paired["ablation_dsi"] - paired["wt_dsi"]
            deltas = paired["delta"].dropna()
            statistic, p_value = _safe_wilcoxon(deltas)
            rows.append(
                {
                    "ablation_type": ablation_type,
                    "cell_type": cell_type,
                    "intensity": target_intensity(cell_type),
                    "n_overlap": int(len(deltas)),
                    "overlap_model_ids": ";".join(paired["model_id"].astype(str)),
                    "mean_delta": float(np.nanmean(deltas)) if len(deltas) else np.nan,
                    "median_delta": float(np.nanmedian(deltas))
                    if len(deltas)
                    else np.nan,
                    "wilcoxon_stat": statistic,
                    "p_value": p_value,
                }
            )

    stats = pd.DataFrame(rows)
    stats["q_value"] = bh_fdr(stats["p_value"])
    return stats


def write_violin_plots(dsi_table, groups: list[str], out_dir: Path) -> None:
    """Write per-cell-type violin plots."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for cell_type in TARGET_DSI_CELL_TYPES:
        plot_groups = []
        plot_data = []
        for group in groups:
            values = dsi_table.loc[
                (dsi_table["group"] == group) & (dsi_table["cell_type"] == cell_type),
                "dsi",
            ].dropna()
            if len(values):
                plot_groups.append(group)
                plot_data.append(values.to_numpy())

        if not plot_data:
            continue

        fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(plot_groups)), 3.0))
        ax.violinplot(plot_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(plot_groups) + 1))
        ax.set_xticklabels(plot_groups, rotation=60, ha="right")
        ax.set_ylabel("DSI")
        ax.set_title(f"{cell_type} DSI")
        fig.tight_layout()
        fig.savefig(out_dir / f"dsi_violin_{cell_type}.png", dpi=200)
        plt.close(fig)


def write_delta_heatmap(stats_table, ablation_types: list[str], out_dir: Path) -> None:
    """Write a median delta heatmap."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = stats_table.pivot(
        index="ablation_type",
        columns="cell_type",
        values="median_delta",
    ).reindex(index=ablation_types, columns=TARGET_DSI_CELL_TYPES)

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.3 * len(ablation_types))))
    image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(TARGET_DSI_CELL_TYPES)))
    ax.set_xticklabels(TARGET_DSI_CELL_TYPES, rotation=45, ha="right")
    ax.set_yticks(range(len(ablation_types)))
    ax.set_yticklabels(ablation_types)
    ax.set_title("Median DSI Delta (Ablation - WT)")
    fig.colorbar(image, ax=ax, label="Median delta")
    fig.tight_layout()
    fig.savefig(out_dir / "median_delta_heatmap.png", dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Assess NC50 ablation DSI after independent task-error ranking."
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
        help="Ablation types to assess, or 'all' for *_50_nc.json files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of independently ranked models to assess per group.",
    )
    parser.add_argument(
        "--wt-validation-subdir",
        default="validation",
        help="Validation subdirectory used to rank WT models.",
    )
    parser.add_argument(
        "--ablation-validation-subdir",
        default="validation",
        help="Validation subdirectory used to rank ablated models.",
    )
    parser.add_argument(
        "--loss-file-name",
        default="epe",
        help="Validation loss file name without .h5.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory. Defaults to <base-parent>/<base-name>_nc50_dsi.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for moving edge response simulation.",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[2.4, 4.8, 9.7, 13, 19, 25],
        help="MovingEdge speeds for DSI assessment.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSVs only.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()
    ablation_types = parse_type_selection(args.types)
    groups = ["WT", *ablation_types]
    out_dir = args.out or args.base.parent / f"{args.base.name}_nc50_dsi"

    rankings = build_rankings(
        args.base,
        ablation_types=ablation_types,
        top_n=args.top_n,
        wt_validation_subdir=args.wt_validation_subdir,
        ablation_validation_subdir=args.ablation_validation_subdir,
        loss_file_name=args.loss_file_name,
    )
    dsi_table = compute_all_dsi_records(
        rankings,
        groups=groups,
        wt_validation_subdir=args.wt_validation_subdir,
        ablation_validation_subdir=args.ablation_validation_subdir,
        loss_file_name=args.loss_file_name,
        speeds=args.speeds,
        batch_size=args.batch_size,
    )
    dsi_table = filter_target_dsi_table(dsi_table)
    stats_table = primary_stats(dsi_table, ablation_types)
    same_id_table = same_id_stats(dsi_table, ablation_types)

    out_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(out_dir / "rankings.csv", index=False)
    dsi_table.to_csv(out_dir / "dsi_long.csv", index=False)
    stats_table.to_csv(out_dir / "stats_summary.csv", index=False)
    same_id_table.to_csv(out_dir / "same_id_supplemental.csv", index=False)
    if not args.skip_plots:
        write_violin_plots(dsi_table, groups, out_dir)
        write_delta_heatmap(stats_table, ablation_types, out_dir)

    print(f"Wrote NC50 DSI assessment outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
