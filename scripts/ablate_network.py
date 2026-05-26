#!/usr/bin/env python
"""Ablate edges to specific cell types and save modified networks.

This script creates ablated versions of trained networks by:
1. Loading trained checkpoints
2. Creating position-specific edge masks
3. Saving new checkpoints with the masks embedded

The ablation zeros edges to a target cell type at positions that would not exist
with a sparser stride pattern (e.g., Mi1 at stride [2,1] instead of [1,1]).

Usage:
    # Single network
    python scripts/ablate_network.py data/results/flow/0000/000 --ablate Mi1

    # Batch (entire ensemble)
    python scripts/ablate_network.py data/results/flow/0000 --ablate Mi1

    # Custom stride (keep positions where u % 4 == 0)
    python scripts/ablate_network.py data/results/flow/0000 --ablate Mi1 --stride-u 4
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

from flyvis import NetworkView
from flyvis.utils.chkpt_utils import atomic_torch_save


STALE_OUTPUTS = (
    "__cache__",
    "training",
    "training_batch",
    "validation",
    "validation_ablation",
    "validation_batch",
    "validation_loss.h5",
)


def remove_path(path):
    """Remove an existing file, symlink, or directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def clean_stale_outputs(output_path):
    """Remove stale outputs that must be recomputed for an ablated checkpoint."""
    for name in STALE_OUTPUTS:
        remove_path(output_path / name)


def create_ablation_mask(network, target_type, stride_u=2, stride_v=1):
    """Create mask for position-specific edge ablation.

    Ablates edges to nodes of `target_type` at positions that would not exist
    with the specified stride pattern.

    Args:
        network: Initialized flyvis Network
        target_type: Cell type to ablate edges to (e.g., "Mi1")
        stride_u: Keep nodes where u % stride_u == 0 (default: 2)
        stride_v: Keep nodes where v % stride_v == 0 (default: 1)

    Returns:
        edge_mask: Tensor of shape (n_edges,) with 0 for ablated edges, 1 otherwise
        ablate_mask: Boolean array indicating which edges are ablated
    """
    edges = network.connectome.edges
    target_types = np.array([
        t.decode() if isinstance(t, bytes) else t
        for t in edges.target_type[:]
    ])
    target_u = np.asarray(edges.target_u[:])
    target_v = np.asarray(edges.target_v[:])

    # Find edges to the target type
    type_mask = target_types == target_type

    # Find positions that would NOT exist with the sparser stride
    off_grid_u = target_u % stride_u != 0
    off_grid_v = target_v % stride_v != 0
    off_grid = off_grid_u | off_grid_v

    # Edges to ablate: target is specified type AND at off-grid position
    ablate_mask = type_mask & off_grid

    # Create mask (1 = keep, 0 = ablate)
    edge_mask = torch.ones(len(target_types), dtype=torch.float32)
    edge_mask[ablate_mask] = 0.0

    return edge_mask, ablate_mask


def ablate_and_save(input_path, output_path, target_type, stride_u=2, stride_v=1):
    """Load network, create ablation mask, save to new directory.

    Args:
        input_path: Path to trained network directory
        output_path: Path to save ablated network
        target_type: Cell type to ablate
        stride_u: Keep positions where u % stride_u == 0
        stride_v: Keep positions where v % stride_v == 0

    Returns:
        output_path: Path to saved ablated network
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load network
    nv = NetworkView(input_path)
    network = nv.init_network(checkpoint="best")

    # Create ablation mask
    edge_mask, ablate_mask = create_ablation_mask(
        network, target_type, stride_u, stride_v
    )
    n_ablated = int(ablate_mask.sum())
    print(f"Ablating {n_ablated} edges to {target_type}")

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    clean_stale_outputs(output_path)
    chkpts_dir = output_path / "chkpts"
    chkpts_dir.mkdir(exist_ok=True)

    # Copy _meta.yaml
    shutil.copy(input_path / "_meta.yaml", output_path / "_meta.yaml")

    # Load original checkpoint using NetworkView's get_checkpoint
    orig_chkpt_path = nv.get_checkpoint("best")
    orig_chkpt = torch.load(orig_chkpt_path, map_location="cpu", weights_only=False)

    # Create new checkpoint with edge mask
    new_chkpt = {
        "network": orig_chkpt["network"],
        "decoder": orig_chkpt.get("decoder", {}),
        "optim": {},
        "val_loss": orig_chkpt.get("val_loss", float("nan")),
        "iteration": orig_chkpt.get("iteration", -1),
        "dt": orig_chkpt.get("dt", 0.02),
        "edge_mask": edge_mask,
        "ablation_info": {
            "target_type": target_type,
            "stride_u": stride_u,
            "stride_v": stride_v,
            "n_ablated": n_ablated,
        },
    }

    # Save checkpoint
    atomic_torch_save(new_chkpt, chkpts_dir / "chkpt_00000")

    # Create index files
    with h5py.File(output_path / "chkpt_index.h5", "w") as f:
        f.create_dataset("data", data=[0])
    with h5py.File(output_path / "best_chkpt_index.h5", "w") as f:
        f.create_dataset("data", data=0)
    with h5py.File(output_path / "chkpt_iter.h5", "w") as f:
        f.create_dataset("data", data=orig_chkpt.get("iteration", -1))
    with h5py.File(output_path / "dt.h5", "w") as f:
        f.create_dataset("data", data=orig_chkpt.get("dt", 0.02))

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Ablate network edges to specific cell types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Input path (single network dir or ensemble dir)"
    )
    parser.add_argument(
        "--ablate",
        required=True,
        help="Cell type to ablate edges to (e.g., Mi1, T4a, Tm1)"
    )
    parser.add_argument(
        "--stride-u",
        type=int,
        default=2,
        help="Keep positions where u %% stride_u == 0 (default: 2)"
    )
    parser.add_argument(
        "--stride-v",
        type=int,
        default=1,
        help="Keep positions where v %% stride_v == 0 (default: 1)"
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Output suffix (default: _<ablate>)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    suffix = args.output_suffix or f"_{args.ablate}"

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1

    # Check if input is single network or ensemble
    if (input_path / "_meta.yaml").exists():
        # Single network: flow/0000/000 -> flow/0000_Mi1/000
        # Parent (ensemble) dir gets suffix, network name stays same
        ensemble_dir = input_path.parent
        output_ensemble = ensemble_dir.parent / f"{ensemble_dir.name}{suffix}"
        output_path = output_ensemble / input_path.name
        print(f"Processing {input_path} -> {output_path}")
        ablate_and_save(
            input_path, output_path, args.ablate, args.stride_u, args.stride_v
        )
    else:
        # Ensemble directory: flow/0000 -> flow/0000_Mi1 (preserving subdirs)
        output_ensemble = input_path.parent / f"{input_path.name}{suffix}"
        print(f"Processing ensemble {input_path} -> {output_ensemble}")

        # Find all network subdirectories
        subdirs = sorted([
            d for d in input_path.iterdir()
            if d.is_dir() and (d / "_meta.yaml").exists()
        ])

        if not subdirs:
            print(f"Error: No network directories found in {input_path}")
            return 1

        print(f"Found {len(subdirs)} networks to process")

        for subdir in subdirs:
            # flow/0000/000 -> flow/0000_Mi1/000
            output_subdir = output_ensemble / subdir.name
            print(f"  {subdir.name}: ", end="", flush=True)
            ablate_and_save(
                subdir, output_subdir, args.ablate, args.stride_u, args.stride_v
            )

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
