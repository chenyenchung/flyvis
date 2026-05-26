import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from assess_nc50_dsi import filter_target_dsi_table, primary_stats  # noqa: E402
from nc50_utils import (  # noqa: E402
    discover_nc50_types,
    extract_nc50_stride,
    extract_node_stride,
    parse_type_selection,
    rank_network_records,
    target_intensity,
)


class _FakeEdges:
    target_type = ["Mi1", "Mi1", "T4a"]
    target_u = [0, 1, 1]
    target_v = [0, 0, 0]


class _FakeConnectome:
    edges = _FakeEdges()


class _FakeNetwork:
    connectome = _FakeConnectome()


class _FakeNetworkView:
    def __init__(self, path):
        self.path = path

    def init_network(self, checkpoint="best"):
        return _FakeNetwork()

    def get_checkpoint(self, checkpoint="best"):
        return self.path / "source_chkpt"


def write_connectome(path, target, stride):
    spec = {
        "nodes": [
            {"name": "R1", "pattern": ["stride", [1, 1]]},
            {"name": target, "pattern": ["stride", list(stride)]},
        ]
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_ablate_and_save_does_not_copy_or_keep_stale_validation(tmp_path, monkeypatch):
    import h5py
    import ablate_network

    input_path = tmp_path / "flow" / "0000" / "000"
    output_path = tmp_path / "flow" / "0000_Mi1" / "000"
    (input_path / "validation").mkdir(parents=True)
    (output_path / "validation").mkdir(parents=True)
    (output_path / "validation_ablation").mkdir(parents=True)
    (output_path / "__cache__").mkdir(parents=True)
    (input_path / "_meta.yaml").write_text("type: NetworkDir\n", encoding="utf-8")
    (input_path / "validation" / "epe.h5").write_text("wt", encoding="utf-8")
    (output_path / "validation" / "epe.h5").write_text("stale", encoding="utf-8")
    (output_path / "validation_ablation" / "epe.h5").write_text(
        "stale",
        encoding="utf-8",
    )
    (output_path / "__cache__" / ".gitignore").write_text("*\n", encoding="utf-8")

    checkpoint = {
        "network": {"param": object()},
        "decoder": {"flow": object()},
        "val_loss": 1.25,
        "iteration": 123,
        "dt": 0.02,
    }
    monkeypatch.setattr(ablate_network, "NetworkView", _FakeNetworkView)
    monkeypatch.setattr(ablate_network.torch, "load", lambda *_, **__: checkpoint)
    monkeypatch.setattr(
        ablate_network,
        "atomic_torch_save",
        lambda _, path: path.write_bytes(b"checkpoint"),
    )

    ablate_network.ablate_and_save(input_path, output_path, "Mi1")

    assert not (output_path / "validation").exists()
    assert not (output_path / "validation_ablation").exists()
    assert not (output_path / "__cache__").exists()
    assert (output_path / "_meta.yaml").exists()
    assert (output_path / "chkpts" / "chkpt_00000").exists()

    with h5py.File(output_path / "chkpt_index.h5") as handle:
        assert handle["data"][()].tolist() == [0]
    with h5py.File(output_path / "best_chkpt_index.h5") as handle:
        assert handle["data"][()] == 0
    with h5py.File(output_path / "chkpt_iter.h5") as handle:
        assert handle["data"][()] == 123
    with h5py.File(output_path / "dt.h5") as handle:
        assert handle["data"][()] == pytest.approx(0.02)


def test_nc50_type_discovery_and_stride_extraction(tmp_path):
    spec = write_connectome(tmp_path / "Mi1_50_nc.json", "Mi1", (2, 1))
    write_connectome(tmp_path / "TmY5a_50_nc.json", "TmY5a", (2, 1))
    write_connectome(tmp_path / "Other.json", "Other", (1, 1))

    assert discover_nc50_types(tmp_path) == ["Mi1", "TmY5a"]
    assert parse_type_selection(["all"], tmp_path) == ["Mi1", "TmY5a"]
    assert extract_node_stride(spec, "Mi1") == (2, 1)
    assert extract_nc50_stride("TmY5a", tmp_path) == (2, 1)


def test_rank_network_records_uses_independent_losses():
    wt = rank_network_records(
        [
            {"model_id": "000", "loss": 0.3},
            {"model_id": "001", "loss": 0.1},
            {"model_id": "002", "loss": 0.2},
        ],
        top_n=2,
    )
    ablated = rank_network_records(
        [
            {"model_id": "000", "loss": 0.01},
            {"model_id": "001", "loss": 0.4},
            {"model_id": "002", "loss": 0.2},
        ],
        top_n=2,
    )

    assert [row["model_id"] for row in wt if row["selected"]] == ["001", "002"]
    assert [row["model_id"] for row in ablated if row["selected"]] == ["000", "002"]
    assert [row["rank"] for row in wt] == [1, 2, 3]


def test_t4_t5_dsi_filtering_uses_expected_intensities():
    table = pd.DataFrame(
        [
            {"cell_type": "T4a", "intensity": 0, "dsi": 0.1},
            {"cell_type": "T4a", "intensity": 1, "dsi": 0.2},
            {"cell_type": "T5d", "intensity": 0, "dsi": 0.3},
            {"cell_type": "T5d", "intensity": 1, "dsi": 0.4},
            {"cell_type": "Mi1", "intensity": 1, "dsi": 0.5},
        ]
    )

    filtered = filter_target_dsi_table(table)

    assert target_intensity("T4a") == 1
    assert target_intensity("T5d") == 0
    assert filtered[["cell_type", "intensity", "dsi"]].to_dict("records") == [
        {"cell_type": "T4a", "intensity": 1, "dsi": 0.2},
        {"cell_type": "T5d", "intensity": 0, "dsi": 0.3},
    ]


def test_primary_stats_compares_wt_and_ablation_independently():
    rows = []
    for model_id, dsi in [("000", 0.1), ("001", 0.2), ("002", 0.3)]:
        rows.append(
            {
                "group": "WT",
                "model_id": model_id,
                "cell_type": "T4a",
                "intensity": 1,
                "dsi": dsi,
            }
        )
    for model_id, dsi in [("010", 0.4), ("011", 0.5), ("012", 0.6)]:
        rows.append(
            {
                "group": "Mi1",
                "model_id": model_id,
                "cell_type": "T4a",
                "intensity": 1,
                "dsi": dsi,
            }
        )

    stats = primary_stats(pd.DataFrame(rows), ["Mi1"])
    t4a = stats[stats["cell_type"] == "T4a"].iloc[0]

    assert t4a["n_wt"] == 3
    assert t4a["n_ablation"] == 3
    assert t4a["median_delta"] == pytest.approx(0.3)
    assert t4a["mean_delta"] == pytest.approx(0.3)
