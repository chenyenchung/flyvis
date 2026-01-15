import pytest
import torch
import shutil
import os
from pathlib import Path
from flyvis.utils.chkpt_utils import (
    atomic_torch_save,
    find_last_good_checkpoint,
    purge_temporary_checkpoints,
    is_valid_checkpoint,
    resolve_checkpoints
)
from flyvis.network.directories import NetworkDir

@pytest.fixture
def temp_chkpt_dir(tmp_path):
    d = tmp_path / "chkpts"
    d.mkdir()
    return d

def test_atomic_torch_save_with_marker(temp_chkpt_dir):
    data = {"a": 1, "b": 2}
    save_path = temp_chkpt_dir / "chkpt_00000"
    
    atomic_torch_save(data, save_path)
    
    # 1. Main file exists
    assert save_path.exists()
    # 2. Marker exists
    marker = save_path.with_suffix(save_path.suffix + ".success")
    assert marker.exists()
    # 3. Temp file gone
    assert not (save_path.with_suffix(".tmp")).exists()
    
    loaded_data = torch.load(save_path)
    assert loaded_data == data

def test_is_valid_checkpoint_requires_marker(temp_chkpt_dir):
    # Case 1: File + Marker (Valid)
    valid_path = temp_chkpt_dir / "valid.pt"
    torch.save({"state": "good"}, valid_path)
    # The implementation uses path.with_suffix(path.suffix + ".success") which results in .pt.success
    marker = valid_path.with_suffix(valid_path.suffix + ".success")
    marker.touch()
    assert is_valid_checkpoint(valid_path)
    
    # Case 2: File only (Invalid - e.g. crashed during marker write)
    orphan_path = temp_chkpt_dir / "orphan.pt"
    torch.save({"state": "orphan"}, orphan_path)
    assert not is_valid_checkpoint(orphan_path)
    
    # Case 3: File + Marker but File Corrupted (Invalid)
    corrupt_path = temp_chkpt_dir / "corrupt.pt"
    with open(corrupt_path, "wb") as f:
        f.write(b"garbage")
    marker = corrupt_path.with_suffix(corrupt_path.suffix + ".success")
    marker.touch()
    assert not is_valid_checkpoint(corrupt_path)

def test_find_last_good_checkpoint_with_markers(temp_chkpt_dir):
    # 0: Good (File + Marker)
    chkpt0 = temp_chkpt_dir / "chkpt_00000"
    torch.save({"iter": 0}, chkpt0)
    chkpt0.with_suffix(chkpt0.suffix + ".success").touch()
    
    # 1: Orphan (File only)
    chkpt1 = temp_chkpt_dir / "chkpt_00001"
    torch.save({"iter": 1}, chkpt1)
    
    # Should find 0, skipping 1
    last_good = find_last_good_checkpoint(temp_chkpt_dir)
    assert last_good == chkpt0
    
    # 2: Good (Newer)
    chkpt2 = temp_chkpt_dir / "chkpt_00002"
    torch.save({"iter": 2}, chkpt2)
    chkpt2.with_suffix(chkpt2.suffix + ".success").touch()
    
    last_good = find_last_good_checkpoint(temp_chkpt_dir)
    assert last_good == chkpt2

def test_purge_orphans_and_temps(temp_chkpt_dir):
    # 1. Good checkpoint (Keep)
    good = temp_chkpt_dir / "chkpt_00000"
    torch.save({}, good)
    good.with_suffix(good.suffix + ".success").touch()
    
    # 2. Orphan checkpoint (Purge)
    orphan = temp_chkpt_dir / "chkpt_00001"
    torch.save({}, orphan)
    
    # 3. Temp file (Purge)
    tmp = temp_chkpt_dir / "chkpt_00002.pt.tmp"
    tmp.touch()
    
    purge_temporary_checkpoints(temp_chkpt_dir)
    
    assert good.exists()
    assert good.with_suffix(good.suffix + ".success").exists()
    assert not orphan.exists()
    assert not tmp.exists()