import logging
import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import torch
from torch import nn

import flyvis
from flyvis.utils.logging_utils import warn_once

logger = logging.getLogger(__name__)


def recover_network(
    network: nn.Module,
    state_dict: Union[Dict, Path, str],
    ensemble_and_network_id: str = None,
) -> nn.Module:
    """
    Load network parameters from state dict.

    Args:
        network: FlyVision network.
        state_dict: State or path to checkpoint containing the "network" parameters.
        ensemble_and_network_id: Optional identifier for the network.

    Returns:
        The updated network.
    """
    state = get_from_state_dict(state_dict, "network")
    if state is not None:
        network.load_state_dict(state)
        logging.info(
            "Recovered network state%s",
            f" {ensemble_and_network_id}." if ensemble_and_network_id else ".",
        )
    else:
        logging.warning("Could not recover network state.")
    return network


def recover_decoder(
    decoder: Dict[str, nn.Module], state_dict: Union[Dict, Path], strict: bool = True
) -> Dict[str, nn.Module]:
    """
    Recover multiple decoders from state dict.

    Args:
        decoder: Dictionary of decoders.
        state_dict: State or path to checkpoint.
        strict: Whether to strictly enforce that the keys in state_dict match.

    Returns:
        The updated dictionary of decoders.
    """
    states = get_from_state_dict(state_dict, "decoder")
    if states is not None:
        for key, dec in decoder.items():
            state = states.pop(key, None)
            if state is not None:
                dec.load_state_dict(state, strict=strict)
                logging.info("Recovered %s decoder state.", key)
            else:
                logging.warning("Could not recover state of %s decoder.", key)
    else:
        logging.warning("Could not recover decoder states.")
    return decoder


def recover_optimizer(
    optimizer: torch.optim.Optimizer, state_dict: Union[Dict, Path]
) -> torch.optim.Optimizer:
    """
    Recover optimizer state from state dict.

    Args:
        optimizer: PyTorch optimizer.
        state_dict: State or path to checkpoint.

    Returns:
        The updated optimizer.
    """
    state = get_from_state_dict(state_dict, "optim")
    if state is not None:
        optimizer.load_state_dict(state)
        logging.info("Recovered optimizer state.")
    else:
        logging.warning("Could not recover optimizer state.")
    return optimizer


def recover_penalty_optimizers(
    optimizers: Dict[str, torch.optim.Optimizer], state_dict: Union[Dict, Path]
) -> Dict[str, torch.optim.Optimizer]:
    """
    Recover penalty optimizers from state dict.

    Args:
        optimizers: Dictionary of penalty optimizers.
        state_dict: State or path to checkpoint.

    Returns:
        The updated dictionary of penalty optimizers.
    """
    states = get_from_state_dict(state_dict, "penalty_optims")
    if states is not None:
        for key, optim in optimizers.items():
            state = states.pop(key, None)
            if state is not None:
                optim.load_state_dict(state)
                logging.info("Recovered %s optimizer state.", key)
            else:
                logging.warning("Could not recover state of %s optimizer.", key)
    else:
        logging.warning("Could not recover penalty optimizer states.")
    return optimizers


def get_from_state_dict(state_dict: Union[Dict, Path, str], key: str) -> Dict:
    """
    Get a specific key from the state dict.

    Args:
        state_dict: State dict or path to checkpoint.
        key: Key to retrieve from the state dict.

    Returns:
        The value associated with the key in the state dict.

    Raises:
        TypeError: If state_dict is not of type Path, str, or dict.
    """
    if state_dict is None:
        return None
    if isinstance(state_dict, (Path, str)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            state = torch.load(
                state_dict, map_location=flyvis.device, weights_only=False
            ).pop(key, None)
    elif isinstance(state_dict, dict):
        state = state_dict.get(key, None)
    else:
        raise TypeError(
            f"state_dict must be of type Path, str or dict, but is {type(state_dict)}."
        )
    return state


@dataclass
class Checkpoints:
    """
    Dataclass to store checkpoint information.

    Attributes:
        indices: List of checkpoint indices.
        paths: List of checkpoint paths.
    """

    indices: List[int]
    paths: List[Path]

    def __repr__(self):
        return (
            f"Checkpoints("
            f"  indices={repr(self.indices)},"
            f"  paths={repr(self.paths)},"
            f")"
        )


def resolve_checkpoints(
    networkdir: "flyvis.network.NetworkDir",
) -> Checkpoints:
    """
    Resolve checkpoints from network directory.

    Args:
        networkdir: FlyVision network directory.

    Returns:
        A Checkpoints object containing indices and paths of checkpoints.
    """
    indices, paths = checkpoint_index_to_path_map(networkdir.chkpts.path)
    return Checkpoints(indices, paths)


def checkpoint_index_to_path_map(
    path: Path,
    glob: str = "chkpt_*",
) -> Tuple[List[int], List[Path]]:
    """
    Returns all numerical identifiers and paths to checkpoints stored in path.

    Args:
        path: Checkpoint directory.
        glob: Glob pattern for checkpoint files.

    Returns:
        A tuple containing a list of indices and a list of paths to checkpoints.
    """
    import re

    path.mkdir(exist_ok=True)
    paths = np.array(sorted(list((path).glob(glob))))
    try:
        _index = [int(re.findall(r"\d{1,10}", p.parts[-1])[0]) for p in paths]
        _sorting_index = np.argsort(_index)
        paths = paths[_sorting_index].tolist()
        index = np.array(_index)[_sorting_index].tolist()
        return index, paths
    except IndexError:
        return [], paths


def best_checkpoint_default_fn(
    path: Path,
    validation_subdir: str = "validation",
    loss_file_name: str = "loss",
) -> Path:
    """
    Find the best checkpoint based on the minimum loss.

    Args:
        path: Path to the network directory.
        validation_subdir: Subdirectory containing validation data.
        loss_file_name: Name of the loss file.

    Returns:
        Path to the best checkpoint.
    """
    networkdir = flyvis.NetworkDir(path)
    checkpoint_dir = networkdir.chkpts.path
    indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
    loss_file_name = check_loss_name(networkdir[validation_subdir], loss_file_name)
    index = np.argmin(networkdir[validation_subdir][loss_file_name][()])
    index = indices[index]
    path = paths[index]
    return path


def check_loss_name(loss_folder, loss_file_name: str) -> str:
    """
    Check if the loss file name exists in the loss folder.

    Args:
        loss_folder: The folder containing loss files.
        loss_file_name: The name of the loss file to check.

    Returns:
        The validated loss file name.
    """
    if loss_file_name not in loss_folder and "loss" in loss_folder:
        warn_once(
            logging,
            f"{loss_file_name} not in {loss_folder.path}, but 'loss' is. "
            "Falling back to 'loss'. You can rerun the ensemble validation to make "
            "appropriate recordings of the losses.",
        )
        loss_file_name = "loss"
    return loss_file_name


def atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    """
    Saves a torch object to a file atomically using a marker file.
    Writes to a temp file first, renames it, then creates a .success marker.
    This is robust for FUSE/Network filesystems where rename might not be atomic.

    Args:
        obj: Object to save.
        path: Destination path.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    success_marker = path.with_suffix(path.suffix + ".success")

    try:
        # Write data to temp file
        torch.save(obj, tmp_path)
        
        # Ensure data is written to disk/network
        if hasattr(os, "fsync"):
            try:
                with open(tmp_path, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass  # fsync might not be supported on all FS

        # Rename to final filename (Best effort atomic on FUSE)
        os.replace(tmp_path, path)
        
        # Create success marker
        success_marker.touch()
        
    except Exception as e:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise e


def is_valid_checkpoint(path: Path) -> bool:
    """
    Checks if a checkpoint file is valid.
    Requires a corresponding .success marker file to exist.
    If marker exists, also attempts to load to verify integrity.

    Args:
        path: Path to the checkpoint file.

    Returns:
        True if the checkpoint is valid, False otherwise.
    """
    path = Path(path)
    success_marker = path.with_suffix(path.suffix + ".success")
    
    # 1. Check for success marker
    if not success_marker.exists():
        return False
        
    # 2. Verify integrity by loading
    try:
        # map_location='cpu' avoids loading to GPU, faster for validation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def find_last_good_checkpoint(
    checkpoint_dir: Path, glob: str = "chkpt_*"
) -> Optional[Path]:
    """
    Finds the latest valid checkpoint.
    Iterates backwards from newest file, validating each.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        glob: Glob pattern for checkpoint files.

    Returns:
        Path to the last valid checkpoint, or None if none found.
    """
    _, paths = checkpoint_index_to_path_map(checkpoint_dir, glob)

    # Iterate backwards (newest first)
    for path in reversed(paths):
        if is_valid_checkpoint(path):
            return path
        else:
            logger.warning(f"Found incomplete/corrupted checkpoint at {path}, skipping.")

    return None


def purge_temporary_checkpoints(checkpoint_dir: Path) -> None:
    """
    Removes temporary files and orphan checkpoints (missing .success marker).

    Args:
        checkpoint_dir: Directory containing checkpoints.
    """
    if not checkpoint_dir.exists():
        return

    # 1. Remove .tmp files
    for tmp_file in checkpoint_dir.glob("*.tmp"):
        try:
            logger.info(f"Purging temporary checkpoint file: {tmp_file}")
            os.remove(tmp_file)
        except OSError as e:
            logger.warning(f"Failed to delete {tmp_file}: {e}")
            
    # 2. Remove orphan checkpoints (no .success marker)
    # Note: We need to be careful not to delete a file that is currently being written.
    # The atomic_torch_save flow (tmp -> rename -> marker) means there is a small window
    # where the file exists without a marker.
    # However, this function is typically called at startup (recovery), where no active
    # writing should be happening.
    
    # We scan for chkpt_* files (excluding markers themselves)
    # Using the same glob pattern logic as checkpoint_index_to_path_map
    _, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
    
    for path in paths:
        # If it's a marker file itself, skip (though glob should filter it if strictly chkpt_*)
        if path.suffix == ".success":
            continue
            
        success_marker = path.with_suffix(path.suffix + ".success")
        if not success_marker.exists():
            try:
                logger.info(f"Purging orphan checkpoint (no success marker): {path}")
                os.remove(path)
            except OSError as e:
                logger.warning(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    nv = flyvis.NetworkView("flow/9998/000")
    print(resolve_checkpoints(nv.dir))