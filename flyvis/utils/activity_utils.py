"""Methods to access and process network activity and currents based on cell types.

Example:
    Get the activity of all cells of a certain cell type
    (note, `LayerActivity` is not the network split into feedforward layers in the
    machine learning sense but the activity of all cells by cell type):
    ```python
    layer_activity = LayerActivity(activity, network.connectome)
    T4a_response = layer_activity.T4a
    T5a_response = layer_activity.T5a
    T4b_central_response = layer_activity.central.T4a
    ```
"""

import weakref
from textwrap import wrap
from typing import List, Union

import numpy as np
import torch
from numpy.typing import NDArray

from flyvis.connectome import ConnectomeFromAvgFilters, ReceptiveFields
from flyvis.utils import nodes_edges_utils

__all__ = [
    "CentralActivity",
    "LayerActivity",
    "SourceCurrentView",
]


class CellTypeActivity(dict):
    """Base class for attribute-style access to network activity based on cell types.

    Args:
        keepref: Whether to keep a reference to the activity. This may not be desired
            during training to avoid memory issues.

    Attributes:
        activity: Weak reference to the activity.
        keepref: Whether to keep a reference to the activity.
        unique_cell_types: List of unique cell types.
        input_indices: Indices of input cells.
        output_indices: Indices of output cells.

    Note:
        Activity is stored as a weakref by default for memory efficiency
        during training. Set keepref=True to keep a reference for analysis.
    """

    def __init__(self, keepref: bool = False):
        self.keepref = keepref
        self.activity: Union[weakref.ref, NDArray, torch.Tensor] = None
        self.unique_cell_types: List[str] = []
        self.input_indices: NDArray = np.array([])
        self.output_indices: NDArray = np.array([])

    def __dir__(self) -> List[str]:
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self) -> int:
        return len(self.unique_cell_types)

    def __iter__(self):
        yield from self.unique_cell_types

    def __repr__(self) -> str:
        return "Activity of: \n{}".format("\n".join(wrap(", ".join(list(self)))))

    def update(self, activity: Union[NDArray, torch.Tensor]) -> None:
        """Update the activity reference."""
        self.activity = activity

    def _slices(self, n: int) -> tuple:
        return tuple(slice(None) for _ in range(n))

    def __getattr__(self, key):
        activity = self.activity() if not self.keepref else self.activity
        if activity is None:
            return
        if isinstance(key, list):
            index = np.stack(list(map(lambda key: dict.__getitem__(self, key), key)))
            slices = self._slices(len(activity.shape) - 1)
            slices += (index,)
            return activity[slices]
        elif key == slice(None):
            return activity
        elif key in self.unique_cell_types:
            slices = self._slices(len(activity.shape) - 1)
            slices += (dict.__getitem__(self, key),)
            return activity[slices]
        elif key == "output":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.output_indices,)
            return activity[slices]
        elif key == "input":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.input_indices,)
            return activity[slices]
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            if self.keepref is False:
                value = weakref.ref(value)
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)


class CentralActivity(CellTypeActivity):
    """Attribute-style access to central cell activity of a cell type.

    Args:
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory with reference to required attributes.
        keepref: Whether to keep a reference to the activity.
        use_mask: If True, mask out non-existent neurons in ragged arrays.

    Attributes:
        activity: Activity of shape (..., n_cells).
        unique_cell_types: Array of unique cell types.
        index: NodeIndexer instance.
        input_indices: Array of input indices.
        output_indices: Array of output indices.
    """

    def __init__(
        self,
        activity: Union[NDArray, torch.Tensor],
        connectome: ConnectomeFromAvgFilters,
        keepref: bool = False,
        use_mask: bool = True,
    ):
        super().__init__(keepref)
        self.use_mask = use_mask  # If False, padding will be applied for ragged arrays
        self.index = nodes_edges_utils.NodeIndexer(connectome)

        unique_cell_types = connectome.unique_cell_types[:]
        input_cell_types = connectome.input_cell_types[:]
        output_cell_types = connectome.output_cell_types[:]
        
        # Create padded indices with validity masks (same logic as LayerActivity)
        self._create_central_padded_indices(unique_cell_types, input_cell_types, output_cell_types)
        
        self.activity = activity
        self.unique_cell_types = unique_cell_types.astype(str)

    def __getattr__(self, key):
        """Override to handle padding and masking for CentralActivity."""
        activity = self.activity() if not self.keepref else self.activity
        if activity is None:
            return
        
        # Handle padding for CentralActivity (uses central cells, so activity shape matches central neurons)
        if hasattr(self, 'output_mask') and hasattr(self, 'input_mask'):
            # For CentralActivity, we work with central cells, so we need to pad based on central activity size
            n_central = len(self.index.central_cells_index)
            if activity.shape[-1] == n_central:
                if isinstance(activity, torch.Tensor):
                    zeros = torch.zeros((*activity.shape[:-1], 1), dtype=activity.dtype, device=activity.device)
                    padded_activity = torch.cat([activity, zeros], dim=-1)
                else:
                    padded_activity = np.concatenate([activity, np.zeros((*activity.shape[:-1], 1))], axis=-1)
            else:
                padded_activity = activity
        else:
            padded_activity = activity
        
        if isinstance(key, list):
            index = np.stack(list(map(lambda key: self.index[key], key)))
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (index,)
            return padded_activity[slices]
        elif key == slice(None):
            return padded_activity
        elif key in self.index.unique_cell_types:
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (self.index[key],)
            return padded_activity[slices]
        elif key == "output":
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (self.output_indices,)
            result = padded_activity[slices]
            # Apply masking if use_mask=True
            if self.use_mask and hasattr(self, 'output_mask'):
                mask_shape = [1] * (len(result.shape) - 2) + list(self.output_mask.shape)
                expanded_mask = self.output_mask.reshape(mask_shape)
                if isinstance(result, torch.Tensor):
                    expanded_mask = torch.from_numpy(expanded_mask).to(result.device, dtype=result.dtype)
                result = result * expanded_mask
            return result
        elif key == "input":
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (self.input_indices,)
            result = padded_activity[slices]
            # Apply masking if use_mask=True
            if self.use_mask and hasattr(self, 'input_mask'):
                mask_shape = [1] * (len(result.shape) - 2) + list(self.input_mask.shape)
                expanded_mask = self.input_mask.reshape(mask_shape)
                if isinstance(result, torch.Tensor):
                    expanded_mask = torch.from_numpy(expanded_mask).to(result.device, dtype=result.dtype)
                result = result * expanded_mask
            return result
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            # Handle the case where value might be a weakref
            actual_value = value() if hasattr(value, '__call__') and hasattr(value, '__weakref__') else value
            if hasattr(actual_value, 'shape') and len(self.index.unique_cell_types) != actual_value.shape[-1]:
                slices = self._slices(len(actual_value.shape) - 1)
                slices += (self.index.central_cells_index,)
                actual_value = actual_value[slices]
                self.keepref = True
                value = actual_value
            if self.keepref is False:
                value = weakref.ref(value)
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __len__(self):
        return len(self.unique_cell_types)

    def __iter__(self):
        for cell_type in self.unique_cell_types:
            yield cell_type

    def _create_central_padded_indices(self, unique_cell_types, input_cell_types, output_cell_types):
        """Create padded indices arrays with validity masks for CentralActivity."""
        # Create variable-length index lists
        input_indices_list = [np.nonzero(unique_cell_types == t)[0] for t in input_cell_types]
        output_indices_list = [np.nonzero(unique_cell_types == t)[0] for t in output_cell_types]
        
        # Handle empty lists
        if input_indices_list:
            max_input_len = max(len(arr) for arr in input_indices_list)
            self.input_indices = np.full((len(input_indices_list), max_input_len), unique_cell_types.shape[0], dtype=int)
            self.input_mask = np.zeros((len(input_indices_list), max_input_len), dtype=bool)
            for i, arr in enumerate(input_indices_list):
                self.input_indices[i, :len(arr)] = arr
                self.input_mask[i, :len(arr)] = True
        else:
            self.input_indices = np.array([]).astype(int)
            self.input_mask = np.array([]).astype(bool)
            
        if output_indices_list:
            max_output_len = max(len(arr) for arr in output_indices_list)
            self.output_indices = np.full((len(output_indices_list), max_output_len), unique_cell_types.shape[0], dtype=int)
            self.output_mask = np.zeros((len(output_indices_list), max_output_len), dtype=bool)
            for i, arr in enumerate(output_indices_list):
                self.output_indices[i, :len(arr)] = arr
                self.output_mask[i, :len(arr)] = True
        else:
            self.output_indices = np.array([]).astype(int)
            self.output_mask = np.array([]).astype(bool)


class LayerActivity(CellTypeActivity):
    """Attribute-style access to hex-lattice activity (cell-type specific).

    Args:
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory with reference to required attributes.
        keepref: Whether to keep a reference to the activity.
        use_central: Whether to use central activity.
        use_mask: If True, mask out non-existent neurons in ragged arrays.
                 If False, padding will be applied for uniform tensor shapes.
                 Training/validation should use use_mask=False for decoder
                 compatibility. Analysis should use use_mask=True to avoid
                 contamination from padded neurons.

    Attributes:
        central: CentralActivity instance for central nodes.
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory.
        unique_cell_types: Array of unique cell types.
        input_indices: Array of input indices.
        output_indices: Array of output indices.
        input_cell_types: Array of input cell types.
        output_cell_types: Array of output cell types.
        n_nodes: Number of nodes.

    Note:
        The name `LayerActivity` might change in future as it is misleading.
        This is not a feedforward layer in the machine learning sense but the
        activity of all cells of a certain cell-type.

    Example:

        Central activity can be accessed by:
        ```python
        a = LayerActivity(activity, network.connectome)
        central_T4a = a.central.T4a
        ```

        Also allows 'virtual types' that are the sum of individuals:
        ```python
        a = LayerActivity(activity, network.connectome)
        summed_a = a['L2+L4']
        ```
    """

    def __init__(
        self,
        activity: Union[NDArray, torch.Tensor],
        connectome: ConnectomeFromAvgFilters,
        keepref: bool = False,
        use_central: bool = True,
        use_mask: bool = True,
    ):
        super().__init__(keepref)
        self.keepref = keepref
        self.use_mask = use_mask  # If False, padding will be applied for ragged arrays

        self.use_central = use_central
        if use_central:
            self.central = CentralActivity(activity, connectome, keepref, use_mask)

        self.activity = activity
        self.connectome = connectome
        self.unique_cell_types = connectome.unique_cell_types[:].astype("str")
        for cell_type in self.unique_cell_types:
            index = connectome.nodes.layer_index[cell_type][:]
            self[cell_type] = index

        # Create padded indices with validity masks
        self._create_padded_indices_with_masks()
        
        self.input_cell_types = self.connectome.input_cell_types[:].astype(str)
        self.output_cell_types = self.connectome.output_cell_types[:].astype(str)
        self.n_nodes = len(self.connectome.nodes.type)

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            original_value = value  # Keep reference to original before weakref
            
            if self.use_central:
                self.central.__setattr__(key, original_value)  # Pass original value to central
            
            if self.keepref is False:
                value = weakref.ref(value)

            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def _create_padded_indices_with_masks(self):
        """Create padded indices arrays with validity masks for handling ragged arrays."""
        _cell_types = self.connectome.nodes.type[:]
        
        # Create variable-length index lists
        input_indices_list = [np.nonzero(_cell_types == t)[0] for t in self.connectome.input_cell_types]
        output_indices_list = [np.nonzero(_cell_types == t)[0] for t in self.connectome.output_cell_types]
        
        # Handle empty lists
        if input_indices_list:
            max_input_len = max(len(arr) for arr in input_indices_list)
            self.input_indices = np.full((len(input_indices_list), max_input_len), _cell_types.shape[0], dtype=int)
            self.input_mask = np.zeros((len(input_indices_list), max_input_len), dtype=bool)
            for i, arr in enumerate(input_indices_list):
                self.input_indices[i, :len(arr)] = arr
                self.input_mask[i, :len(arr)] = True
        else:
            self.input_indices = np.array([]).astype(int)
            self.input_mask = np.array([]).astype(bool)
            
        if output_indices_list:
            max_output_len = max(len(arr) for arr in output_indices_list)
            self.output_indices = np.full((len(output_indices_list), max_output_len), _cell_types.shape[0], dtype=int)
            self.output_mask = np.zeros((len(output_indices_list), max_output_len), dtype=bool)
            for i, arr in enumerate(output_indices_list):
                self.output_indices[i, :len(arr)] = arr
                self.output_mask[i, :len(arr)] = True
        else:
            self.output_indices = np.array([]).astype(int)
            self.output_mask = np.array([]).astype(bool)

    def __getattr__(self, key):
        """Override parent method to handle padding and masking for ragged arrays."""
        activity = self.activity() if not self.keepref else self.activity
        if activity is None:
            return
        
        # Handle the special case where we need padding
        if hasattr(self, 'output_mask') and hasattr(self, 'input_mask'):
            # Extend activity with one extra zero-neuron for padding positions
            if activity.shape[-1] == self.n_nodes:
                if isinstance(activity, torch.Tensor):
                    zeros = torch.zeros((*activity.shape[:-1], 1), dtype=activity.dtype, device=activity.device)
                    padded_activity = torch.cat([activity, zeros], dim=-1)
                else:
                    padded_activity = np.concatenate([activity, np.zeros((*activity.shape[:-1], 1))], axis=-1)
            else:
                padded_activity = activity
        else:
            padded_activity = activity

        if isinstance(key, list):
            index = np.stack(list(map(lambda key: dict.__getitem__(self, key), key)))
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (index,)
            return padded_activity[slices]
        elif key == slice(None):
            return padded_activity
        elif key in self.unique_cell_types:
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (dict.__getitem__(self, key),)
            return padded_activity[slices]
        elif key == "output":
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (self.output_indices,)
            result = padded_activity[slices]
            # Apply masking if use_mask=True
            if self.use_mask and hasattr(self, 'output_mask'):
                # Expand mask to match result dimensions
                mask_shape = [1] * (len(result.shape) - 2) + list(self.output_mask.shape)
                expanded_mask = self.output_mask.reshape(mask_shape)
                if isinstance(result, torch.Tensor):
                    expanded_mask = torch.from_numpy(expanded_mask).to(result.device, dtype=result.dtype)
                result = result * expanded_mask
            return result
        elif key == "input":
            slices = self._slices(len(padded_activity.shape) - 1)
            slices += (self.input_indices,)
            result = padded_activity[slices]
            # Apply masking if use_mask=True
            if self.use_mask and hasattr(self, 'input_mask'):
                # Expand mask to match result dimensions
                mask_shape = [1] * (len(result.shape) - 2) + list(self.input_mask.shape)
                expanded_mask = self.input_mask.reshape(mask_shape)
                if isinstance(result, torch.Tensor):
                    expanded_mask = torch.from_numpy(expanded_mask).to(result.device, dtype=result.dtype)
                result = result * expanded_mask
            return result
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")


class SourceCurrentView:
    """Create views of source currents for a target type.

    Args:
        rfs: ReceptiveFields instance.
        currents: Current values.

    Attributes:
        target_type: Target cell type.
        source_types: List of source cell types.
        rfs: ReceptiveFields instance.
        currents: Current values.
    """

    def __init__(self, rfs: ReceptiveFields, currents: Union[NDArray, torch.Tensor]):
        self.target_type = rfs.target_type
        self.source_types = list(rfs)
        self.rfs = rfs
        self.currents = currents

    def __getattr__(self, key: str) -> Union[NDArray, torch.Tensor]:
        if key in self.source_types:
            return np.take(self.currents, self.rfs[key].index, axis=-1)
        return object.__getattr__(self, key)

    def __getitem__(self, key: str) -> Union[NDArray, torch.Tensor]:
        return self.__getattr__(key)

    def update(self, currents: Union[NDArray, torch.Tensor]) -> None:
        """Update the currents."""
        self.currents = currents
