"""Read a PyTorch .pth state dict without PyTorch.

Exists for one reason: torch cannot always be installed where the ONNX artifacts need to be
produced. A .pth is a zip of pickled metadata plus raw little-endian storage blobs, so numpy
can do the job — the only trick is that the pickle references those blobs through
`persistent_id` and rebuilds tensors through `torch._utils._rebuild_tensor_v2`, both of which
have to be stubbed rather than imported.

Only what a state dict actually contains is supported: ordered dicts of dense CPU tensors.
Anything else raises rather than guessing, because a silently partial load is exactly the
failure this is meant to avoid.
"""

import collections
import io
import pickle
import zipfile

import numpy as np

# torch storage class name -> numpy dtype. The names are what appear in the pickle stream.
_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


class _Storage:
    """A pointer into the archive, resolved lazily when a tensor is rebuilt from it."""

    def __init__(self, archive, key, dtype):
        self.archive = archive
        self.key = key
        self.dtype = dtype

    def read(self):
        for name in (f"{self.archive}/data/{self.key}", f"data/{self.key}"):
            try:
                return np.frombuffer(self._zip.read(name), dtype=self.dtype)
            except KeyError:
                continue
        raise KeyError(f"storage {self.key} not found in the archive")


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_rest):
    """torch._utils._rebuild_tensor_v2, restricted to dense contiguous CPU tensors."""
    flat = storage.read()
    count = int(np.prod(size)) if len(size) else 1
    values = flat[storage_offset : storage_offset + count]

    if len(size):
        expected = tuple(_contiguous_stride(size))
        if tuple(stride) != expected:
            raise ValueError(f"non-contiguous tensor (stride {tuple(stride)}, expected {expected})")
    return values.reshape(tuple(size)).copy()


def _contiguous_stride(size):
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * size[i + 1]
    return stride


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, zf, archive):
        super().__init__(file)
        self._zf = zf
        self._archive = archive

    def find_class(self, module, name):
        if module.startswith("torch") and name.endswith("Storage"):
            if name not in _DTYPES:
                raise ValueError(f"unsupported storage type {name}")
            return name
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "collections" and name == "OrderedDict":
            # The real class, not dict: OrderedDict is restored through the BUILD opcode and
            # a plain dict has no __dict__ for the unpickler to populate.
            return collections.OrderedDict
        # Anything else in a state dict is a surprise worth stopping for.
        raise ValueError(f"refusing to unpickle {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key = pid[0], pid[1], pid[2]
        if kind != "storage":
            raise ValueError(f"unsupported persistent id kind {kind!r}")
        storage = _Storage(self._archive, key, _DTYPES[storage_type])
        storage._zip = self._zf
        return storage


def load_state_dict(path):
    """.pth -> {name: numpy array}. Unwraps the usual model/state_dict nesting."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        archive = pkl.rsplit("/", 1)[0] if "/" in pkl else ""
        state = _Unpickler(io.BytesIO(zf.read(pkl)), zf, archive).load()

    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError(f"{path} did not contain a state dict")
    return state
