"""JSON-safe random state and bounded-memory checkpoint artifact verification."""

import base64
import errno
import hashlib
import os
import platform
import random
from math import prod
from typing import Any, Dict

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    """Capture default generators without initializing CUDA on CPU-only runs."""
    numpy_state = np.random.get_state()
    return {
        "version": 1,
        "runtime": [platform.python_version(), np.__version__, torch.__version__],
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode("ascii"),
        "cuda": [base64.b64encode(s.cpu().numpy().tobytes()).decode("ascii")
                 for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Validate with isolated generators before changing process-wide state."""
    if state["runtime"] != [platform.python_version(), np.__version__, torch.__version__]:
        raise ValueError("RNG checkpoint runtime version changed")
    if state["version"] != 1:
        raise ValueError("unsupported RNG checkpoint version")
    python_state = (state["python"][0], tuple(state["python"][1]), state["python"][2])
    numpy_state = (state["numpy"][0], np.asarray(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:])
    random.Random().setstate(python_state)
    np.random.RandomState().set_state(numpy_state)

    def decode(encoded):
        return torch.tensor(list(base64.b64decode(encoded, validate=True)), dtype=torch.uint8, device="cpu")

    cpu_state = decode(state["torch"])
    torch.Generator(device="cpu").set_state(cpu_state)
    cuda_states = state["cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG checkpoint does not match available devices")
        cuda_states = [decode(s) for s in cuda_states]
        for i, cuda_state in enumerate(cuda_states):
            torch.Generator(device=f"cuda:{i}").set_state(cuda_state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


_DIGEST_BLOCK_SIZE = 1024 * 1024
_DIGEST_PREFIX = "chunk-sha256-v1:"


def shard_digest(path: str, *, legacy: bool = False) -> str:
    """Hash content independently of sparse/dense filesystem allocation.

    The versioned format hashes file length and each nonzero 1 MiB block's
    index, length and SHA-256. All-zero blocks are implicit, so SEEK_DATA/SEEK_HOLE
    can skip holes. Unsupported filesystems fall back to sequential reads with
    the same digest. Legacy mode verifies previous whole-file SHA-256 records.
    """
    with open(path, "rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        size = before.st_size
        digest = hashlib.sha256()
        if legacy:
            for block in iter(lambda: stream.read(_DIGEST_BLOCK_SIZE), b""):
                digest.update(block)
        else:
            digest.update(b"vitriol-chunk-sha256-v1\0" + size.to_bytes(8, "big"))
            use_extents = hasattr(os, "SEEK_DATA") and hasattr(os, "SEEK_HOLE")
            offset = 0
            while offset < size:
                start, end = offset, size
                if use_extents:
                    try:
                        try:
                            data = os.lseek(stream.fileno(), offset, os.SEEK_DATA)
                        except OSError as exc:
                            if exc.errno == errno.ENXIO:
                                break  # No data remains beyond this offset.
                            raise
                        if data >= size:
                            break
                        hole = os.lseek(stream.fileno(), data, os.SEEK_HOLE)
                        if hole <= data:
                            raise OSError("invalid sparse extent")
                        start = data // _DIGEST_BLOCK_SIZE * _DIGEST_BLOCK_SIZE
                        end = min(size, ((hole + _DIGEST_BLOCK_SIZE - 1) // _DIGEST_BLOCK_SIZE) * _DIGEST_BLOCK_SIZE)
                    except OSError as exc:
                        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                            raise
                        use_extents = False
                stream.seek(start)
                for position in range(start, end, _DIGEST_BLOCK_SIZE):
                    length = min(_DIGEST_BLOCK_SIZE, size - position)
                    block = stream.read(length)
                    while len(block) < length:
                        remainder = stream.read(length - len(block))
                        if not remainder:
                            raise OSError("shard truncated during hashing")
                        block += remainder
                    if block.strip(b"\0"):
                        digest.update((position // _DIGEST_BLOCK_SIZE).to_bytes(8, "big"))
                        digest.update(length.to_bytes(4, "big"))
                        digest.update(hashlib.sha256(block).digest())
                offset = end
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise OSError("shard changed during hashing")
    return ("" if legacy else _DIGEST_PREFIX) + digest.hexdigest()


def inspect_shard(path: str):
    """Return tensor shapes, dtypes and accounted storage bytes.

    Safetensors metadata is inspected without materializing dense weights.
    PyTorch shards use the restricted loader and preserve compact storage views.
    """
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        dtype_names = {
            "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
            "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
            "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
            "F8_E4M3": "float8_e4m3fn", "F8_E4M3FNUZ": "float8_e4m3fnuz",
            "F8_E5M2": "float8_e5m2", "F8_E5M2FNUZ": "float8_e5m2fnuz", "C64": "complex64",
        }
        shapes, dtypes, total = {}, {}, 0
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                view = shard.get_slice(name)
                shapes[name] = tuple(view.get_shape())
                dtype = getattr(torch, dtype_names[view.get_dtype()])
                dtypes[name] = str(dtype)
                if not name.startswith("__vitriol_pad__"):
                    total += prod(shapes[name]) * torch.empty((), dtype=dtype, device="cpu").element_size()
        return shapes, dtypes, total
    shard = torch.load(path, map_location="cpu", weights_only=True)
    return (
        {n: tuple(t.shape) for n, t in shard.items()},
        {n: str(t.dtype) for n, t in shard.items()},
        sum(t.untyped_storage().nbytes() for n, t in shard.items() if not n.startswith("__vitriol_pad__")),
    )
