"""Content digests should skip sparse holes without depending on allocation."""

import os

import pytest

from vitriol.core.checkpoint_state import shard_digest


def test_sparse_digest_read_volume_is_bounded_by_data_extents(tmp_path, monkeypatch):
    import vitriol.core.checkpoint_state as module

    path = tmp_path / "large-sparse.bin"
    with path.open("wb") as stream:
        stream.write(b"header")
        stream.seek(64 * 1024 * 1024 - 1)
        stream.write(b"x")
    if not hasattr(os, "SEEK_HOLE"):
        pytest.skip("filesystem extent queries unavailable")
    with path.open("rb") as stream:
        if os.lseek(stream.fileno(), 0, os.SEEK_HOLE) >= path.stat().st_size:
            pytest.skip("filesystem does not expose sparse extents")
    actual_open = open
    read_bytes = 0

    class Reader:
        def __init__(self, *args, **kwargs):
            self.stream = actual_open(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            nonlocal read_bytes
            result = self.stream.read(size)
            read_bytes += len(result)
            return result

    monkeypatch.setattr(module, "open", Reader, raising=False)
    assert shard_digest(str(path))
    assert read_bytes <= 2 * 1024 * 1024


@pytest.mark.parametrize("size,positions", [
    (0, []), (4096, []), (3 * 1024 * 1024, []),
    (4 * 1024 * 1024 + 13, [0, 1024 * 1024 - 1, 1024 * 1024 + 9, 4 * 1024 * 1024 + 12]),
])
def test_sparse_and_dense_contents_have_identical_digests(tmp_path, size, positions):
    sparse = tmp_path / "sparse"
    dense = tmp_path / "dense"
    with sparse.open("wb") as stream:
        stream.truncate(size)
        for position in positions:
            stream.seek(position)
            stream.write(b"x")
    dense.write_bytes(sparse.read_bytes())
    assert shard_digest(str(sparse)) == shard_digest(str(dense))


@pytest.mark.parametrize("unsupported_at", ["data", "hole", "missing_api"])
def test_extent_fallback_preserves_content_digest(tmp_path, monkeypatch, unsupported_at):
    import errno

    path = tmp_path / "sparse"
    with path.open("wb") as stream:
        stream.write(b"first")
        stream.seek(4 * 1024 * 1024 + 17)
        stream.write(b"last")
    expected = shard_digest(str(path))
    if unsupported_at == "missing_api":
        monkeypatch.delattr(os, "SEEK_DATA", raising=False)
    else:
        lseek = os.lseek

        def unsupported(fd, offset, whence):
            if whence == (os.SEEK_DATA if unsupported_at == "data" else os.SEEK_HOLE):
                raise OSError(errno.EINVAL, "extent query unsupported")
            return lseek(fd, offset, whence)

        monkeypatch.setattr(os, "lseek", unsupported)
    assert shard_digest(str(path)) == expected


@pytest.mark.parametrize("mutation", ["append_zero", "truncate", "write_hole", "move_data"])
def test_digest_detects_length_position_and_content_changes(tmp_path, mutation):
    path = tmp_path / "sparse"
    with path.open("wb") as stream:
        stream.truncate(5 * 1024 * 1024)
        stream.write(b"x")
    before = shard_digest(str(path))
    with path.open("r+b") as stream:
        if mutation == "append_zero":
            stream.seek(0, os.SEEK_END)
            stream.write(b"\0")
        elif mutation == "truncate":
            stream.truncate(4 * 1024 * 1024)
        elif mutation == "write_hole":
            stream.seek(2 * 1024 * 1024)
            stream.write(b"x")
        else:
            stream.write(b"\0")
            stream.seek(1024 * 1024)
            stream.write(b"x")
    assert shard_digest(str(path)) != before


def test_legacy_digest_matches_standard_sha256(tmp_path):
    import hashlib

    path = tmp_path / "content"
    path.write_bytes(b"legacy payload" + b"\0" * 20)
    assert shard_digest(str(path), legacy=True) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_extent_io_errors_are_not_treated_as_empty_files(tmp_path, monkeypatch):
    import errno

    path = tmp_path / "content"
    path.write_bytes(b"data")

    def fail(*args):
        raise OSError(errno.EIO, "read failure")

    monkeypatch.setattr(os, "lseek", fail)
    with pytest.raises(OSError, match="read failure"):
        shard_digest(str(path))


@pytest.mark.parametrize("behavior", ["short_read", "truncate", "rewrite"])
def test_digest_handles_short_reads_and_rejects_concurrent_changes(tmp_path, monkeypatch, behavior):
    import vitriol.core.checkpoint_state as module

    path = tmp_path / "content"
    path.write_bytes(b"x" * (1024 * 1024 + 17))
    expected = shard_digest(str(path))
    actual_open = open
    changed = False

    class Reader:
        def __init__(self, *args, **kwargs):
            self.stream = actual_open(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            nonlocal changed
            block = self.stream.read(min(size, 32768))
            if not changed and behavior != "short_read":
                changed = True
                with actual_open(path, "r+b") as writer:
                    if behavior == "truncate":
                        writer.truncate(0)
                    else:
                        writer.write(b"y")
            return block

    monkeypatch.setattr(module, "open", Reader, raising=False)
    if behavior == "short_read":
        assert shard_digest(str(path)) == expected
    else:
        with pytest.raises(OSError, match="truncated|changed"):
            shard_digest(str(path))


def test_seek_hole_failure_is_not_mistaken_for_an_empty_tail(tmp_path, monkeypatch):
    import errno

    path = tmp_path / "content"
    path.write_bytes(b"data")
    lseek = os.lseek

    def fail(fd, offset, whence):
        if whence == os.SEEK_HOLE:
            raise OSError(errno.ENXIO, "hole query failure")
        return lseek(fd, offset, whence)

    monkeypatch.setattr(os, "lseek", fail)
    with pytest.raises(OSError, match="hole query failure"):
        shard_digest(str(path))


@pytest.mark.parametrize("descriptor", [True, False])
def test_byte_accounting_prefers_storage_then_descriptor_protocol(descriptor):
    from types import SimpleNamespace

    import torch

    from vitriol.core.generator import MinimalWeightGenerator

    value = SimpleNamespace(nbytes=8) if descriptor else torch.ones(1).expand(4096)
    assert MinimalWeightGenerator._estimate_tensor_nbytes(
        value, fallback_numel=4096, fallback_dtype=torch.float32,
    ) == (8 if descriptor else 4)
