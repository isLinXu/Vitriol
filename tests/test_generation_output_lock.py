"""A competing generator must not enter or mutate an active output directory."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import pytest
from test_core_generation_integrity import make_generator


@pytest.mark.parametrize("stage", ["_load_hf_config", "_initialize_model", "_save_shard", "_save_configs", "_write_manifest"])
def test_competing_generation_is_rejected_throughout_run(tmp_path, monkeypatch, stage):
    first, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    second, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    second_load = MagicMock(wraps=second._load_hf_config)
    monkeypatch.setattr(second, "_load_hf_config", second_load)
    original = getattr(first, stage)

    def compete(*args, **kwargs):
        with pytest.raises(RuntimeError, match="already.*generation"):
            second.generate()
        assert second_load.call_count == 0
        return original(*args, **kwargs)

    monkeypatch.setattr(first, stage, compete)
    first.generate()
    second.generate()  # Normal completion must release the lock.


def test_thread_contender_fails_without_waiting_for_owner(tmp_path, monkeypatch):
    first, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    second, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    entered, release = Event(), Event()
    original = first._load_hf_config

    def hold():
        entered.set()
        assert release.wait(15), "test owner was not released"
        return original()

    monkeypatch.setattr(first, "_load_hf_config", hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(first.generate)
        try:
            assert entered.wait(15)
            with pytest.raises(RuntimeError, match="already.*generation"):
                second.generate()
        finally:
            release.set()
        owner.result(timeout=15)


def test_directory_symlink_cannot_bypass_writer_exclusion(tmp_path, monkeypatch):
    destination = tmp_path / "output"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    first, _ = make_generator(destination, monkeypatch, indexed=False)
    second, _ = make_generator(alias, monkeypatch, indexed=False)

    def compete():
        with pytest.raises(RuntimeError, match="already.*generation"):
            second.generate()

    monkeypatch.setattr(first, "_write_manifest", compete)
    first.generate()


@pytest.mark.parametrize("error", [RuntimeError("failure"), KeyboardInterrupt(), SystemExit(73)])
def test_generation_exception_releases_directory(tmp_path, monkeypatch, error):
    first, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    monkeypatch.setattr(first, "_write_manifest", MagicMock(side_effect=error))
    with pytest.raises(type(error)):
        first.generate()
    second, _ = make_generator(tmp_path, monkeypatch, indexed=False)
    monkeypatch.setattr(second.strategy, "generate_tensor", MagicMock(side_effect=AssertionError("regenerated")))
    second.generate()


def test_recursive_use_of_same_generator_is_rejected(tmp_path, monkeypatch):
    generator, _ = make_generator(tmp_path, monkeypatch, indexed=False)

    def recurse():
        with pytest.raises(RuntimeError, match="already.*generation"):
            generator.generate()

    monkeypatch.setattr(generator, "_write_manifest", recurse)
    generator.generate()


def test_different_output_directories_do_not_share_lock(tmp_path, monkeypatch):
    first, _ = make_generator(tmp_path / "first", monkeypatch, indexed=False)
    second, _ = make_generator(tmp_path / "second", monkeypatch, indexed=False)
    monkeypatch.setattr(first, "_write_manifest", second.generate)
    first.generate()
    assert list((tmp_path / "first").glob("*.index.json"))
    assert list((tmp_path / "second").glob("*.index.json"))


def test_relative_directory_cannot_bypass_writer_exclusion(tmp_path, monkeypatch):
    from vitriol.core.output_lock import output_directory_lock

    monkeypatch.chdir(tmp_path)
    with output_directory_lock(str(tmp_path / "output")):
        with pytest.raises(RuntimeError, match="already.*generation"):
            with output_directory_lock("output/../output"):
                pytest.fail("entered competing lock")


def test_lock_marker_is_retained_without_truncation(tmp_path):
    from vitriol.core.output_lock import output_directory_lock

    path = tmp_path / ".vitriol-generation.lock"
    path.write_bytes(b"retained marker")
    inode = path.stat().st_ino
    for _ in range(3):
        with output_directory_lock(str(tmp_path)):
            assert path.read_bytes() == b"retained marker"
        assert path.stat().st_ino == inode


def test_symlink_lock_marker_is_not_followed(tmp_path):
    from vitriol.core.output_lock import output_directory_lock

    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    (tmp_path / ".vitriol-generation.lock").symlink_to(outside)
    with pytest.raises(OSError):
        with output_directory_lock(str(tmp_path)):
            pytest.fail("followed lock symlink")
    assert outside.read_bytes() == b"preserve"


@pytest.mark.parametrize("error_number", [5, 38])
def test_lock_io_failures_propagate_and_close_descriptor(tmp_path, monkeypatch, error_number):
    import os

    import vitriol.core.output_lock as module

    descriptors = []

    def fail(fd):
        descriptors.append(fd)
        raise OSError(error_number, "lock unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_acquire", fail)
        with pytest.raises(OSError, match="lock unavailable"):
            with module.output_directory_lock(str(tmp_path)):
                pytest.fail("entered after lock failure")
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    with module.output_directory_lock(str(tmp_path)):
        pass


def test_owned_orphan_cleanup_preserves_committed_and_unrelated_paths(tmp_path):
    import tempfile

    from vitriol.core.output_lock import output_directory_lock

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important").write_bytes(b"preserve")
    orphan_dirs = [Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_path)) for prefix in (".vitriol-shard-", ".vitriol-index-")]
    for directory in orphan_dirs:
        (directory / "partial").write_bytes(b"partial")
        (directory / "outside-link").symlink_to(outside, target_is_directory=True)
    with tempfile.NamedTemporaryFile(prefix=".vitriol-checkpoint-", dir=tmp_path, delete=False) as file:
        orphan_file = Path(file.name)
        file.write(b"partial")
    preserved = [".vitriol_checkpoint.json", "model.safetensors", "model.safetensors.index.json", ".vitriol-checkpoint-user.json"]
    for name in preserved:
        (tmp_path / name).write_bytes(b"preserve")
    custom = tmp_path / ".vitriol-index-custom"
    custom.mkdir()
    symlink = tmp_path / ".vitriol-index-abcd1234"
    symlink.symlink_to(outside, target_is_directory=True)
    with output_directory_lock(str(tmp_path)):
        assert not orphan_file.exists()
        assert not any(directory.exists() for directory in orphan_dirs)
        assert all((tmp_path / name).read_bytes() == b"preserve" for name in preserved)
        assert custom.exists()
        assert symlink.is_symlink()
        assert (outside / "important").read_bytes() == b"preserve"


def test_losing_contender_does_not_clean_live_temporary_files(tmp_path):
    import tempfile

    from vitriol.core.output_lock import output_directory_lock

    with output_directory_lock(str(tmp_path)):
        live = Path(tempfile.mkdtemp(prefix=".vitriol-shard-", dir=tmp_path))
        (live / "weight").write_bytes(b"active")
        with pytest.raises(RuntimeError, match="already.*generation"):
            with output_directory_lock(str(tmp_path)):
                pytest.fail("entered contender")
        assert (live / "weight").read_bytes() == b"active"
    with output_directory_lock(str(tmp_path)):
        assert not live.exists()


def test_cleanup_failure_is_logged_without_leaking_ownership(tmp_path, monkeypatch, caplog):
    import tempfile

    import vitriol.core.output_lock as module

    orphan = Path(tempfile.mkdtemp(prefix=".vitriol-shard-", dir=tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, "rmtree", MagicMock(side_effect=PermissionError("denied")))
        with module.output_directory_lock(str(tmp_path)):
            assert orphan.exists()
    assert "Could not remove orphan" in caplog.text
    with module.output_directory_lock(str(tmp_path)):
        assert not orphan.exists()


@pytest.mark.parametrize("body_fails", [False, True])
def test_unlock_error_closes_descriptor_and_preserves_original_error(tmp_path, monkeypatch, body_fails, caplog):
    import vitriol.core.output_lock as module

    with monkeypatch.context() as patch:
        patch.setattr(module, "_release", MagicMock(side_effect=OSError("unlock failure")))
        with pytest.raises(RuntimeError if body_fails else OSError, match="generation failure" if body_fails else "unlock failure"):
            with module.output_directory_lock(str(tmp_path)):
                if body_fails:
                    raise RuntimeError("generation failure")
    with module.output_directory_lock(str(tmp_path)):
        pass
    if body_fails:
        assert "while handling an error" in caplog.text
