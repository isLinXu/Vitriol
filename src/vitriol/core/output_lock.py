"""Exclusive generation ownership and orphan cleanup for an output directory."""

import errno
import logging
import os
import re
import shutil
import stat
import sys
import threading
from contextlib import contextmanager

from ..utils.exceptions import OutputDirectoryBusyError

logger = logging.getLogger(__name__)
_LOCK_NAME = ".vitriol-generation.lock"
_TEMP_FILE = re.compile(r"\.vitriol-checkpoint-[a-z0-9_]{8}")
_TEMP_DIR = re.compile(r"\.vitriol-(?:shard|index)-[a-z0-9_]{8}")


class _DescriptorRegistry:
    """Keep fork from inheriting lock descriptors, including an open/register race."""

    def __init__(self):
        self._guard = threading.Lock()
        self._descriptors = set()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(
                before=self._guard.acquire,
                after_in_parent=self._guard.release,
                after_in_child=self._after_fork_child,
            )

    def open(self, path, flags, mode):
        # Fork's before-hook waits until every new descriptor is registered.
        with self._guard:
            fd = os.open(path, flags, mode)
            try:
                self._descriptors.add(fd)
            except BaseException:
                os.close(fd)
                raise
            return fd

    def close(self, fd):
        # Close and unregister together so a reused FD cannot be closed in a child.
        with self._guard:
            try:
                os.close(fd)
            finally:
                self._descriptors.discard(fd)

    def _after_fork_child(self):
        try:
            for fd in self._descriptors:
                try:
                    # Only close: LOCK_UN would unlock the parent's shared flock.
                    os.close(fd)
                except OSError:
                    pass
            self._descriptors.clear()
        finally:
            self._guard.release()


# Preserve live registrations if this module is explicitly reloaded.
if "_descriptor_registry" not in globals():
    _descriptor_registry = _DescriptorRegistry()


def _acquire(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def output_directory_lock(output_dir: str):
    """Fail immediately if another cooperating generator owns the directory.

    The marker is deliberately retained: unlinking a locked file lets different
    processes acquire different inodes for the same path. OS locks, not marker
    existence or PID files, determine ownership and release on process death.
    Forked children close inherited descriptors without unlocking their parent.
    """
    directory = os.path.realpath(output_dir)
    os.makedirs(directory, exist_ok=True)
    lock_path = os.path.join(directory, _LOCK_NAME)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    owner_pid = os.getpid()
    registry = _descriptor_registry
    fd = registry.open(lock_path, flags, 0o600)
    locked = False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("Generation lock must be a regular file")
        try:
            _acquire(fd)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise OutputDirectoryBusyError(directory) from exc
            raise
        locked = True
        current = os.stat(lock_path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("Generation lock file changed while acquiring ownership; retry")
        _cleanup_generation_temporary_files(directory)
        yield
    finally:
        # A child may unwind the inherited context after opening other files
        # that reuse the old descriptor number. It must neither unlock nor close.
        if os.getpid() == owner_pid:
            handling_error = sys.exc_info()[0] is not None
            try:
                if locked:
                    _release(fd)
            except OSError as exc:
                if not handling_error:
                    raise
                logger.warning("Could not explicitly release generation lock while handling an error: %s", exc)
            finally:
                registry.close(fd)


def _cleanup_generation_temporary_files(output_dir: str) -> None:
    """Remove reserved temporary artifacts only while holding directory ownership.

    Never follow symlinks or remove committed shards, checkpoints, the lock
    marker, or files outside the exact tempfile naming convention used here.
    """
    with os.scandir(output_dir) as entries:
        for entry in entries:
            try:
                if _TEMP_FILE.fullmatch(entry.name) and entry.is_file(follow_symlinks=False):
                    os.unlink(entry.path)
                elif _TEMP_DIR.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
            except OSError as exc:
                logger.warning("Could not remove orphan generation temporary artifact %s: %s", entry.name, exc)
