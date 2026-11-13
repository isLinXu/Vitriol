"""Run fork regressions in clean interpreters, isolated from pytest's ML threads."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")


def run_script(script, tmp_path):
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True, text=True, timeout=25,
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("directories", [1, 3])
@pytest.mark.parametrize("reload_module", [False, True])
def test_forked_context_exit_does_not_unlock_parent(tmp_path, directories, reload_module):
    run_script(r'''
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from vitriol.core.output_lock import output_directory_lock
from vitriol.utils.exceptions import OutputDirectoryBusyError

paths = [str(Path(sys.argv[1]) / str(i)) for i in range(DIRECTORIES)]
child = False
with ExitStack() as stack:
    for path in paths:
        stack.enter_context(output_directory_lock(path))
    if RELOAD_MODULE:
        import importlib
        import vitriol.core.output_lock as module
        importlib.reload(module)
    pid = os.fork()
    if pid == 0:
        child = True
    else:
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        for path in paths:
            try:
                with output_directory_lock(path):
                    raise AssertionError("child unlocked parent's directory")
            except OutputDirectoryBusyError:
                pass
if child:
    os._exit(0)
for path in paths:
    with output_directory_lock(path):
        pass
'''.replace("DIRECTORIES", str(directories)).replace("RELOAD_MODULE", str(reload_module)), tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="inspects /proc/self/fd")
def test_fork_closes_inherited_descriptors_from_another_thread(tmp_path):
    run_script(r'''
import os
import sys
import threading
from vitriol.core.output_lock import output_directory_lock

entered, release = threading.Event(), threading.Event()
def hold():
    with output_directory_lock(sys.argv[1]):
        entered.set()
        assert release.wait(10)
owner = threading.Thread(target=hold)
owner.start()
assert entered.wait(10)
pid = os.fork()
if pid == 0:
    marker = os.stat(os.path.join(sys.argv[1], '.vitriol-generation.lock'))
    inherited = []
    for name in os.listdir('/proc/self/fd'):
        try:
            current = os.fstat(int(name))
            if (current.st_dev, current.st_ino) == (marker.st_dev, marker.st_ino):
                inherited.append(name)
        except OSError:
            pass
    os._exit(1 if inherited else 0)
try:
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "child inherited a writer lock descriptor"
finally:
    release.set()
    owner.join(10)
''', tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux child subreaper to reap orphan workers")
def test_live_worker_does_not_keep_dead_owners_lock(tmp_path):
    run_script(r'''
import ctypes
import os
import sys
from vitriol.core.output_lock import output_directory_lock

# Reap the worker after its owner exits, without leaving a test orphan behind.
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
ready_r, ready_w = os.pipe()
exit_r, exit_w = os.pipe()
owner = os.fork()
if owner == 0:
    os.close(ready_r)
    os.close(exit_w)
    with output_directory_lock(sys.argv[1]):
        worker = os.fork()
        if worker == 0:
            os.close(ready_w)
            os.read(exit_r, 1)
            os._exit(0)
        os.write(ready_w, str(worker).encode())
        os._exit(73)
os.close(ready_w)
os.close(exit_r)
worker = int(os.read(ready_r, 64))
os.close(ready_r)
try:
    _, status = os.waitpid(owner, 0)
    assert os.waitstatus_to_exitcode(status) == 73
    with output_directory_lock(sys.argv[1]):
        pass  # The worker is still alive; it must not keep the dead owner's lock.
finally:
    os.write(exit_w, b'x')
    os.close(exit_w)
    os.waitpid(worker, 0)
''', tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="inspects /proc/self/fd")
def test_fork_waits_until_opened_descriptor_is_registered(tmp_path):
    run_script(r'''
import os
import sys
import threading
import vitriol.core.output_lock as module

opened, release_open, entered, release_owner = [threading.Event() for _ in range(4)]
actual_open = os.open
def delayed_open(*args):
    fd = actual_open(*args)
    opened.set()
    assert release_open.wait(10)
    return fd
module.os.open = delayed_open
def hold():
    with module.output_directory_lock(sys.argv[1]):
        entered.set()
        assert release_owner.wait(10)
owner = threading.Thread(target=hold)
owner.start()
assert opened.wait(10)
# Before-hooks run in reverse registration order: release the blocked open,
# then the registry must wait for its registration before allowing the fork.
os.register_at_fork(before=release_open.set)
pid = os.fork()
if pid == 0:
    marker = os.stat(os.path.join(sys.argv[1], '.vitriol-generation.lock'))
    inherited = []
    for name in os.listdir('/proc/self/fd'):
        try:
            current = os.fstat(int(name))
            if (current.st_dev, current.st_ino) == (marker.st_dev, marker.st_ino):
                inherited.append(name)
        except OSError:
            pass
    os._exit(1 if inherited else 0)
try:
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "unregistered descriptor escaped into child"
    assert entered.wait(10)
finally:
    release_owner.set()
    owner.join(10)
''', tmp_path)


def test_inherited_context_does_not_close_reused_child_descriptor(tmp_path):
    run_script(r'''
import os
import sys
from pathlib import Path
import vitriol.core.output_lock as module

acquire = module._acquire
owned_fd = None
def capture(fd):
    global owned_fd
    owned_fd = fd
    return acquire(fd)
module._acquire = capture
child = False
with module.output_directory_lock(sys.argv[1]):
    pid = os.fork()
    if pid == 0:
        child = True
        replacement = os.open(str(Path(sys.argv[1]) / 'child-file'), os.O_RDWR | os.O_CREAT, 0o600)
        if replacement != owned_fd:
            os.dup2(replacement, owned_fd)
            os.close(replacement)
    else:
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
if child:
    os.write(owned_fd, b'child still owns this fd')
    os.close(owned_fd)
    os._exit(0)
assert (Path(sys.argv[1]) / 'child-file').read_bytes() == b'child still owns this fd'
''', tmp_path)


def test_child_can_acquire_its_own_directory_and_fork_again(tmp_path):
    run_script(r'''
import os
import sys
from pathlib import Path
from vitriol.core.output_lock import output_directory_lock
from vitriol.utils.exceptions import OutputDirectoryBusyError

parent_path = str(Path(sys.argv[1]) / 'parent')
child_path = str(Path(sys.argv[1]) / 'child')
child = False
with output_directory_lock(parent_path):
    pid = os.fork()
    if pid == 0:
        child = True
        try:
            with output_directory_lock(parent_path):
                raise AssertionError('parent lock was lost')
        except OutputDirectoryBusyError:
            pass
        with output_directory_lock(child_path):
            grandchild = os.fork()
            if grandchild != 0:
                _, status = os.waitpid(grandchild, 0)
                assert os.waitstatus_to_exitcode(status) == 0
                try:
                    with output_directory_lock(child_path):
                        raise AssertionError('grandchild unlocked child')
                except OutputDirectoryBusyError:
                    pass
        if grandchild == 0:
            os._exit(0)
    else:
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
if child:
    with output_directory_lock(child_path):
        pass
    os._exit(0)
''', tmp_path)
