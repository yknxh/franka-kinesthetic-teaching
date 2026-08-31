import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinesteach.backend import make_backend  # noqa: E402
from kinesteach.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = Config.default()
    c.data_root = str(tmp_path / "episodes")
    return c


@pytest.fixture
def backend(cfg):
    b = make_backend(cfg.backend)
    b.connect()
    yield b
    b.close()


@pytest.fixture
def episode(cfg, backend):
    """A saved 0.5 s mock teaching episode."""
    import time

    from kinesteach.record import save_teaching_episode
    from kinesteach.teach import TeachingSession

    s = TeachingSession(backend, cfg)
    s.start()
    time.sleep(0.5)
    buf, gripper = s.stop()
    return save_teaching_episode(
        cfg.data_root, buf, backend.spec(), cfg,
        controller=s.controller_metadata, gripper=gripper, notes="fixture",
    )


@pytest.fixture
def stub_connected(monkeypatch):
    """Point `cli._connected` at an already-open backend.

    `_connected` is a context manager, so the stub has to be one too -- a
    function returning a `(backend, guard)` pair would leave the command's
    `with` statement raising AttributeError rather than exercising it.
    """
    from contextlib import contextmanager

    class _Guard:
        def fire(self): pass
        def uninstall(self): pass
        def __call__(self): return False

    def install(backend, guard=None):
        guard = guard or _Guard()

        @contextmanager
        def _fake(_cfg):
            yield backend, guard

        monkeypatch.setattr("kinesteach.cli.common._connected", _fake)
        return guard

    return install
