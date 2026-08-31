"""FastAPI front end.

Commands go in over HTTP and land on the worker's queue; state comes back over
a WebSocket at display rate. No handler ever calls the robot directly -- see
worker.py for why.

Array endpoints downsample before serialising. A five-minute episode is 300k
samples per joint; sending that to a browser helps nobody, and the plots this
feeds are for judging shape, not for measurement.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config
from ..dataset import CUTOFF_SWEEP, VALIDATION, Episode, list_episodes
from .worker import Worker

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

__all__ = ["create_app"]


def _downsample(x: np.ndarray, max_points: int) -> np.ndarray:
    """Stride down to at most `max_points` rows.

    Plain striding, not averaging: these plots exist to show the shape of the
    trajectory, and averaging would hide exactly the spikes worth seeing.
    """
    n = x.shape[0]
    if n <= max_points:
        return x
    return x[:: int(np.ceil(n / max_points))]


def _jsonable(a: np.ndarray) -> Any:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        a = a.astype(np.int8)
    return np.where(np.isfinite(a), a, None).tolist() if a.dtype.kind == "f" else a.tolist()


def create_app(cfg: Optional[Config] = None) -> FastAPI:
    cfg = cfg or Config.default()
    app = FastAPI(title="Franka Kinesthetic Teaching")
    worker = Worker(cfg)
    app.state.cfg = cfg
    app.state.worker = worker

    @app.on_event("startup")
    def _startup() -> None:
        worker.start()
        worker.submit("connect")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        worker.shutdown()
        worker.join(timeout=5.0)

    # ---- pages ---------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/vendor/plotly.min.js", include_in_schema=False)
    def plotly_js() -> FileResponse:
        """Serve plotly.js out of the installed python package.

        The workstation is on the robot network and may have no route to a
        CDN, and a plotting library that only loads sometimes is worse than
        one that is simply there. The `plotly` conda package already ships
        the bundle, so this costs nothing.
        """
        import plotly

        js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if not js.exists():
            raise HTTPException(404, "plotly.min.js not found in the plotly package")
        return FileResponse(js, media_type="application/javascript")

    # ---- control -------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        return {"config": cfg.to_dict(), "data_root": str(Path(cfg.data_root).resolve())}

    @app.get("/api/state")
    def get_state() -> Dict[str, Any]:
        return worker.snapshot()

    @app.post("/api/command")
    def post_command(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        name = body.get("name")
        if not name:
            raise HTTPException(400, "command needs a 'name'")
        return worker.submit(str(name), **(body.get("args") or {}))

    @app.get("/api/telemetry/history")
    def telemetry_history() -> Dict[str, Any]:
        return {"samples": worker.history()}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        try:
            while True:
                await sock.send_json(worker.snapshot())
                await asyncio.sleep(1.0 / 20.0)
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover
            log.exception("websocket closed unexpectedly")

    # ---- episodes ------------------------------------------------------

    def _episode(name: str) -> Episode:
        for ep in list_episodes(cfg.data_root):
            if ep.name == name:
                return ep
        raise HTTPException(404, "no episode named %r" % name)

    @app.get("/api/episodes")
    def episodes() -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for ep in list_episodes(cfg.data_root):
            try:
                m = ep.read_metadata()
            except Exception:
                continue
            v = ep.read_json(VALIDATION) if ep.file(VALIDATION).exists() else {}
            out.append(
                {
                    "name": ep.name,
                    "created_at": m.get("created_at"),
                    "duration_s": m.get("duration_s"),
                    "n_states": m.get("n_states"),
                    "effective_hz": m.get("effective_hz"),
                    "backend": m.get("backend"),
                    "num_dofs": m.get("num_dofs"),
                    "notes": m.get("notes", ""),
                    "processed": ep.has_processed(),
                    "n_replays": len(ep.replay_passes()),
                    "ok": v.get("ok"),
                    "n_warnings": len(v.get("warnings", [])),
                }
            )
        return {"episodes": out}

    @app.get("/api/episodes/{name}")
    def episode_detail(name: str) -> Dict[str, Any]:
        ep = _episode(name)
        return {
            "name": ep.name,
            "metadata": ep.read_metadata(),
            "validation": ep.read_json(VALIDATION) if ep.file(VALIDATION).exists() else None,
            "processed": ep.has_processed(),
            "replays": [p.name for p in ep.replay_passes()],
        }

    @app.get("/api/episodes/{name}/series")
    def episode_series(
        name: str,
        keys: str = Query("q", description="comma-separated array names"),
        source: str = Query("raw", pattern="^(raw|processed|sweep|replay)$"),
        replay: Optional[str] = None,
        max_points: int = Query(2000, ge=50, le=50000),
    ) -> Dict[str, Any]:
        ep = _episode(name)
        if source == "replay":
            passes = {p.name: p for p in ep.replay_passes()}
            if replay not in passes:
                raise HTTPException(404, "no replay pass %r (have %s)" % (replay, sorted(passes)))
            buf = passes[replay].read_raw()
            arrays = dict(buf.as_dict(), t=buf.t)
        elif source == "raw":
            buf = ep.read_raw()
            arrays = dict(buf.as_dict(), t=buf.t)
        elif source == "sweep":
            if not ep.file(CUTOFF_SWEEP).exists():
                raise HTTPException(404, "no cutoff sweep; process the episode first")
            arrays = ep.read_arrays(CUTOFF_SWEEP)
        else:
            if not ep.has_processed():
                raise HTTPException(404, "not processed yet")
            arrays = ep.read_processed()

        wanted = [k.strip() for k in keys.split(",") if k.strip()]
        missing = [k for k in wanted if k not in arrays]
        if missing:
            raise HTTPException(400, "unknown array(s) %s; have %s" % (missing, sorted(arrays)))
        # `t` rides along so the client always has an x-axis for what it asked for.
        out = {k: _jsonable(_downsample(arrays[k], max_points)) for k in wanted}
        if "t" in arrays and "t" not in out:
            out["t"] = _jsonable(_downsample(arrays["t"], max_points))
        return {"name": ep.name, "source": source, "arrays": out, "available": sorted(arrays)}

    @app.post("/api/episodes/{name}/process")
    def episode_process(name: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        ep = _episode(name)
        worker.submit("process", episode=ep.name, cutoff_hz=body.get("cutoff_hz"))
        return {"queued": True, "episode": ep.name}

    return app


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Franka kinesthetic teaching WebUI")
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backend", default=None, choices=["mock", "real"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = Config.load(args.config)
    if args.backend:
        cfg.backend.kind = args.backend
    if args.data_root:
        cfg.data_root = args.data_root

    log.info("backend=%s data_root=%s", cfg.backend.kind, cfg.data_root)
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level=args.log_level)
    return 0
