"""Command line entry points.

    kinesteach list
    kinesteach teach --duration 20 --notes "pick from bin"
    kinesteach process episode_0001 --cutoff 10
    kinesteach replay episode_0001 --time-scale 2.0
    kinesteach report episode_0001
    kinesteach webui --backend mock
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import numpy as np

from . import common
from ..dataset import VALIDATION, list_episodes
from .calibrate import cmd_payload_fit, cmd_payload_sweep, cmd_workspace
from .common import HOME_ARRIVAL_TOL_RAD, print_issues

log = logging.getLogger("kinesteach")


def cmd_list(args) -> int:
    cfg = common._cfg(args)
    eps = list_episodes(cfg.data_root)
    if not eps:
        print("no episodes under %s" % cfg.data_root)
        return 0
    print("%-16s %8s %9s %9s %8s %7s  %s" % ("episode", "states", "duration", "rate", "backend", "checks", "notes"))
    for ep in eps:
        m = ep.read_metadata()
        v = ep.read_json(VALIDATION) if ep.file(VALIDATION).exists() else {}
        checks = "fail" if v.get("ok") is False else ("%dwarn" % len(v.get("warnings", [])) if v.get("warnings") else "ok")
        print(
            "%-16s %8d %8.1fs %8.0fHz %8s %7s  %s"
            % (ep.name, m.get("n_states", 0), m.get("duration_s", 0.0),
               m.get("effective_hz", 0.0), m.get("backend", "?"), checks, m.get("notes", ""))
        )
    return 0


def cmd_teach(args) -> int:
    from ..record import save_teaching_episode
    from ..teach import TeachingAborted, run_teaching

    cfg = common._cfg(args)
    aborted = None
    with common._connected(cfg) as (backend, guard):
        if args.home:
            home = common._checked_home_pose(backend, backend.spec())
            log.info("homing")
            backend.move_to_joint_positions(
                home, time_to_go=backend.expected_move_time_s(home), blocking=True)
            # Say whether it arrived. The controller has no integral term, so a
            # joint stalls where `Kq * error` balances whatever resists it and
            # stays there however long the move is given. On this arm j7 stops
            # ~0.26 rad short -- about 2.6 Nm of gripper cabling against its
            # 10 Nm/rad -- and homing used to report success anyway, which reads
            # as "the command did nothing".
            err = np.abs(backend.get_joint_positions() - home)
            if err.max() > HOME_ARRIVAL_TOL_RAD:
                bad = np.argsort(-err)[:3]
                log.warning(
                    "homing stopped %.3f rad short (%s); the arm is NOT at its "
                    "home pose. A joint that stalls has more resistance than "
                    "its stiffness can overcome, and waiting does not help.",
                    err.max(),
                    ", ".join("j%d %.3f rad" % (i + 1, err[i]) for i in bad),
                )
            else:
                log.info("homed, within %.3f rad", err.max())
        limit = cfg.teach.max_duration_s if args.duration is None else args.duration
        print(
            "teaching for up to %.1f s -- move the arm by hand now.\n"
            "  Ctrl-C to finish and keep the recording (twice to force a stop)."
            % limit
        )
        try:
            buf, gripper, controller = run_teaching(backend, cfg, args.duration, stop=guard)
        except TeachingAborted as exc:
            # The session broke after the policy was live, but the log came back
            # with the exception. Save it, then report the failure.
            aborted = exc
            buf, gripper, controller = exc.buf, exc.gripper, exc.controller
        if controller.get("ended_by") == "stop_requested":
            print("stopped by Ctrl-C; keeping what was recorded")
        ep = save_teaching_episode(
            cfg.data_root, buf, backend.spec(), cfg,
            controller=controller, gripper=gripper, notes=args.notes,
        )
    rep = ep.read_json(VALIDATION)
    print("saved %s: %d states, %.2f s, %.1f Hz" % (ep.name, buf.n, buf.duration, buf.effective_hz))
    print_issues(rep)
    if aborted is not None:
        print(
            "  the session FAILED and this episode is what could be recovered: %r"
            % (aborted.cause,)
        )
        return 1
    return 0 if rep["ok"] else 1


def cmd_process(args) -> int:
    from ..process import process_episode

    cfg = common._cfg(args)
    if args.cutoff is not None:
        cfg.process.cutoff_hz = args.cutoff
    if args.filter:
        cfg.process.filter = args.filter
    targets = list_episodes(cfg.data_root) if args.all else [common._find(cfg, args.episode)]
    for ep in targets:
        process_episode(ep, cfg)
        info = ep.read_metadata()["processing"]
        print("%s: %d samples at %.0f Hz, %s cutoff %.1f Hz%s"
              % (ep.name, info["n_uniform"], info["resampled_hz"], info["filter"],
                 info["cutoff_hz"], ", FK -> " + str(info["ee_frame"]) if info.get("ee_frame") else ""))
        for k, v in (info.get("cutoff_sweep") or {}).items():
            print("    %-12s rms_dev %.2e rad   accel_rms %.2f rad/s^2"
                  % (k, v["rms_deviation_rad"], v["accel_rms_rad_s2"]))
    return 0


def cmd_replay(args) -> int:
    from ..replay import replay_episode

    cfg = common._cfg(args)
    if args.time_scale is not None:
        cfg.replay.time_scale = args.time_scale
    if args.kq_scale is not None:
        cfg.replay.Kq_scale = args.kq_scale
    if args.start_tol_nm is not None:
        cfg.replay.start_pose_tol_nm = (
            None if args.start_tol_nm <= 0 else float(args.start_tol_nm)
        )
        log.warning("start-pose torque check %s",
                    "disabled; gating on the angle cap alone"
                    if cfg.replay.start_pose_tol_nm is None
                    else "set to %.2f Nm" % cfg.replay.start_pose_tol_nm)
    if args.start_tol is not None:
        log.warning("start-pose tolerance raised to %.4f rad (default %.4f): the "
                    "arm will begin the trajectory from further away than the "
                    "gate normally allows", args.start_tol, cfg.replay.start_pose_tol_rad)
        cfg.replay.start_pose_tol_rad = args.start_tol
    ep = common._find(cfg, args.episode)
    print(
        "replaying -- the arm moves on its own. Keep the physical E-stop in reach.\n"
        "  Ctrl-C stops it where it stands (twice to force a stop)."
    )
    with common._connected(cfg) as (backend, guard):
        buf, saved = replay_episode(backend, ep, cfg, notes=args.notes, stop=guard)
    print("replayed %s -> %s (%d states, %.2f s, %s)"
          % (ep.name, saved.path.name if saved else "(not saved)", buf.n,
             buf.duration, controller_ended_by(saved)))

    from ..validate import check_d_replay_comparison

    d = check_d_replay_comparison(ep.read_raw(), buf)
    print("  tracking rms %.5f rad, max %.5f rad" % (d["tracking_rms_rad"], d["tracking_max_rad"]))
    return 0


def controller_ended_by(saved) -> str:
    """How the replay finished, per the pass we just wrote."""
    if saved is None:
        return "not saved"
    try:
        return str(saved.read_metadata()["controller"].get("ended_by", "completed"))
    except Exception:
        return "completed"


def cmd_report(args) -> int:
    cfg = common._cfg(args)
    ep = common._find(cfg, args.episode)
    rep = ep.read_json(VALIDATION)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    print("%s -- %s" % (ep.name, "OK" if rep["ok"] else "FAILED"))
    print("  %d states, %.2f s, %.1f Hz (nominal %.0f Hz)"
          % (rep["n_states"], rep["duration_s"], rep["effective_hz"], rep.get("nominal_hz", 0)))
    dt = rep["dt"]
    print("  dt: median %.4f ms, p99 %.4f ms, max %.4f ms" % (dt["median_ms"], dt["p99_ms"], dt["max_ms"]))
    print("  dropped ~%d, gaps>2x: %d, buffer fill %.1f%%"
          % (rep["estimated_dropped_samples"], rep["n_gaps_over_2x"], 100 * rep["buffer_fill"]))
    print("  |tau_external| max: %s" % np.round(rep["tau_external_absmax"], 2).tolist())
    print_issues(rep)
    return 0 if rep["ok"] else 1


def cmd_webui(args) -> int:
    import uvicorn

    from ..webui.server import create_app

    cfg = common._cfg(args)
    log.info(
        "serving http://%s:%d  (backend=%s, data_root=%s)",
        args.host, args.http_port, cfg.backend.kind, cfg.data_root,
    )
    uvicorn.run(
        create_app(cfg), host=args.host, port=args.http_port, log_level=args.log_level
    )
    return 0


# ------------------------------------------------------------------- main


def _shared_parser() -> argparse.ArgumentParser:
    """Options that apply to every subcommand."""
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument("--config", default=None, help="YAML config file")
    q.add_argument("--data-root", default=None)
    q.add_argument("--log-level", default=None)
    return q


def build_parser() -> argparse.ArgumentParser:
    common = _shared_parser()
    p = argparse.ArgumentParser(
        prog="kinesteach", description="Franka kinesthetic teaching", parents=[common]
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def robot_args(sp, port_flag="--port"):
        sp.add_argument("--backend", choices=["mock", "real"], default=None)
        sp.add_argument("--ip", default=None, help="polymetis server address")
        # `webui` needs --port for its own HTTP listener, so the robot port is
        # spelled differently there.
        sp.add_argument(port_flag, dest="port", type=int, default=None,
                        help="polymetis server port")
        sp.add_argument("--gripper", action="store_true", help="also poll the Robotiq")

    sp = sub.add_parser("list", help="list episodes", parents=[common]); sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("teach", help="run a teaching session", parents=[common])
    robot_args(sp)
    sp.add_argument("--duration", type=float, default=None,
                    help="upper bound in seconds; Ctrl-C ends it sooner and keeps the log")
    sp.add_argument("--notes", default="")
    sp.add_argument("--home", action="store_true", help="go home before teaching")
    sp.add_argument("--kqd-scale", type=float, default=None, dest="kqd_scale",
                    help="scale the robot's default damping for this run "
                         "(0 disables damping entirely); overrides teach.Kqd_scale")
    sp.set_defaults(fn=cmd_teach)

    sp = sub.add_parser("workspace", parents=[common],
                        help="hand-walk the region the arm may drive itself through")
    robot_args(sp)
    sp.add_argument("--duration", type=float, default=180.0)
    sp.add_argument("--out", default=None, help="where to write the envelope")
    sp.add_argument("--vertices", type=int, default=300,
                    help="how many configurations to keep as hull vertices")
    sp.add_argument("--kqd-scale", type=float, default=None, dest="kqd_scale",
                    help="damping for this walk; higher slows the sag while you "
                         "reposition your grip")
    sp.add_argument("--guided", dest="guided", action="store_true", default=None,
                    help="prompt through the walk step by step (default on a tty)")
    sp.add_argument("--no-guided", dest="guided", action="store_false",
                    help="just record for --duration seconds")
    sp.set_defaults(fn=cmd_workspace)

    sp = sub.add_parser("payload-fit", parents=[common],
                        help="re-fit saved payload-sweep measurements")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--urdf", default=None)
    sp.set_defaults(fn=cmd_payload_fit)

    sp = sub.add_parser("payload-sweep", parents=[common],
                        help="drive to computed poses and fit the unregistered load")
    robot_args(sp)
    sp.add_argument("--poses", type=int, default=10)
    sp.add_argument("--candidates", type=int, default=400)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--settle", type=float, default=3.0,
                    help="seconds to let the arm stop sagging before sampling")
    sp.add_argument("--dwell", type=float, default=2.0, help="sampling window per pose")
    sp.add_argument("--slow", type=float, default=2.0, help="multiplier on the planned move time")
    sp.add_argument("--envelope", default=None,
                    help="joint-space envelope from `kinesteach workspace` "
                         "(default: <data_root>/../workspace/envelope.json)")
    sp.add_argument("--hull-k", type=int, default=3, dest="hull_k",
                    help="vertices blended per sampled pose")
    sp.add_argument("--hull-alpha", type=float, default=0.3, dest="hull_alpha",
                    help="<1 spreads samples towards the envelope boundary")
    sp.add_argument("--no-envelope", action="store_true", dest="no_envelope",
                    help="ignore any envelope and use a bare floor/reach check")
    sp.add_argument("--min-z", type=float, default=0.25, dest="min_z",
                    help="lowest flange height allowed, metres")
    sp.add_argument("--limit-margin", type=float, default=0.35, dest="limit_margin",
                    help="rad to stay clear of every joint limit")
    sp.add_argument("--still-thresh", type=float, default=2e-3, dest="still_thresh")
    sp.add_argument("--min-samples", type=int, default=30, dest="min_samples")
    sp.add_argument("--home-first", action="store_true", dest="home_first",
                    help="go to the home pose before the first measured pose; "
                         "the approach is path-checked like any other leg")
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp.set_defaults(fn=cmd_payload_sweep)

    sp = sub.add_parser("process", help="filter, differentiate, FK", parents=[common])
    sp.add_argument("episode", nargs="?")
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--cutoff", type=float, default=None)
    sp.add_argument("--filter", choices=["butterworth", "savgol"], default=None)
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("replay", help="replay an episode", parents=[common])
    robot_args(sp)
    sp.add_argument("episode")
    sp.add_argument("--time-scale", type=float, default=None, help=">1 is slower than the demo")
    sp.add_argument("--kq-scale", type=float, default=None)
    sp.add_argument("--start-tol-nm", type=float, default=None, dest="start_tol_nm",
                    help="start-pose tolerance on Kq*error, in Nm. Pass 0 to "
                         "disable the torque check and gate on the angle alone, "
                         "which is what runs recorded before it existed used.")
    sp.add_argument("--start-tol", type=float, default=None, dest="start_tol",
                    help="start-pose tolerance in rad; raising it lets a replay "
                         "run that the gate would refuse, which is how the cost "
                         "of an unregistered load gets measured rather than "
                         "predicted. Say why in --notes.")
    sp.add_argument("--notes", default="")
    sp.set_defaults(fn=cmd_replay)

    sp = sub.add_parser("report", help="print an episode's validation report", parents=[common])
    sp.add_argument("episode")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("webui", help="serve the web interface", parents=[common])
    robot_args(sp, port_flag="--robot-port")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--http-port", type=int, default=8000)
    sp.set_defaults(fn=cmd_webui)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pull the shared options out first so they work on either side of the
    # subcommand: argparse hands a subparser a fresh namespace and copies it
    # back over the parent's, which would otherwise silently discard a
    # `kinesteach --data-root X list`. They stay declared on both parsers so `--help`
    # still shows them.
    shared, rest = _shared_parser().parse_known_args(argv)
    args = build_parser().parse_args(rest)
    args.config = shared.config
    args.data_root = shared.data_root
    args.log_level = shared.log_level or "info"
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.cmd == "process" and not args.all and not args.episode:
        raise SystemExit("process needs an episode name, or --all")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
