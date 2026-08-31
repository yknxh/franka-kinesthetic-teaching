"""The hand-walked safe region, and the load identification beside it.

    kinesteach workspace                 walk the region by hand
    kinesteach payload-sweep             drive inside it and weigh the load
    kinesteach payload-fit sweep.json    re-fit saved measurements

Split from `cli.py` because these are not part of the teach/process/replay
pipeline at all. `workspace` produces a safety artefact for the autonomous
tools; `payload-sweep` is the only command that moves the arm to poses nobody
demonstrated, and it is the longest and most guarded thing the CLI does.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import common
from ..dataset import list_episodes
from .common import default_envelope

log = logging.getLogger("kinesteach")


#: Below this, the widest joint travelled less than ~3 deg: not a region.
WORKSPACE_MIN_SPAN_RAD = 0.05

#: The walk, as steps the operator can act on one at a time. Each says what it
#: pins down, because an operator who knows why a step exists can tell when
#: they have done enough of it.
#:
#: Mass and centre of mass are fixed by different motions and neither
#: substitutes for the other: mass only shows up through the moment arm, so it
#: needs the reach to change, while the centre of mass only shows up when the
#: flange's own axes turn relative to gravity. Covering a wide xy area with the
#: wrist always pointing down leaves the centre of mass unidentifiable.
WORKSPACE_STEPS = (
    ("reach", "Reach the arm out as far as you are willing to let it go.",
     "fixes the mass: it is only visible through the moment arm"),
    ("fold", "Fold it back in towards the base.",
     "the other end of that same lever"),
    ("left", "Swing to the left edge of the allowed region.", "sets the width"),
    ("right", "Swing to the right edge.", "sets the width"),
    ("high", "Raise the flange as high as you will allow.", "sets the height"),
    ("low", "Lower it as far as you will allow, keeping clear of the table.",
     "sets the floor -- stop short of anything it could hit"),
    ("wrist-down", "Back near the middle: point the gripper straight down.",
     "starts on the centre of mass"),
    ("wrist-left", "Roll the wrist so the gripper lies over to one side.",
     "centre of mass, sideways component"),
    ("wrist-right", "Roll it the other way.", "centre of mass, the other side"),
    ("wrist-up", "Tip the gripper up. It does not have to pass horizontal -- "
     "about 60 deg off straight down is enough.",
     "centre of mass, along the flange axis"),
    ("wrist-far", "THE ONE THAT MATTERS. Now stretch the arm right out and "
     "repeat those same wrist turns there -- then again with the arm folded "
     "in. Do not stay in the middle.",
     "the first usable walk turned the wrist only at 0.55 m and was unusable "
     "for it; two poses turning it with the arm extended fixed the whole fit"),
    ("free", "Anything else you want inside the region. Nothing outside it.",
     "last chance to widen it"),
)


def _wait_for_step(session) -> str:
    """Count up on one line from the moment the instruction appears.

    Polls stdin rather than blocking on `input()` so the operator, who has both
    hands on the arm and is reading rather than watching, can see how long this
    step has taken and how much of the walk is left. The first usable walk spent
    a quarter of its 280 s with the arm standing still and ran out before the
    step that mattered, with nothing on screen to show it.
    """
    import select
    import sys
    import time

    started = time.monotonic()
    while True:
        held = time.monotonic() - started
        sys.stdout.write("\r         %4.0f s on this step   (walk: %3.0f s left)"
                         "   Enter when done " % (held, session.remaining))
        sys.stdout.flush()
        if session.should_stop():
            print("\n  reached the session limit; wrapping up")
            return "duration_limit"
        if select.select([sys.stdin], [], [], 0.25)[0]:
            sys.stdin.readline()
            print("\r         %4.0f s on this step   (walk: %3.0f s left)"
                  "            " % (held, session.remaining))
            return "operator"


def _guided_walk(session, steps=WORKSPACE_STEPS) -> str:
    """Prompt through the walk while the policy stays live.

    One keystroke per step, with the clock running from the moment the
    instruction lands. The server logs at its own rate throughout, so reading
    time costs nothing but the ring buffer -- which is why the counter shows
    what is left of the whole walk, not just of this step.
    """
    for i, (_, what, why) in enumerate(steps, 1):
        print()
        print("  [%2d/%d] %s" % (i, len(steps), what))
        print("         (%s)" % why)
        try:
            ended = _wait_for_step(session)
        except (EOFError, KeyboardInterrupt):
            print()
            return "stopped_by_operator"
        if ended == "duration_limit":
            return "duration_limit"
    return "completed"


def cmd_workspace(args) -> int:
    """Record, by hand, the region the arm may move through on its own.

    A teaching session with a different purpose: the operator walks the arm
    around the space they are willing to let it drive itself through, and the
    configurations they passed become the vertices of a convex region in joint
    space. `payload-sweep` then samples inside it, which keeps not just its
    poses but every straight-line move between them inside what was walked.

    Deliberately outside the teach/process/replay pipeline -- this is a safety
    artefact for the autonomous tools, not part of any episode.
    """
    from ..envelope import save_envelope
    from ..kinematics import fk_from_urdf_text
    from ..safety import policy_guard
    from ..teach import TeachingAborted, TeachingSession, run_teaching

    cfg = common._cfg(args)
    with common._connected(cfg) as (backend, guard):
        spec = backend.spec()
        if not spec.urdf_text:
            raise SystemExit("the server sent no URDF")
        print(
            "Walk the arm around the region it may later drive itself through.\n"
            "  Only the extremes matter -- the region is the convex hull of where\n"
            "  you go, so anything between two poses you visit is included, and\n"
            "  wandering about inside adds nothing.\n"
            "  Nothing may be left standing inside it: guiding *around* an object\n"
            "  puts that object in the hull."
        )
        guided = args.guided if args.guided is not None else sys.stdin.isatty()
        if not guided:
            try:
                buf, _, _ = run_teaching(backend, cfg, args.duration, stop=guard)
            except TeachingAborted as exc:
                buf = exc.buf
        else:
            session = TeachingSession(backend, cfg)
            with policy_guard(backend):
                session.start()
                try:
                    ended = _guided_walk(session)
                finally:
                    buf, _ = session.stop()
            print("walk %s: %d states over %.0f s" % (ended, buf.n, buf.duration))

    if buf.n < 100:
        raise SystemExit("only %d state(s) recorded; nothing to build from" % buf.n)
    fk = fk_from_urdf_text(spec.urdf_text, spec.ee_link_name)
    step = max(1, buf.n // args.vertices)
    qs = buf.q[::step]

    # A walk in which the arm never moved is not a small region, it is a failed
    # session -- and it used to be saved as a success. The first real walk
    # recorded 203 configurations spanning 0.0008 deg, which is encoder noise:
    # the joints were still locked. Refuse it here rather than let payload-sweep
    # inherit a degenerate hull it would have to sample inside.
    span = float(np.ptp(qs, axis=0).max())
    if span < WORKSPACE_MIN_SPAN_RAD:
        raise SystemExit(
            "the arm did not move: the widest joint spans %.4f deg over %d state(s).\n"
            "Nothing was saved. Push the arm by hand before starting -- if it does\n"
            "not follow, the joints are still locked (Desk) and no gain we set can\n"
            "free them." % (np.degrees(span), buf.n)
        )

    dest = Path(args.out or default_envelope(cfg))
    data = save_envelope(str(dest), qs, spec, fk)
    lo, hi = data["flange_extent_m"]["min"], data["flange_extent_m"]["max"]
    print("envelope: %d vertices from %d state(s)" % (data["n_vertices"], buf.n))
    print("  flange x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f"
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    print("saved %s" % dest)
    return 0


def cmd_payload_fit(args) -> int:
    """Re-fit a saved sweep, or several of them pooled together."""
    from ..kinematics import fk_from_urdf_text
    from ..payload import envelope_report, estimate_payload

    qs, taus, urdf, ee = [], [], None, None
    for path in args.files:
        d = json.loads(Path(path).read_text())
        for row in d["measurements"]:
            qs.append(np.asarray(row["q"], dtype=np.float64))
            taus.append(np.asarray(row["tau_external"], dtype=np.float64))
        ee = ee or d.get("robot", {}).get("ee_link_name")
    if len(qs) < 2:
        raise SystemExit("only %d measurement(s) across %d file(s)" % (len(qs), len(args.files)))

    urdf = Path(args.urdf).read_text() if args.urdf else None
    if urdf is None:
        cfg = common._cfg(args)
        eps = list_episodes(cfg.data_root)
        if not eps:
            raise SystemExit("pass --urdf; no saved episode to take one from")
        urdf = eps[-1].file("robot.urdf").read_text()
    fk = fk_from_urdf_text(urdf, ee or "panda_link8")

    fit = estimate_payload(qs, taus, fk)
    _print_payload_fit(fit, envelope_report(qs, fk))
    return 0


def _print_payload_fit(fit, report=None) -> None:
    print("unregistered load, from %d pose(s):" % fit["n_poses"])
    print("  mass          %.3f +/- %.3f kg" % (fit["mass_kg"], fit["mass_stderr_kg"]))
    print("  CoM (flange)  [%.4f %.4f %.4f] m" % tuple(fit["com_flange_m"]))
    print("  sensor bias   %s Nm" % np.array2string(np.array(fit["bias_nm"]), precision=3))
    print("  residual      rms %.4f / max %.4f Nm  (signal rms %.4f)"
          % (fit["residual_rms_nm"], fit["residual_max_nm"], fit["signal_rms_nm"]))
    print("  condition     %.1f, %s rank" % (fit["condition_number"],
                                             "full" if fit["full_rank"] else "DEFICIENT"))
    if fit["sign_flipped"]:
        print("  NOTE tau_external reports the load's torque negated; mass is |m|.")
    if report is not None:
        print("  poses         %s" % report["advice"])


def _sweep_candidates(args, cfg, spec, fk, rng):
    """The pool of poses the sweep is allowed to draw from.

    Fail closed: joint limits and a floor say nothing about the table or the
    cabling, so an autonomous sweep needs a region a human walked.
    """
    from ..envelope import load_envelope, sample_hull
    from ..payload import sample_poses

    env_path = args.envelope or str(default_envelope(cfg))
    if Path(env_path).exists() and not args.no_envelope:
        env = load_envelope(env_path)
        if int(env["num_dofs"]) != spec.num_dofs:
            raise SystemExit("%s is for a %d DOF robot, this one has %d"
                             % (env_path, env["num_dofs"], spec.num_dofs))
        print("envelope %s: %d vertices walked by hand" % (env_path, env["n_vertices"]))
        pool = sample_hull(env["vertices_q"], args.candidates, rng=rng,
                           k=args.hull_k, alpha=args.hull_alpha)
    elif args.no_envelope:
        pool = sample_poses(spec, fk, args.candidates, rng=rng,
                            limit_margin_rad=args.limit_margin,
                            min_flange_z=args.min_z)
    else:
        raise SystemExit(
            "refusing to move without a verified region. Record one with\n"
            "  kinesteach workspace --config <cfg>\n"
            "or pass --no-envelope to fall back on a floor and reach check that\n"
            "knows nothing about the table."
        )

    # Poses drawn from the hull are legal by construction, but the operator may
    # have walked close to a limit; drop anything the spec rejects.
    lo = np.asarray(spec.joint_pos_min) + args.limit_margin
    hi = np.asarray(spec.joint_pos_max) - args.limit_margin
    pool = [q for q in pool if np.all(q >= lo) and np.all(q <= hi)]
    if len(pool) < args.poses:
        raise SystemExit("only %d candidate pose(s); loosen --min-z" % len(pool))
    return pool


def _sweep_plan(args, backend, spec, fk, pool):
    """(approach, legs, skipped): where the arm will go, in order, checked.

    Nothing has moved when this returns. Every pose and every straight line
    between consecutive poses has been verified against the joint limits and a
    Cartesian floor, so the caller's job is to print it and ask.
    """
    from ..envelope import path_is_safe
    from ..payload import select_poses

    plan = [pool[i] for i in select_poses(pool, fk, args.poses)]
    start = backend.get_joint_positions()

    # Homing first is a convenience when the arm was left somewhere awkward, but
    # `go_home` moves autonomously without checking anything, which is exactly
    # the wrong tool for an awkward pose. Treat home as an ordinary waypoint
    # instead: same floor, same limit margin, same slow move, and refuse the
    # whole run if the approach to it does not clear.
    approach = []
    if args.home_first:
        home = common._checked_home_pose(
            backend, spec,
            limit_margin_rad=args.limit_margin - 0.1,
            min_flange_z=args.min_z - 0.05)
        approach = [home]
        start = home

    # Visit them nearest-first from where the arm already is, so the moves stay
    # short.
    ordered, remaining, here = [], list(plan), start
    while remaining:
        j = int(np.argmin([np.linalg.norm(q - here) for q in remaining]))
        ordered.append(remaining.pop(j))
        here = ordered[-1]

    # Inside the hull every leg is safe by construction, but the arm starts
    # wherever it was left, which need not be inside. Check the approach against
    # the floor whatever region we are using.
    legs, skipped, here = [], [], start
    for i, q in enumerate(ordered):
        if args.no_envelope or i == 0:
            ok, why = path_is_safe(here, q, fk, spec,
                                   limit_margin_rad=args.limit_margin - 0.1,
                                   min_flange_z=args.min_z - 0.05)
        else:
            ok, why = True, "ok"
        if ok:
            legs.append(q)
            here = q
        else:
            skipped.append("leg %d: %s" % (i + 1, why))
    if len(legs) < 2:
        raise SystemExit("only %d reachable pose(s) after the path check" % len(legs))
    return approach, legs, skipped


def _confirm_sweep(args, fk, approach, legs, skipped) -> bool:
    """Print the whole plan and get a yes. False means nothing should move."""
    print("payload sweep: %d pose(s), %d skipped as unsafe" % (len(legs), len(skipped)))
    for q in approach:
        pos, _ = fk.fk(q[None])
        print("   0  home (transit, not measured)  flange=[%.3f %.3f %.3f]" % tuple(pos[0]))
    for w in skipped:
        print("  skipped: %s" % w)
    for i, q in enumerate(legs):
        pos, _ = fk.fk(q[None])
        print("  %2d  q=%s  flange=[%.3f %.3f %.3f]"
              % (i + 1, np.array2string(q, precision=2, suppress_small=True), *pos[0]))
    print("THE ARM WILL MOVE ON ITS OWN. Keep the physical E-stop in reach.")
    if args.yes:
        return True
    return input("proceed? [y/N] ").strip().lower() in ("y", "yes")


def _dwell_at_pose(args, backend):
    """Sample one settled pose. Returns the samples that were actually still."""
    import time

    samples = []
    t_end = time.time() + args.dwell
    while time.time() < t_end:
        s = backend.get_telemetry()
        if s.error_code:
            raise SystemExit("robot reports error_code %d; stopping" % s.error_code)
        samples.append(s)
        time.sleep(0.02)
    still = [s for s in samples if np.abs(s.dq).max() < args.still_thresh]
    return samples, still


def _measure_pose(i, q, still) -> dict:
    """One row of the sweep: what was asked for, what came back, how healthy."""
    q_meas = np.mean([s.q for s in still], axis=0)
    tau = np.mean([s.tau_external for s in still], axis=0)
    # Record what we asked for, not just what we got. The first sweep kept only
    # the reached pose, and recovering the commanded one afterwards meant
    # re-deriving the plan and matching poses by distance -- which mismatched
    # two of ten. The steady-state positioning error is the cleanest single
    # number for how much an unregistered load costs, and it is free here.
    q_err = q_meas - q
    # Whether the samples were healthy, not just how many there were. `n_still`
    # counts slow samples; it says nothing about control latency or dropped
    # command packets, so a disturbed measurement could look fine. Record both
    # -- the baseline is 0.18 ms latency and 1.7% drops at rest (progress 8.5).
    lat = [float(x.controller_latency_ms) for x in still]
    okc = [bool(x.command_successful) for x in still]
    return {
        "pose": i + 1, "n_still": len(still),
        "q": q_meas.tolist(), "q_target": np.asarray(q).tolist(),
        "q_error_rad": q_err.tolist(),
        "q_error_max_rad": float(np.abs(q_err).max()),
        "tau_external": tau.tolist(),
        "latency_ms_mean": float(np.mean(lat)) if lat else None,
        "latency_ms_max": float(np.max(lat)) if lat else None,
        "frac_failed_commands": float(1.0 - np.mean(okc)) if okc else None,
    }


def _run_sweep(args, backend, approach, legs):
    """Drive the plan and sample each pose. This is the part that moves."""
    import time

    for q in approach:
        t_go = backend.expected_move_time_s(q) * args.slow
        log.info("homing over %.1f s", t_go)
        backend.move_to_joint_positions(q, time_to_go=t_go, blocking=True)

    rows = []
    for i, q in enumerate(legs):
        t_go = backend.expected_move_time_s(q) * args.slow
        log.info("pose %d/%d, moving over %.1f s", i + 1, len(legs), t_go)
        backend.move_to_joint_positions(q, time_to_go=t_go, blocking=True)
        time.sleep(args.settle)

        samples, still = _dwell_at_pose(args, backend)
        if len(still) < args.min_samples:
            print("  pose %d: only %d/%d still sample(s); skipped"
                  % (i + 1, len(still), len(samples)))
            continue
        row = _measure_pose(i, q, still)
        rows.append(row)
        print("  pose %d: %d still sample(s), |tau_ext| max %.3f Nm, "
              "settled %.4f rad from the target, latency %.2f ms, drops %.1f%%"
              % (i + 1, row["n_still"],
                 np.abs(np.asarray(row["tau_external"])).max(),
                 row["q_error_max_rad"],
                 row["latency_ms_mean"] if row["latency_ms_mean"] is not None else float("nan"),
                 100.0 * row["frac_failed_commands"]
                 if row["frac_failed_commands"] is not None else float("nan")))
    return rows


def _save_sweep(cfg, spec, fit, rows) -> None:
    import time

    out = Path(cfg.data_root).parent / "payload"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / ("sweep_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    dest.write_text(json.dumps({"fit": fit, "measurements": rows,
                                "robot": spec.to_metadata()}, indent=2))
    print("saved %s" % dest)
    print()
    print("Add this to the CONTROL BOX (Desk -> End Effector); it is not a config")
    print("value this repo can apply. Combine with what is already registered.")


def cmd_payload_sweep(args) -> int:
    """Drive the arm to computed poses and fit the load nobody registered.

    Autonomous motion, so everything is checked before anything moves: every
    pose and every straight line between consecutive poses is verified against
    the FR3 joint limits and a Cartesian floor, the whole plan is printed, and
    it waits for the operator unless --yes. The steps are separate functions so
    that order is visible here rather than buried in a hundred lines: choose,
    plan, confirm, only then move.
    """
    from ..kinematics import fk_from_urdf_text
    from ..payload import envelope_report, estimate_payload

    cfg = common._cfg(args)
    with common._connected(cfg) as (backend, guard):
        spec = backend.spec()
        if not spec.urdf_text:
            raise SystemExit("the server sent no URDF; cannot compute poses")
        fk = fk_from_urdf_text(spec.urdf_text, spec.ee_link_name)
        rng = np.random.default_rng(args.seed)

        pool = _sweep_candidates(args, cfg, spec, fk, rng)
        approach, legs, skipped = _sweep_plan(args, backend, spec, fk, pool)
        if not _confirm_sweep(args, fk, approach, legs, skipped):
            print("aborted; nothing moved")
            return 1
        rows = _run_sweep(args, backend, approach, legs)

    if len(rows) < 2:
        raise SystemExit("only %d usable pose(s); nothing to fit" % len(rows))
    qs = [np.asarray(r["q"]) for r in rows]
    taus = [np.asarray(r["tau_external"]) for r in rows]
    fit = estimate_payload(qs, taus, fk)

    print()
    _print_payload_fit(fit, envelope_report(qs, fk))
    _save_sweep(cfg, spec, fit, rows)
    return 0
