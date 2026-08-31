# Kinesthetic teaching for Franka arms

**Guide the arm by hand → record at the controller's full rate → filter offline
→ replay it server-side.** No external force/torque sensor, no teaching handle —
just the arm's own torque sensing and the
[polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis)
control stack.

The package, and the command you type, is `kinesteach`.

```
teach ──▶ episode ──▶ process ──▶ replay ──▶ replay pass
 (hand)    (raw npz)   (filter,     (server    (a full episode
                        FK, dq)      RT loop)   of its own)
```

Built for a Franka Research 3 with a Robotiq 2F gripper, but nothing about the
DOF count, control rate, or gain vector is hardcoded — they are read from the
controller server at connect time.

---

## What you get

- **Zero-stiffness hand guiding** that logs every field of the robot state at
  the server's own rate, fetched in one batch from the server's ring buffer
  rather than streamed over gRPC.
- **Immutable raw captures.** The untouched server log is written once and
  checksummed; everything else in an episode is derived and can be regenerated.
- **Offline processing** — uniform resampling, zero-phase filtering, numerical
  differentiation, forward kinematics — with a cutoff sweep so you can see what
  the filter threw away.
- **Server-side replay** through `JointTrajectoryExecutor`, with the whole
  trajectory checked against the robot's position and velocity limits *before*
  anything is sent.
- **Payload identification.** Least-squares estimation of a load the control box
  does not know about, from poses chosen by a D-optimal design, driven inside a
  safe region you walked by hand.
- **A validation report per capture** that flags the quiet failures: a log that
  silently lost its head to the ring buffer, a session that ran at 640 Hz
  instead of 1000, a joint that spent the demo against a limit with the safety
  controller pushing back.

---

## Requirements

| | |
|---|---|
| Python | 3.8 or newer |
| Always | `numpy`, `scipy`, `pyyaml` |
| Real robot, and offline FK | [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis) (pulls in `torch`, `torchcontrol`, `pinocchio`) |
| Web interface (optional) | `fastapi`, `uvicorn`, `websockets`, `plotly` |
| Tests | `pytest` |

polymetis is not on PyPI — install it from the fairo repository following its
own instructions. Without it you can still record from the mock backend, process
episodes, and run most of the test suite; only forward kinematics and the real
backend are unavailable.

## Installation

```bash
git clone https://github.com/<you>/franka-kinesthetic-teaching.git
cd franka-kinesthetic-teaching

python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pyyaml
pip install fastapi uvicorn websockets plotly    # optional web interface
pip install pytest                               # to run the tests
```

The package is deliberately not installed — run it from the repository root:

```bash
python -m kinesteach --help
```

From any other directory, put the repository on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/franka-kinesthetic-teaching python -m kinesteach --help
```

To talk to a real robot, install polymetis into the same environment and start
its controller server against your arm before running anything here.

> **Note — our lab's setup**
>
> We share one conda environment (`cy-droid-polymetis`) with other work that
> already has polymetis built from source, so this repository is **never**
> `pip install`ed: a develop install would leave an egg-link in a shared
> environment. `scripts/kinesteach` puts the repository on `PYTHONPATH` and
> picks the right interpreter, so the environment is left exactly as found:
>
> ```bash
> ./scripts/kinesteach --help
> ./scripts/kinesteach --config configs/real.yaml teach --duration 60
> ```
>
> Override the interpreter with `KINESTEACH_PYTHON=/path/to/python`.
>
> On that build, `import polymetis` prints
> `Failed to load 'libtorchscript_pinocchio.so' from CONDA_PREFIX`. This is
> expected — the library is loaded from the source build directory instead.
>
> `configs/real.yaml` holds our robot's addresses and joint limits. Copy and
> edit it for your own arm.

---

## Quick start — no robot needed

The mock backend runs a synthetic 7-DOF arm in-process at 1 kHz. The whole
pipeline works against it.

```bash
python -m kinesteach teach --backend mock --duration 10 --notes "bin pick"
python -m kinesteach process episode_0001 --cutoff 10
python -m kinesteach replay  episode_0001 --backend mock --time-scale 2.0
python -m kinesteach list
python -m kinesteach report  episode_0001
```

```
saved episode_0001: 10000 states, 10.00 s, 1000.0 Hz
episode_0001: 9999 samples at 1000 Hz, butterworth cutoff 10.0 Hz, FK -> flange
replayed episode_0001 -> pass_0001 (19996 states, 19.99 s, completed)
  tracking rms 0.00039 rad, max 0.01415 rad
```

---

## Using a real robot

### 1. Configure

Point a config file at your controller server and, if your server's URDF
disagrees with your hardware, override the joint limits:

```yaml
backend:
  kind: real
  ip_address: 10.0.0.10
  port: 50051
  # The server sends DOF, rate, URDF and default gains -- but not joint limits,
  # so they are read out of its URDF. If that URDF is not your robot (ours runs
  # a Panda URDF against an FR3), state the real ones here. Nothing is relaxed:
  # the replay gate takes these as-is, so a wrong entry is a real hazard.
  joint_pos_min: [...]
  joint_pos_max: [...]
```

Then pass it to every command: `python -m kinesteach --config configs/real.yaml ...`

### 2. Walk a safe region, and weigh what is on the flange

Before anything drives itself, record the region it is allowed to move through.
You push the arm around by hand; the configurations you pass become the vertices
of a convex region **in joint space**, so every straight-line move between two
poses inside it is also inside it. (A Cartesian box cannot promise that — on our
arm, legs between two in-box poses bulged up to 0.43 m outside it.)

```bash
python -m kinesteach --config configs/real.yaml workspace     # guided, step by step
```

With a region on file, the sweep can drive to computed poses and fit the load
nobody registered:

```bash
python -m kinesteach --config configs/real.yaml payload-sweep --poses 10
python -m kinesteach payload-fit data/payload/sweep_*.json    # re-fit, offline
```

It prints a mass, a centre of mass in the flange frame, per-joint sensor bias,
and how well the poses actually pinned each of those down. **The result has to
be entered into the control box** (Franka Desk → End Effector) — gravity
compensation happens there, and nothing in this repository can set it.

<details>
<summary>Why this matters — measured on our arm</summary>

A ZED 2i camera, its bracket and cabling added **0.2349 ± 0.0155 kg** that was
never registered. The arm sank whenever the operator let go, and no teaching
gain fixes that: teaching requires `Kq = 0`, and damping only sets the *speed*
of the descent. After registering the measured load, the residual `tau_external`
signal fell from 0.4035 to 0.1473 Nm rms and the sag was gone.

It also showed what registration does *not* fix. Replay deviation follows

```
max deviation = 30.6 / Kq_scale + 37.5 mm      (R² = 0.9974)
```

The `1/Kq` term is joint friction. During teaching a human hand absorbs it;
during replay the controller has no integral term, so a joint settles wherever
`Kq · error` balances the friction and stays there — more time or a different
path will not get it closer.

</details>

### 3. Teach, process, replay

```bash
CFG="--config configs/real.yaml"
python -m kinesteach $CFG teach --duration 60 --notes "insert peg"
python -m kinesteach $CFG process episode_0001 --cutoff 10
python -m kinesteach $CFG replay  episode_0001 --time-scale 2.0 --kq-scale 0.5
```

`teach` ends on `Ctrl-C` and **keeps** the recording — saying "I am done" is the
natural reason to interrupt a demonstration, and losing four minutes of it to
that would be its own kind of failure. A second `Ctrl-C` stops the policy
immediately.

---

## What an episode looks like

```
episode_0001/
├── metadata.json        robot, gains, rates, checksums, the config used
├── robot.urdf           the exact model this log was produced under
├── robot_raw.npz        untouched server log            ← written once
├── gripper_raw.npz      polled gripper states (its own clock)
├── validation.json      the capture report
├── processed.npz        filtered / resampled / differentiated / FK
├── cutoff_sweep.npz     one filtered copy per candidate cutoff
└── replay/
    └── pass_0001/       a replay pass — itself a complete episode directory
```

An episode is self-contained: the URDF and the `ee_link_name` it was recorded
under travel with it, because `RobotState` carries no end-effector pose and
Cartesian data has to be reconstructed afterwards.

---

## Safety

This tool moves a 7-DOF arm under its own power. Three things it does, and one
it does not:

**Every trajectory is checked before it is sent.** Position and velocity limits,
with a margin. The server's own safety controller would push back on a
violation, but by then the arm is already moving into it.

**Every autonomous move is path-checked, not just endpoint-checked.**
`move_to_joint_positions` plans a straight line in joint space; its Cartesian
shape is uncontrolled, so a move between two perfectly good poses can still
swing the flange through the table. Homing is treated as an ordinary waypoint
for exactly this reason — reaching for "go home" is what you do when the arm is
somewhere awkward, which is precisely when the path deserves a look.

**Every code path that starts a policy guarantees it can be stopped.** The
real-time loop lives in the server process, not this one; if this process dies,
the server keeps running whatever it was given. `atexit`, `SIGINT` and `SIGTERM`
all terminate the policy.

**A software stop is not an emergency stop.** `terminate_policy()` does not
leave the arm limp — the server falls back to a PD hold at the default
stiffness, so the arm freezes where it is. It stops commanded motion. It does
not remove power, engage the brakes, or release anything the arm is pressing on.
**Keep the physical E-stop in reach.**

### Invariants

These are enforced by tests, not by documentation
([`tests/test_invariants.py`](tests/test_invariants.py)):

| # | Rule | How it is enforced |
|---|---|---|
| 1 | The full-rate record never travels through the web layer | telemetry is display-only; the log is fetched in one batch when the session ends |
| 2 | Raw captures are never rewritten | `write_raw` refuses to overwrite, and `metadata.raw_sha256` is re-verified on read |
| 3 | DOF and control rate are never hardcoded | read from server metadata; a 13-DOF @ 240 Hz mock is taught and saved in the suite |
| 4 | Teaching never applies a Cartesian stiffness | `Config` rejects `adaptive: true`; `JointImpedanceControl` is built directly |
| 5 | Starting a policy guarantees stopping it | `policy_guard` + a process-level `atexit`/signal backstop |
| 6 | `tau_external` during teaching is not an environment force | it contains the operator's hand; the comparison report says so |
| 7 | protobuf types never leave `backend/` | a test walks the AST of every module |

---

## Repository layout

```
kinesteach/
├── config.py       dataclass config with a YAML overlay
├── dataset.py      on-disk layout; the guard that keeps raw data raw
├── record.py       a finished capture → an episode directory
├── validate.py     capture report + acceptance checks
├── teach.py        hand-guiding sessions, gripper polling thread
├── process.py      resample · filter · differentiate · replay trajectory
├── kinematics.py   offline FK / Jacobian (flange frame)
├── replay.py       JointTrajectoryExecutor + the gates in front of it
├── envelope.py     the hand-walked safe region + path checking
├── payload.py      least-squares load identification, D-optimal pose choice
├── safety.py       nothing starts a policy without a way to stop it
├── backend/        the only place that knows polymetis exists
│   ├── base.py       the RobotBackend protocol, EpisodeBuffer, RobotSpec
│   ├── polymetis.py  the real robot
│   └── mock.py       an in-process synthetic arm
├── cli/            command line
└── webui/          optional browser front end
```

## Tests

```bash
python -m pytest tests/ -q
```

88 tests, all against the mock backend — no robot and no controller server
needed. Without polymetis installed, 72 pass and 16 skip (the FK and real-backend
ones).

## Optional: web interface

A small FastAPI front end for driving a session from a browser: state machine
buttons, live telemetry, plots of a recorded episode, and a stop button that
stays responsive while the arm is moving.

```bash
python -m kinesteach webui --backend mock      # http://127.0.0.1:8000
```

It is a convenience, not the main interface — the CLI does everything it does.
Plotly is served from the installed Python package rather than a CDN, so it
works on an isolated robot network.

---

## Status and limitations

- **Free space only, so far.** Everything measured to date is free-space motion;
  contact-rich replay is not yet characterised.
- **Flange, not TCP.** Forward kinematics returns the flange pose. Turning it
  into a tool pose needs a gripper calibration this repository does not do.
- **Payload registration is manual.** The sweep measures the load; you enter it
  into the control box. libfranka's `setLoad` is not reachable from here.
- **Gripper states are on their own clock.** They are polled and stored
  separately; aligning them with the arm log is an offline problem.
