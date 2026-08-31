/* Franka kinesthetic teaching -- front end.
 *
 * No build step and no framework: the lab machine has no node/npm, and a tool
 * you cannot edit on the machine it runs on is a tool you stop maintaining.
 * Plotly is served from the installed python package, so this works offline.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const post = (name, args) =>
  fetch("/api/command", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, args: args || {} }),
  }).then((r) => r.json());
const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : r.json().then((e) => Promise.reject(e))));

const JOINT_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d"];
const DARK = matchMedia("(prefers-color-scheme: dark)").matches;
const LAYOUT = {
  margin: { l: 52, r: 12, t: 28, b: 34 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: DARK ? "#9aa4b2" : "#6b7480", size: 11 },
  xaxis: { gridcolor: DARK ? "#2b313a" : "#e8ebef", zeroline: false },
  yaxis: { gridcolor: DARK ? "#2b313a" : "#e8ebef", zeroline: false },
  showlegend: true,
  legend: { orientation: "h", y: 1.16, x: 0, font: { size: 10 } },
};
const CONF = { displayModeBar: false, responsive: true };
const layout = (title, extra) => Object.assign({}, LAYOUT, { title: { text: title, font: { size: 12 } } }, extra || {});

let state = {};
let selected = null;
let liveInit = false;

/* ---------------------------------------------------------------- live */

const LIVE_N = 900; // ~30 s at the 30 Hz telemetry rate
const live = { t: [], q: [], tau: [] };

function initLivePlots(dof) {
  const mk = (name) =>
    Array.from({ length: dof }, (_, j) => ({
      x: [], y: [], mode: "lines", name: `${name}${j + 1}`,
      line: { width: 1.5, color: JOINT_COLORS[j % JOINT_COLORS.length] },
    }));
  Plotly.newPlot("plot-q", mk("q"), layout("joint position (rad)"), CONF);
  Plotly.newPlot("plot-tau", mk("τ"), layout("external torque (Nm)"), CONF);
  liveInit = true;
}

function pushLive(tel, dof) {
  if (!liveInit) initLivePlots(dof);
  if (live.t.length === 0) live.t0 = tel.timestamp;
  live.t.push(tel.timestamp - live.t0);
  live.q.push(tel.q);
  live.tau.push(tel.tau_external);
  while (live.t.length > LIVE_N) { live.t.shift(); live.q.shift(); live.tau.shift(); }

  const idx = Array.from({ length: dof }, (_, j) => j);
  const cols = (rows, j) => rows.map((r) => r[j]);
  Plotly.react("plot-q", idx.map((j) => ({
    x: live.t, y: cols(live.q, j), mode: "lines", name: `q${j + 1}`,
    line: { width: 1.5, color: JOINT_COLORS[j % JOINT_COLORS.length] },
  })), layout("joint position (rad)"), CONF);
  Plotly.react("plot-tau", idx.map((j) => ({
    x: live.t, y: cols(live.tau, j), mode: "lines", name: `τ${j + 1}`,
    line: { width: 1.5, color: JOINT_COLORS[j % JOINT_COLORS.length] },
  })), layout("external torque (Nm)"), CONF);
}

/* --------------------------------------------------------------- state */

function render(s) {
  const prev = state.state;
  state = s;

  $("state-badge").textContent = s.state;
  $("state-badge").dataset.s = s.state;
  $("backend-badge").textContent = s.backend;
  $("robot-info").textContent = s.robot
    ? `${s.robot.model} · ${s.robot.num_dofs} DOF · ${s.robot.control_hz} Hz`
    : "—";
  $("message").textContent = s.message || "";
  $("error").textContent = s.error || "";
  $("error").hidden = !s.error;

  const teaching = s.state === "TEACHING";
  const busy = ["HOMING", "SAVING", "APPROACHING", "REPLAYING"].includes(s.state);
  // The arm can be moving under its own power in these; the stop button is the
  // only control that must never be disabled while it is.
  const moving = ["HOMING", "APPROACHING", "REPLAYING"].includes(s.state);
  $("btn-stop").disabled = !s.connected;
  $("stop-help").classList.toggle("live", moving);
  $("btn-teach").disabled = teaching || busy || !s.connected;
  $("btn-home").disabled = teaching || busy || !s.connected;
  $("btn-save").disabled = !teaching;
  $("btn-discard").disabled = !teaching;
  $("btn-replay").disabled = busy || teaching || !s.connected;

  const sess = s.session;
  $("session-progress").hidden = !sess;
  if (sess) {
    const total = sess.max_duration_s || sess.expected_s || 1;
    $("progress-fill").style.width = `${Math.min(100, (100 * sess.elapsed_s) / total)}%`;
    $("progress-text").textContent =
      `${sess.elapsed_s.toFixed(1)} s / ${total.toFixed(0)} s` +
      (teaching ? "  (ring buffer limit)" : "");
  }
  if (s.telemetry && s.robot) pushLive(s.telemetry, s.robot.num_dofs);

  // The episode list changes when a session ends or a replay is stored.
  if (prev && prev !== s.state && ["SAVING", "REPLAYING", "TEACHING"].includes(prev)) {
    loadEpisodes();
    if (selected) setTimeout(() => showEpisode(selected), 300);
  }
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = () => {
    $("state-badge").textContent = "NO BACKEND";
    $("state-badge").dataset.s = "DISCONNECTED";
    setTimeout(connectWS, 1500);
  };
}

/* ------------------------------------------------------------ episodes */

function loadEpisodes() {
  return getJSON("/api/episodes").then(({ episodes }) => {
    const tb = document.querySelector("#episode-table tbody");
    tb.innerHTML = "";
    if (!episodes.length) {
      tb.innerHTML = `<tr><td colspan="10" class="muted">no episodes yet</td></tr>`;
      return;
    }
    for (const e of episodes) {
      const tr = document.createElement("tr");
      if (e.name === selected) tr.className = "sel";
      const check =
        e.ok === false ? `<span class="tag bad">failed</span>`
        : e.n_warnings ? `<span class="tag warn">${e.n_warnings} warn</span>`
        : `<span class="tag ok">ok</span>`;
      tr.innerHTML = `
        <td><b>${e.name}</b></td>
        <td class="muted">${(e.created_at || "").replace("T", " ").slice(0, 19)}</td>
        <td>${(e.n_states || 0).toLocaleString()}</td>
        <td>${(e.duration_s || 0).toFixed(1)} s</td>
        <td>${(e.effective_hz || 0).toFixed(0)} Hz</td>
        <td><span class="tag">${e.backend}</span></td>
        <td>${e.processed ? "✓" : "—"}</td>
        <td>${e.n_replays || "—"}</td>
        <td>${check}</td>
        <td class="muted">${e.notes || ""}</td>`;
      tr.onclick = () => showEpisode(e.name);
      tb.appendChild(tr);
    }
  });
}

function statBlock(v) {
  const rows = [
    ["states", (v.n_states || 0).toLocaleString()],
    ["duration", `${(v.duration_s || 0).toFixed(2)} s`],
    ["effective rate", `${(v.effective_hz || 0).toFixed(1)} Hz`],
    ["dropped", v.estimated_dropped_samples],
    ["max gap", `${(v.max_gap_ms || 0).toFixed(2)} ms`],
    ["buffer fill", `${(100 * (v.buffer_fill || 0)).toFixed(1)} %`],
  ];
  return `<div class="grid">${rows
    .map(([k, val]) => `<div class="stat"><b>${val}</b><span>${k}</span></div>`)
    .join("")}</div>`;
}

function showEpisode(name) {
  selected = name;
  document.querySelectorAll("#episode-table tbody tr").forEach((tr) =>
    tr.classList.toggle("sel", tr.firstChild && tr.firstChild.textContent.trim() === name));
  $("detail").hidden = false;
  $("detail-name").textContent = name;

  getJSON(`/api/episodes/${name}`).then((d) => {
    $("meta-json").textContent = JSON.stringify(d.metadata, null, 2);
    const v = d.validation || {};
    const issues = []
      .concat((v.errors || []).map((t) => `<li class="bad">${t}</li>`))
      .concat((v.warnings || []).map((t) => `<li class="warn">${t}</li>`));
    $("validation").innerHTML =
      statBlock(v) + (issues.length ? `<ul>${issues.join("")}</ul>` : `<p class="muted">no issues found</p>`);

    const cut = d.metadata.processing && d.metadata.processing.cutoff_hz;
    if (cut) { $("cutoff").value = cut; $("cutoff-out").textContent = `${cut} Hz`; }

    drawFilterPlot(name, d.processed);
    drawSweepPlot(name, d.processed);
    drawComparePlot(name, d.replays || []);
  });
}

function drawFilterPlot(name, processed) {
  const raw = getJSON(`/api/episodes/${name}/series?source=raw&keys=q&max_points=3000`);
  const proc = processed
    ? getJSON(`/api/episodes/${name}/series?source=processed&keys=q_filtered&max_points=3000`).catch(() => null)
    : Promise.resolve(null);
  Promise.all([raw, proc]).then(([r, p]) => {
    const traces = [];
    const dof = r.arrays.q[0].length;
    for (let j = 0; j < dof; j++) {
      traces.push({
        x: r.arrays.t, y: r.arrays.q.map((row) => row[j]),
        mode: "lines", name: `q${j + 1} raw`, legendgroup: `j${j}`,
        line: { width: 1, color: JOINT_COLORS[j % JOINT_COLORS.length], dash: "dot" },
        opacity: 0.45,
      });
      if (p) traces.push({
        x: p.arrays.t, y: p.arrays.q_filtered.map((row) => row[j]),
        mode: "lines", name: `q${j + 1} filtered`, legendgroup: `j${j}`, showlegend: false,
        line: { width: 1.8, color: JOINT_COLORS[j % JOINT_COLORS.length] },
      });
    }
    Plotly.react("plot-filter", traces,
      layout(p ? "raw (dotted) vs filtered (solid)" : "raw joint positions — not processed yet",
             { xaxis: Object.assign({}, LAYOUT.xaxis, { title: "t (s)" }) }), CONF);
  });
}

function drawSweepPlot(name, processed) {
  if (!processed) { Plotly.purge("plot-sweep"); return; }
  getJSON(`/api/episodes/${name}/series?source=sweep&keys=t&max_points=3000`)
    .then((meta) => {
      const keys = meta.available.filter((k) => k.startsWith("cutoff_"));
      if (!keys.length) { Plotly.purge("plot-sweep"); return; }
      return getJSON(
        `/api/episodes/${name}/series?source=sweep&keys=${keys.join(",")}&max_points=3000`
      ).then((d) => {
        // One joint only: the point is to compare cutoffs, and seven joints
        // times four cutoffs is 28 lines nobody can read.
        const j = 1;
        const traces = keys.map((k, i) => ({
          x: d.arrays.t, y: d.arrays[k].map((row) => row[j]),
          mode: "lines", name: `${k.replace("cutoff_", "")} Hz`,
          line: { width: 1.6, color: JOINT_COLORS[i % JOINT_COLORS.length] },
        }));
        Plotly.react("plot-sweep", traces,
          layout(`cutoff sweep — joint ${j + 1}`,
                 { xaxis: Object.assign({}, LAYOUT.xaxis, { title: "t (s)" }) }), CONF);
      });
    })
    .catch(() => Plotly.purge("plot-sweep"));
}

function drawComparePlot(name, replays) {
  if (!replays.length) { Plotly.purge("plot-compare"); return; }
  const last = replays[replays.length - 1];
  Promise.all([
    getJSON(`/api/episodes/${name}/series?source=raw&keys=tau_external&max_points=3000`),
    getJSON(`/api/episodes/${name}/series?source=replay&replay=${last}&keys=tau_external&max_points=3000`),
  ]).then(([t, r]) => {
    const dof = t.arrays.tau_external[0].length;
    const traces = [];
    for (let j = 0; j < dof; j++) {
      traces.push({
        x: t.arrays.t, y: t.arrays.tau_external.map((row) => row[j]),
        mode: "lines", name: `τ${j + 1} taught`, legendgroup: `j${j}`,
        line: { width: 1.2, color: JOINT_COLORS[j % JOINT_COLORS.length], dash: "dot" },
        opacity: 0.5,
      });
      traces.push({
        x: r.arrays.t, y: r.arrays.tau_external.map((row) => row[j]),
        mode: "lines", name: `τ${j + 1} replay`, legendgroup: `j${j}`, showlegend: false,
        line: { width: 1.6, color: JOINT_COLORS[j % JOINT_COLORS.length] },
      });
    }
    Plotly.react("plot-compare", traces,
      layout(`external torque: taught (dotted) vs ${last} (solid) — the taught trace includes the operator's hand`,
             { xaxis: Object.assign({}, LAYOUT.xaxis, { title: "t (s)" }) }), CONF);
  }).catch(() => Plotly.purge("plot-compare"));
}

/* --------------------------------------------------------------- wiring */

$("btn-home").onclick = () => post("home");
$("btn-teach").onclick = () => { live.t = []; live.q = []; live.tau = []; post("start_teaching"); };
$("btn-save").onclick = () => post("stop_teaching", { save: true, notes: $("notes").value });
$("btn-discard").onclick = () => post("stop_teaching", { save: false });
$("btn-stop").onclick = () => post("stop");
$("cutoff").oninput = (e) => ($("cutoff-out").textContent = `${e.target.value} Hz`);
$("btn-process").onclick = () => {
  if (!selected) return;
  fetch(`/api/episodes/${selected}/process`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ cutoff_hz: Number($("cutoff").value) }),
  }).then(() => setTimeout(() => showEpisode(selected), 700));
};
$("btn-replay").onclick = () => {
  if (!selected) return;
  post("replay", { episode: selected, time_scale: Number($("time-scale").value) });
};

connectWS();
loadEpisodes();
