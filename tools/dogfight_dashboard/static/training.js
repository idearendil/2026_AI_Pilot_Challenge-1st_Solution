"use strict";

let API_BASE = "/api/training";

const CHART_GROUPS = [
  {
    id: "reward",
    title: "Episode Reward",
    metrics: ["episode/score", "episode/reward_mean"],
  },
  {
    id: "outcome",
    title: "Outcome Rates",
    metrics: [
      "episode/win_rate",
      "episode/loss_rate",
      "episode/timeout_rate",
      "episode/crash_rate",
    ],
  },
  {
    id: "length",
    title: "Episode Length and Count",
    metrics: ["episode/length", "episode/count"],
  },
  {
    id: "tactical",
    title: "Tactical Distance / WEZ",
    metrics: [
      "dogfight/wez_steps",
      "dogfight/distance_mean",
      "dogfight/distance_min",
    ],
  },
  {
    id: "tactical_angles",
    title: "Tactical Angles",
    metrics: [
      "dogfight/initial_alpha_deg",
      "dogfight/initial_ata_deg",
      "dogfight/initial_aa_deg",
      "dogfight/final_ata_deg",
      "dogfight/final_aa_deg",
    ],
  },
  {
    id: "safety",
    title: "Safety / Envelope",
    metrics: [
      "dogfight/altitude_penalty_steps",
      "dogfight/headon_guard_fail",
      "episode/crash_rate",
      "episode/timeout_rate",
    ],
  },
  {
    id: "reward_parts",
    title: "Reward Components",
    metrics: [
      "reward/pursuit",
      "reward/damage",
      "reward/safety",
      "reward/survival",
    ],
  },
  {
    id: "stability",
    title: "Learner Stability",
    metrics: [
      "train/loss/policy",
      "train/loss/value",
      "train/entropy",
      "train/kl",
    ],
  },
  {
    id: "action",
    title: "Action Health",
    metrics: [
      "action/saturation_rate",
      "action/roll_mean",
      "action/pitch_mean",
      "action/rudder_mean",
      "action/throttle_mean",
    ],
  },
  {
    id: "action_std",
    title: "Action Variability",
    metrics: [
      "action/roll_std",
      "action/pitch_std",
      "action/rudder_std",
      "action/throttle_std",
    ],
  },
  {
    id: "value",
    title: "Value Diagnostics",
    metrics: ["train/clip_frac", "train/explained_var"],
  },
  {
    id: "sac",
    title: "SAC / Replay Diagnostics",
    metrics: [
      "train/loss/actor",
      "train/loss/critic",
      "train/loss/alpha",
      "train/alpha",
      "replay/memory_mb",
    ],
  },
  {
    id: "curriculum",
    title: "Curriculum / Throughput",
    metrics: [
      "curriculum/stage",
      "perf/env_steps_per_sec",
      "perf/learner_steps_per_sec",
      "perf/iteration_time_s",
    ],
  },
];

const STATUS_CARDS = [
  { key: "episode/score", label: "Reward", fmt: formatNumber },
  { key: "episode/win_rate", label: "Win Rate", fmt: formatPercent },
  { key: "episode/crash_rate", label: "Crash Rate", fmt: formatPercent },
  { key: "dogfight/distance_min", label: "Min Range", fmt: value => `${formatCompact(value)}m` },
  { key: "dogfight/wez_steps", label: "WEZ Steps", fmt: formatNumber },
  { key: "train/entropy", label: "Entropy", fmt: formatNumber },
  { key: "action/saturation_rate", label: "Sat Rate", fmt: formatPercent },
  { key: "curriculum/stage", label: "Stage", fmt: value => formatNumber(value) },
];

const COLORS = [
  "#4db6ac",
  "#f3b34c",
  "#7aa6ff",
  "#f06f64",
  "#b084d8",
  "#77c66e",
  "#d9c86c",
  "#66c6e0",
  "#ff9f7a",
  "#9ee37d",
  "#e8e2a7",
  "#d5a6ff",
];

const EXPECTED_METRICS = [...new Set(CHART_GROUPS.flatMap(group => group.metrics))];

const State = {
  runs: [],
  selectedRun: "",
  compareMode: false,
  compareRuns: [],
  allData: {},
  lastStep: {},
  smooth: 8,
  pollTimer: null,
  lastUpdate: 0,
};

const $ = id => document.getElementById(id);

export function initTrainingDashboard(options = {}) {
  API_BASE = options.apiBase || API_BASE;
  buildChartCards();
  bindEvents();
  refreshRunList().then(() => {
    poll();
    State.pollTimer = setInterval(poll, 5000);
  });
  setInterval(updateStatus, 1000);
}

function buildChartCards() {
  const grid = $("chart-grid");
  grid.innerHTML = "";
  for (const group of CHART_GROUPS) {
    const card = document.createElement("section");
    card.className = "chart-card";
    card.innerHTML = `
      <div class="chart-title">${group.title}</div>
      <canvas id="chart-${group.id}"></canvas>
      <div id="legend-${group.id}" class="chart-legend"></div>
    `;
    card.addEventListener("click", () => openModal(group));
    grid.appendChild(card);
  }
}

function bindEvents() {
  $("run-select").addEventListener("change", event => {
    selectRun(event.target.value);
  });
  $("compare-button").addEventListener("click", () => {
    State.compareMode = !State.compareMode;
    $("compare-button").classList.toggle("active", State.compareMode);
    renderRunList();
    renderCharts();
  });
  $("smooth-slider").addEventListener("input", event => {
    State.smooth = Number(event.target.value);
    $("smooth-value").textContent = String(State.smooth);
    State.allData = {};
    State.lastStep = {};
    poll();
  });
  $("modal-close").addEventListener("click", closeModal);
  $("modal").addEventListener("click", event => {
    if (event.target === $("modal")) {
      closeModal();
    }
  });
  window.addEventListener("resize", renderCharts);
}

async function refreshRunList() {
  const data = await apiFetch("/api/runs");
  State.runs = data.runs || [];
  const select = $("run-select");
  const previous = select.value;
  select.innerHTML = "";
  for (const run of State.runs) {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = run.name;
    select.appendChild(option);
  }
  if (State.runs.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No runs";
    select.appendChild(option);
    $("status-text").textContent = "No dashboard runs found";
    return;
  }
  const next = State.runs.some(run => run.name === previous)
    ? previous
    : State.runs[0].name;
  select.value = next;
  if (!State.selectedRun) {
    State.selectedRun = next;
    loadConfig(next);
  }
  renderRunList();
}

function renderRunList() {
  const list = $("run-list");
  list.innerHTML = "";
  for (const run of State.runs) {
    const item = document.createElement("div");
    item.className = "run-item";
    if (run.name === State.selectedRun) {
      item.classList.add("selected");
    }

    if (State.compareMode) {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = State.compareRuns.includes(run.name);
      box.addEventListener("change", event => {
        event.stopPropagation();
        if (box.checked) {
          State.compareRuns.push(run.name);
        } else {
          State.compareRuns = State.compareRuns.filter(name => name !== run.name);
        }
        poll();
      });
      item.appendChild(box);
    } else {
      const spacer = document.createElement("span");
      item.appendChild(spacer);
    }

    const body = document.createElement("div");
    body.innerHTML = `
      <div class="run-name">${run.name}</div>
      <div class="run-step">step ${formatCompact(run.last_step)}</div>
    `;
    item.appendChild(body);
    item.addEventListener("click", () => selectRun(run.name));
    list.appendChild(item);
  }
}

function selectRun(name) {
  if (!name) {
    return;
  }
  State.selectedRun = name;
  $("run-select").value = name;
  loadConfig(name);
  renderRunList();
  poll();
}

async function loadConfig(run) {
  try {
    const config = await apiFetch(`/api/config?run=${encodeURIComponent(run)}`);
    renderConfig(config);
  } catch (error) {
    $("config-box").innerHTML = "";
  }
}

function renderConfig(config) {
  const box = $("config-box");
  box.innerHTML = "";
  const entries = flattenConfig(config);
  if (entries.length === 0) {
    box.textContent = "No config";
    return;
  }
  const table = document.createElement("table");
  table.className = "config-table";
  const tbody = document.createElement("tbody");
  for (const [key, value] of entries) {
    const row = document.createElement("tr");
    row.className = "config-row";
    const keyEl = document.createElement("th");
    keyEl.className = "config-key";
    keyEl.scope = "row";
    keyEl.textContent = key;
    keyEl.title = key;
    const valueEl = document.createElement("td");
    valueEl.className = "config-value";
    valueEl.textContent = formatConfigValue(value);
    row.appendChild(keyEl);
    row.appendChild(valueEl);
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  box.appendChild(table);
}

function flattenConfig(value, prefix = "", out = []) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    if (prefix) {
      out.push([prefix, value]);
    }
    return out;
  }
  for (const [key, child] of Object.entries(value)) {
    const next = prefix ? `${prefix}.${key}` : key;
    flattenConfig(child, next, out);
  }
  return out;
}

function formatConfigValue(value) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

async function poll() {
  await refreshRunList();
  const runs = getActiveRuns();
  await Promise.all(runs.map(fetchMetrics));
  if (State.selectedRun) {
    await refreshLatest(State.selectedRun);
  }
  renderCharts();
  renderMetricInventory();
  State.lastUpdate = Date.now();
  updateStatus();
}

async function fetchMetrics(run) {
  const since = State.lastStep[run] || 0;
  const path = `/api/metrics?run=${encodeURIComponent(run)}`
    + `&since_step=${since}&smooth=${State.smooth}`;
  const data = await apiFetch(path);
  State.allData[run] ||= {};
  for (const [key, points] of Object.entries(data.metrics || {})) {
    State.allData[run][key] ||= [];
    State.allData[run][key].push(...points);
  }
  if (data.last_step > (State.lastStep[run] || 0)) {
    State.lastStep[run] = data.last_step;
  }
}

async function refreshLatest(run) {
  const latest = await apiFetch(`/api/latest?run=${encodeURIComponent(run)}`);
  const cards = $("cards");
  cards.innerHTML = "";
  for (const spec of STATUS_CARDS) {
    const value = latest.values?.[spec.key];
    const card = document.createElement("div");
    card.className = "metric-card";
    if (latest.alerts?.[spec.key] !== undefined) {
      card.classList.add("alert");
    }
    card.innerHTML = `
      <div class="card-label">${spec.label}</div>
      <div class="card-value">${value === undefined ? "--" : spec.fmt(value)}</div>
    `;
    cards.appendChild(card);
  }
}

function getActiveRuns() {
  if (State.compareMode && State.compareRuns.length > 0) {
    return [...new Set(State.compareRuns)];
  }
  return State.selectedRun ? [State.selectedRun] : [];
}

function renderCharts() {
  for (const group of CHART_GROUPS) {
    const canvas = $(`chart-${group.id}`);
    if (canvas) {
      const datasets = buildDatasets(group);
      drawChart(canvas, group, datasets);
      renderLegend($(`legend-${group.id}`), datasets);
    }
  }
  const modal = $("modal");
  if (!modal.hidden && modal.dataset.groupId) {
    const group = CHART_GROUPS.find(item => item.id === modal.dataset.groupId);
    if (group) {
      const datasets = buildDatasets(group);
      drawChart($("modal-canvas"), group, datasets);
      renderLegend($("modal-legend"), datasets);
    }
  }
}

function renderMetricInventory() {
  const box = $("metric-box");
  if (!box) {
    return;
  }
  const runData = State.allData[State.selectedRun] || {};
  const available = Object.keys(runData)
    .filter(key => (runData[key] || []).length > 0)
    .sort();
  const missing = EXPECTED_METRICS.filter(key => !available.includes(key));
  const extra = available.filter(key => !EXPECTED_METRICS.includes(key));
  box.innerHTML = "";
  for (const [title, values] of [
    ["Available", available],
    ["Expected Missing", missing],
    ["Other Logged", extra],
  ]) {
    const group = document.createElement("div");
    group.className = "metric-group";
    const titleEl = document.createElement("div");
    titleEl.className = "metric-group-title";
    titleEl.textContent = `${title} (${values.length})`;
    const listEl = document.createElement("div");
    listEl.className = "metric-list";
    listEl.textContent = values.length ? values.join(", ") : "None";
    group.appendChild(titleEl);
    group.appendChild(listEl);
    box.appendChild(group);
  }
}

function buildDatasets(group) {
  const datasets = [];
  let colorIndex = 0;
  for (const run of getActiveRuns()) {
    const runData = State.allData[run] || {};
    for (const metric of group.metrics) {
      const points = runData[metric] || [];
      if (points.length === 0) {
        continue;
      }
      datasets.push({
        name: `${getActiveRuns().length > 1 ? run + " / " : ""}${shortName(metric)}`,
        points,
        color: COLORS[colorIndex % COLORS.length],
      });
      colorIndex += 1;
    }
  }
  return datasets;
}

function drawChart(canvas, group, datasets) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#171b1f";
  ctx.fillRect(0, 0, width, height);
  if (datasets.length === 0) {
    ctx.fillStyle = "#9aa7b0";
    ctx.font = "13px Segoe UI, Arial";
    ctx.fillText("Waiting for metrics", 24, height / 2);
    return;
  }

  const bounds = getBounds(datasets);
  const padding = chartPadding(ctx, width, height, bounds);
  const plot = {
    x: padding.left,
    y: padding.top,
    w: Math.max(1, width - padding.left - padding.right),
    h: Math.max(1, height - padding.top - padding.bottom),
  };

  drawGrid(ctx, plot, bounds);
  for (const dataset of datasets) {
    drawSeries(ctx, plot, bounds, dataset);
  }
}

function chartPadding(ctx, width, height, bounds) {
  ctx.font = "11px Segoe UI, Arial";
  const labels = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = bounds.maxY - ((bounds.maxY - bounds.minY) * i) / 4;
    labels.push(formatCompact(value));
  }
  const widest = Math.max(...labels.map(label => ctx.measureText(label).width));
  return {
    left: Math.min(Math.max(widest + 18, 46), Math.max(54, width * 0.24)),
    right: Math.max(12, width * 0.03),
    top: Math.max(10, height * 0.04),
    bottom: Math.max(26, height * 0.12),
  };
}

function getBounds(datasets) {
  const xs = [];
  const ys = [];
  for (const dataset of datasets) {
    for (const [x, y] of dataset.points) {
      xs.push(x);
      ys.push(y);
    }
  }
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  if (minX === maxX) {
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const yPad = (maxY - minY) * 0.08;
  return { minX, maxX, minY: minY - yPad, maxY: maxY + yPad };
}

function drawGrid(ctx, plot, bounds) {
  ctx.strokeStyle = "#303941";
  ctx.lineWidth = 1;
  ctx.font = "11px Segoe UI, Arial";
  ctx.fillStyle = "#9aa7b0";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 4; i += 1) {
    const y = plot.y + (plot.h * i) / 4;
    const value = bounds.maxY - ((bounds.maxY - bounds.minY) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillText(formatCompact(value), plot.x - 8, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i += 1) {
    const x = plot.x + (plot.w * i) / 4;
    const value = bounds.minX + ((bounds.maxX - bounds.minX) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, plot.y);
    ctx.lineTo(x, plot.y + plot.h);
    ctx.stroke();
    ctx.fillText(formatCompact(value), x, plot.y + plot.h + 8);
  }
}

function drawSeries(ctx, plot, bounds, dataset) {
  ctx.strokeStyle = dataset.color;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  dataset.points.forEach(([xValue, yValue], index) => {
    const x = plot.x
      + ((xValue - bounds.minX) / (bounds.maxX - bounds.minX)) * plot.w;
    const y = plot.y + plot.h
      - ((yValue - bounds.minY) / (bounds.maxY - bounds.minY)) * plot.h;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function renderLegend(container, datasets) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  if (datasets.length === 0) {
    container.textContent = "No visible series";
    return;
  }
  for (const dataset of datasets) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.title = dataset.name;
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = dataset.color;
    const label = document.createElement("span");
    label.className = "legend-label";
    label.textContent = dataset.name;
    item.appendChild(swatch);
    item.appendChild(label);
    container.appendChild(item);
  }
}

function openModal(group) {
  const modal = $("modal");
  modal.hidden = false;
  modal.dataset.groupId = group.id;
  $("modal-title").textContent = group.title;
  const datasets = buildDatasets(group);
  drawChart($("modal-canvas"), group, datasets);
  renderLegend($("modal-legend"), datasets);
}

function closeModal() {
  const modal = $("modal");
  modal.hidden = true;
  modal.dataset.groupId = "";
}

function updateStatus() {
  if (!State.lastUpdate) {
    return;
  }
  const seconds = Math.round((Date.now() - State.lastUpdate) / 1000);
  const step = State.lastStep[State.selectedRun] || 0;
  $("status-text").textContent =
    `Updated ${seconds}s ago | step ${formatCompact(step)}`;
}

async function apiFetch(path) {
  const response = await fetch(path.replace(/^\/api/, API_BASE), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${path}`);
  }
  return response.json();
}

function shortName(key) {
  return key.split("/").slice(-2).join("/");
}

function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatCompact(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  const abs = Math.abs(value);
  if (abs >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`;
  }
  if (abs >= 1e3) {
    return `${(value / 1e3).toFixed(1)}k`;
  }
  if (abs < 1 && value !== 0) {
    return value.toFixed(3);
  }
  return value.toFixed(0);
}
