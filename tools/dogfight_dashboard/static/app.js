import { initTrainingDashboard } from "./training.js";
import { initReplayViewer } from "./replay.js";

const AppState = {
  trainingStarted: false,
  replayStarted: false,
  config: null,
};

const $ = id => document.getElementById(id);

start().catch(error => {
  console.error(error);
  $("app-status").textContent = `Startup failed: ${error.message}`;
});

async function start() {
  bindTabs();
  AppState.config = await fetchJson("/api/app/config");
  const params = new URLSearchParams(window.location.search);
  const initialTab = params.get("tab") || AppState.config.defaultTab || "training";
  await showTab(initialTab === "replay" ? "replay" : "training");
  $("app-status").textContent = formatPaths(AppState.config);
}

function bindTabs() {
  for (const button of document.querySelectorAll(".tab-button")) {
    button.addEventListener("click", () => {
      showTab(button.dataset.tab).catch(error => {
        console.error(error);
        $("app-status").textContent = `Tab failed: ${error.message}`;
      });
    });
  }
}

async function showTab(tab) {
  for (const button of document.querySelectorAll(".tab-button")) {
    button.classList.toggle("active", button.dataset.tab === tab);
  }
  $("training-panel").classList.toggle("active", tab === "training");
  $("replay-panel").classList.toggle("active", tab === "replay");
  window.history.replaceState(null, "", `?tab=${tab}`);

  if (tab === "training" && !AppState.trainingStarted) {
    initTrainingDashboard({ apiBase: "/api/training" });
    AppState.trainingStarted = true;
  }
  if (tab === "replay" && !AppState.replayStarted) {
    await initReplayViewer({ apiBase: "/api/replay" });
    AppState.replayStarted = true;
  }
  window.dispatchEvent(new Event("resize"));
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function formatPaths(config) {
  return `Training: ${config.trainingLogdir} | Replay: ${config.replayLogdir}`;
}
