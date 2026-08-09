/** Bootstrap: wire the action registry, subscribe the tabs, load initial data. */

import { registerActions, startActionDispatch } from "./actions.js";
import {
  closeAnalysisDrawer,
  disposeAnalysisCharts,
  exportAnalysisPng,
  initAnalysisDrawer,
  loadAnalysisData,
  openAnalysisDrawer,
} from "./analysis.js";
import { api } from "./api.js";
import { EVENTS, on } from "./events.js";
import {
  addToQueue,
  moveQueue,
  pollTaskStatus,
  removeQueue,
  startQueueProcessing,
  updateQueueUI,
} from "./queue.js";
import {
  bindResearchFlagInputs,
  initResearchMode,
  toggleResearchMode,
} from "./research.js";
import { state } from "./state.js";
import {
  loadBacktestDefaults,
  onBacktestRunChange,
  populateBacktestRuns,
  resetBacktestAssumptions,
  runBacktest,
} from "./tabs/backtest.js";
import {
  deleteSymbol,
  fixQuality,
  initializeDates,
  loadInventory,
  loadSymbolMetadata,
  queueInlineAggregation,
  runInlineAggregation,
  showAggregatePanel,
} from "./tabs/data.js";
import {
  exportWorstPredictionsCsv,
  inspectEvaluationPrediction,
  loadEvaluations,
  loadStability,
  onEvaluationFlagChange,
  onStabilityMetricChange,
  populateEvaluationModelSelect,
  runPredictionEvaluation,
} from "./tabs/evaluation.js";
import {
  changePredictionMonth,
  deleteModelRun,
  downloadFeatureImportances,
  loadAIInventory,
  loadTrainingDefaults,
  openPredictionCalendar,
  runTrain,
  updateTrainDates,
  updateTrainModelUI,
} from "./tabs/models.js";

function switchTab(tabId, element) {
  document
    .querySelectorAll(".tab-content")
    .forEach((t) => t.classList.remove("active"));
  document.getElementById(tabId).classList.add("active");

  document
    .querySelectorAll(".tab-link")
    .forEach((l) => l.classList.remove("active"));
  element.classList.add("active");

  document.getElementById("topbar-title").innerText = element.innerText;
}

async function reportEngineStatus() {
  try {
    await api.status();
  } catch {
    document.getElementById("api-status").innerText = "Engine Offline";
    const dot = document.querySelector(".status-dot.online");
    if (dot) {
      dot.style.backgroundColor = "var(--red-main)";
      dot.style.boxShadow = "0 0 10px var(--red-main)";
    }
  }
}

registerActions({
  "switch-tab": ({ tab }, element) => switchTab(tab, element),
  "toggle-research": () => toggleResearchMode(),

  // Data Manager
  "queue-add": () => addToQueue(),
  "queue-start": () => startQueueProcessing(),
  "queue-move": ({ id, direction }) => moveQueue(Number(id), Number(direction)),
  "queue-remove": ({ id }) => removeQueue(Number(id)),
  "inventory-refresh": () => loadInventory(),
  "inventory-verify": () => loadInventory(true),
  "symbol-analyze": ({ symbol }) => openAnalysisDrawer(symbol),
  "symbol-delete": ({ symbol }) => deleteSymbol(symbol),
  "symbol-fix": ({ symbol }) => fixQuality(symbol),
  "aggregate-panel": ({ symbol }) => showAggregatePanel(symbol),
  "aggregate-queue": ({ symbol, start, end }) =>
    queueInlineAggregation(symbol, start, end),
  "aggregate-run": ({ symbol, start, end }) =>
    runInlineAggregation(symbol, start, end),

  // AI Models
  "models-refresh": () => loadAIInventory(),
  "train-run": () => runTrain(),
  "run-view": ({ symbol, modelType, timestamp, interval }) =>
    openPredictionCalendar(symbol, modelType, timestamp, interval),
  "run-delete": ({ symbol, modelType, timestamp, interval }) =>
    deleteModelRun(symbol, modelType, timestamp, interval),
  "calendar-month": ({ delta }) => changePredictionMonth(Number(delta)),

  // Evaluation
  "evaluation-run": () => runPredictionEvaluation(),
  "evaluation-refresh": () => loadEvaluations(),
  "evaluation-export": () => exportWorstPredictionsCsv(),
  "evaluation-inspect": ({ index }) => inspectEvaluationPrediction(Number(index)),
  "stability-load": () => loadStability(),

  // Backtest
  "backtest-run": () => runBacktest(),
  "backtest-reset": () => resetBacktestAssumptions(),

  // Analysis drawer
  "analysis-load": () => loadAnalysisData(),
  "analysis-close": () => closeAnalysisDrawer(),
  "analysis-export": () => exportAnalysisPng(),
  "analysis-download-fi": () => downloadFeatureImportances(),
});

function bootstrap() {
  startActionDispatch();
  // The assumptions form must never carry literals; it fills from the API.
  loadBacktestDefaults();

  // Tabs own their own pickers; the modules that reload data just announce it.
  on(EVENTS.TASK_FINISHED, () => {
    loadInventory(true);
    loadAIInventory();
    loadEvaluations();
  });
  on(EVENTS.INVENTORY_LOADED, () => {
    updateTrainDates();
    updateTrainModelUI();
  });
  on(EVENTS.RUNS_LOADED, (runs) => {
    populateEvaluationModelSelect(runs);
    populateBacktestRuns(runs);
  });

  document
    .getElementById("eval-flag-filter")
    ?.addEventListener("change", onEvaluationFlagChange);
  document
    .getElementById("stability-metric")
    ?.addEventListener("change", onStabilityMetricChange);
  document
    .getElementById("backtest-run")
    ?.addEventListener("change", onBacktestRunChange);
  document
    .getElementById("train-symbol")
    ?.addEventListener("change", updateTrainDates);
  document
    .getElementById("model-select")
    ?.addEventListener("change", updateTrainModelUI);

  reportEngineStatus();
  pollTaskStatus();

  initializeDates();
  loadSymbolMetadata();
  loadTrainingDefaults();
  loadInventory();
  loadAIInventory();
  loadEvaluations();

  bindResearchFlagInputs();
  initResearchMode();
  initAnalysisDrawer();
  updateQueueUI();
}

// `<script type="module">` is deferred, so DOMContentLoaded has usually already
// fired by the time this runs — listening for it alone would never boot the app.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}

// Exported so the drawer teardown and app state stay reachable from the console
// while debugging; nothing imports them.
export { disposeAnalysisCharts, state };
