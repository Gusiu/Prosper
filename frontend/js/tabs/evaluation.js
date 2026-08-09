/** Evaluation tab: model ranking, calibration, worst predictions. */

import { api } from "../api.js";
import { loadAnalysisData, openAnalysisDrawer } from "../analysis.js";
import { escapeHtml, formatMetric, formatPercentMetric } from "../format.js";
import { getResearchFlags } from "../research.js";
import { submitTask } from "../queue.js";
import { state } from "../state.js";

const evaluationState = {
  evaluations: [],
  selected: null,
  summary: null,
  worstRows: [],
  recommendations: [],
  flags: [],
  stability: null,
};

export function populateEvaluationModelSelect(models) {
  const select = document.getElementById("prediction-eval-run");
  if (!select) return;

  const current = select.value;
  select.innerHTML = "";
  if (!models || models.length === 0) {
    select.innerHTML = '<option value="">No model runs found</option>';
    return;
  }

  models
    .slice()
    .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)))
    .forEach((model) => {
      const option = document.createElement("option");
      option.value = JSON.stringify({
        symbol: model.symbol,
        model_type: model.model_type,
        timestamp: model.timestamp,
        interval: model.interval || "1d",
      });
      option.textContent = `${model.symbol} / ${model.model_type} / ${model.interval || "1d"} / ${model.timestamp}`;
      select.appendChild(option);
    });

  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}


export function getSelectedEvaluationRun() {
  const select = document.getElementById("prediction-eval-run");
  if (!select || !select.value) return null;
  try {
    return JSON.parse(select.value);
  } catch (e) {
    return null;
  }
}


export async function runPredictionEvaluation() {
  const run = getSelectedEvaluationRun();
  if (!run) return alert("Select a prediction run first.");

  const status = document.getElementById("eval-run-status");
  if (status) status.innerText = `Starting evaluation for ${run.symbol} ${run.model_type}...`;

  try {
    const snapshot = await submitTask(
      `Evaluate ${run.model_type} [${run.symbol}]`,
      "/api/eval/run",
      { ...run, flags: getResearchFlags() },
    );
    if (status && snapshot) {
      status.innerText = `Evaluation queued (${snapshot.queued} in queue).`;
    }
  } catch (error) {
    console.error(error);
    alert(`Evaluation failed to start: ${error.message}`);
  }
}


export async function loadEvaluations() {
  try {
    const data = await api.evaluations();
    evaluationState.evaluations = data.evaluations || [];
    renderEvaluationRanking();
    if (!evaluationState.selected && evaluationState.evaluations.length > 0) {
      openEvaluationDetails(evaluationState.evaluations[0]);
    }
  } catch (e) {
    console.error("Failed to load evaluations", e);
  }
}


export function renderEvaluationRanking() {
  const tbody = document.getElementById("eval-ranking-tbody");
  const count = document.getElementById("eval-ranking-count");
  if (!tbody) return;
  if (count) count.innerText = `(${evaluationState.evaluations.length})`;
  tbody.innerHTML = "";

  if (evaluationState.evaluations.length === 0) {
    tbody.innerHTML = `<tr><td class="text-muted">No evaluations yet</td></tr>`;
    return;
  }

  evaluationState.evaluations.forEach((item, index) => {
    const tr = document.createElement("tr");
    const untrained = Number(item.untrained_rows || 0);
    const untrainedCell = untrained
      ? `<span class="text-red" title="Horizons the model could not train; excluded from every metric">${untrained}</span>`
      : `<span class="text-muted">0</span>`;
    tr.innerHTML = `
      <td>#${index + 1}</td>
      <td>${escapeHtml(item.symbol)}</td>
      <td>${escapeHtml(item.model_type)} <span class="text-muted">(${escapeHtml(item.interval || "1d")})</span></td>
      <td><span class="text-muted mono-cell">${escapeHtml(item.timestamp)}</span></td>
      <td><strong>${formatMetric(item.model_score, 1)}</strong></td>
      <td>${formatPercentMetric(item.accuracy, 1)}</td>
      <td>${formatMetric(item.ece, 3)}</td>
      <td>${untrainedCell}</td>
      <td></td>
    `;
    const button = document.createElement("button");
    button.className = "action-btn";
    button.textContent = "Details";
    button.addEventListener("click", () => openEvaluationDetails(item));
    tr.querySelector("td:last-child").appendChild(button);
    tbody.appendChild(tr);
  });
}


export async function openEvaluationDetails(summary) {
  if (!summary || !summary.symbol || !summary.model_type || !summary.timestamp)
    return;

  const status = document.getElementById("eval-run-status");
  if (status) status.innerText = `Loading evaluation ${summary.model_type} ${summary.timestamp}...`;

  try {
    const interval = summary.interval || "1d";
    const flag = document.getElementById("eval-flag-filter")?.value || "";
    const [detail, flagsPayload] = await Promise.all([
      api.evaluationDetail({ ...summary, interval, limit: 100, flag }),
      api.evaluationFlags({ ...summary, interval }).catch(() => ({ flags: [] })),
    ]);

    evaluationState.selected = detail;
    evaluationState.summary = summary;
    evaluationState.worstRows = detail.worst_predictions || [];
    evaluationState.recommendations = detail.recommendations || [];
    evaluationState.flags = flagsPayload.flags || [];
    renderEvaluationMetrics(detail.metrics || {});
    renderCalibration(detail.calibration || {});
    renderSharpness(detail.metrics || {});
    populateEvaluationFlagFilter();
    renderWorstPredictions();
    renderRecommendations();
    if (status) {
      const score = detail.metrics?.overall?.model_score;
      status.innerText = `Loaded ${summary.symbol} ${summary.model_type}; score ${formatMetric(score, 1)}.`;
    }
  } catch (e) {
    console.error(e);
    if (status) status.innerText = `Failed to load evaluation: ${e.message}`;
  }
}


export function renderEvaluationMetrics(metrics) {
  const overall = metrics.overall || {};
  const score = document.getElementById("eval-model-score");
  const accuracy = document.getElementById("eval-accuracy");
  const brier = document.getElementById("eval-brier");
  const ece = document.getElementById("eval-ece");
  if (score) score.innerText = formatMetric(overall.model_score, 1);
  if (accuracy) accuracy.innerText = formatPercentMetric(overall.accuracy, 1);
  if (brier) brier.innerText = formatMetric(overall.brier, 3);
  if (ece) ece.innerText = formatMetric(overall.ece, 3);
}


export function renderCalibration(calibration) {
  const container = document.getElementById("eval-calibration-bars");
  if (!container) return;
  const bins = calibration?.overall?.bins || [];
  if (bins.length === 0) {
    container.innerHTML = `<div class="analysis-empty-subchart">No calibration data.</div>`;
    return;
  }

  container.innerHTML = bins
    .map((bin) => {
      const conf = Math.max(0, Math.min(1, Number(bin.confidence || 0)));
      const acc = Math.max(0, Math.min(1, Number(bin.accuracy || 0)));
      return `
        <div class="eval-bar-row">
          <span>${Math.round((bin.lower || 0) * 100)}-${Math.round((bin.upper || 0) * 100)}%</span>
          <div class="eval-bar-track">
            <div class="eval-bar confidence" style="width:${conf * 100}%"></div>
            <div class="eval-bar accuracy" style="width:${acc * 100}%"></div>
          </div>
          <strong>${bin.count || 0}</strong>
        </div>
      `;
    })
    .join("");
}


export function renderSharpness(metrics) {
  const container = document.getElementById("eval-sharpness-bars");
  if (!container) return;
  const byHorizon = metrics.by_horizon || {};
  const entries = Object.entries(byHorizon);
  if (entries.length === 0) {
    container.innerHTML = `<div class="analysis-empty-subchart">No sharpness data.</div>`;
    return;
  }
  container.innerHTML = entries
    .map(([horizon, item]) => {
      // The reference is the entropy of a uniform forecast over the classes
      // this horizon actually carries. Hardcoding ln(3) survived the move to
      // two classes and made every bar read as sharper than it is: a coin flip
      // scored 1 - 0.693/1.099 = 0.37 of the sharpness it should score none of.
      const classes = Number(item.n_classes) || 2;
      const maxEntropy = Math.log(Math.max(2, classes));
      const entropy = Number(item.entropy || 0);
      const sharpness = Math.max(0, Math.min(1, 1 - entropy / maxEntropy));
      return `
        <div class="eval-bar-row">
          <span>${escapeHtml(horizon)}</span>
          <div class="eval-bar-track single">
            <div class="eval-bar sharpness" style="width:${sharpness * 100}%"></div>
          </div>
          <strong>${formatMetric(item.score, 1)}</strong>
        </div>
      `;
    })
    .join("");
}


// ── Quality over time ───────────────────────────────────────────────────────
// The one question a whole-run aggregate cannot answer: did the quality hold?
// A model that was excellent in one regime and useless since averages out to
// "mediocre", which reads the same as consistently mediocre.

const HORIZON_COLOURS = {
  short: "#4ade80",
  medium: "#60a5fa",
  long: "#f472b6",
};

let stabilityChart = null;

export async function loadStability() {
  const run = getSelectedEvaluationRun();
  const status = document.getElementById("stability-status");
  if (!run) {
    if (status) status.innerText = "Select a prediction run first.";
    return;
  }

  if (status) status.innerText = `Scoring ${run.symbol} ${run.model_type} month by month...`;
  try {
    const params = new URLSearchParams({
      model_type: run.model_type,
      timestamp: run.timestamp,
      interval: run.interval || "1d",
    });
    const data = await api.stability(run.symbol, params);
    evaluationState.stability = data;
    renderStability();
    if (status) {
      // Report the months actually covered, not the range that was asked for:
      // a run that starts later than the request would otherwise be described
      // by a window it has no data in.
      const first = data.labels[0];
      const last = data.labels[data.labels.length - 1];
      status.innerText = data.labels.length
        ? `${data.run.slug}: ${data.labels.length} months, ${first} to ${last}. ` +
          `Months with fewer than ${data.min_month_samples} scored bars are marked.`
        : `${data.run.slug}: no scored months in ${data.start}..${data.end}.`;
    }
  } catch (e) {
    console.error(e);
    if (status) status.innerText = `Failed to load stability: ${e.message}`;
  }
}

export function onStabilityMetricChange() {
  if (evaluationState.stability) renderStability();
}

export function renderStability() {
  const data = evaluationState.stability;
  const wrapper = document.getElementById("stability-chart");
  const empty = document.getElementById("stability-empty");
  const canvas = document.getElementById("stabilityChart");
  if (!data || !wrapper || !canvas) return;

  const metric = document.getElementById("stability-metric")?.value || "logloss";
  const datasets = Object.entries(data.series)
    .filter(([, series]) => (series[metric] || []).some((v) => v !== null))
    .map(([horizon, series]) => ({
      label: horizon,
      data: series[metric],
      borderColor: HORIZON_COLOURS[horizon] || "#8e9bb0",
      backgroundColor: HORIZON_COLOURS[horizon] || "#8e9bb0",
      // A month a horizon could not be scored in must stay a hole; joining
      // across it would draw a trend that was never measured.
      spanGaps: false,
      borderWidth: 2,
      tension: 0.2,
      pointRadius: series.samples.map((n, i) => (series.sparse[i] ? 4 : 2)),
      pointStyle: series.sparse.map((sparse) => (sparse ? "triangle" : "circle")),
    }));

  if (datasets.length === 0) {
    empty.innerText = "No scored months for this run.";
    empty.classList.remove("hidden");
    wrapper.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  wrapper.classList.remove("hidden");

  if (stabilityChart) stabilityChart.destroy();
  stabilityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels: data.labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            afterLabel: (item) => {
              const series = data.series[item.dataset.label];
              const n = series?.samples?.[item.dataIndex];
              const sparse = series?.sparse?.[item.dataIndex];
              return sparse ? `${n} bars — too few to read` : `${n} bars`;
            },
          },
        },
      },
      scales: {
        y: {
          title: { display: true, text: metric },
          beginAtZero: metric === "accuracy",
          ...(metric === "accuracy" ? { max: 1 } : {}),
        },
      },
    },
  });
}


export function populateEvaluationFlagFilter() {
  const select = document.getElementById("eval-flag-filter");
  if (!select) return;
  const current = select.value;
  select.innerHTML = '<option value="">All flags</option>';
  (evaluationState.flags || []).forEach(({ flag, count }) => {
    const option = document.createElement("option");
    option.value = flag;
    option.textContent = `${flag} (${count})`;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

// Filtering happens server-side: the full quality set is far too large to
// ship to the browser just so it can hide rows.


export function onEvaluationFlagChange() {
  if (evaluationState.summary) openEvaluationDetails(evaluationState.summary);
}


export function renderWorstPredictions() {
  const tbody = document.getElementById("eval-worst-tbody");
  if (!tbody) return;
  const rows = evaluationState.worstRows || [];

  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td class="text-muted">No worst prediction rows</td></tr>`;
    return;
  }

  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    const flags = row.flags ? String(row.flags).split("|").filter(Boolean) : [];
    tr.innerHTML = `
      <td>${escapeHtml(row.date)}</td>
      <td>${escapeHtml(row.horizon)}</td>
      <td>${escapeHtml(row.actual_direction || "-")}</td>
      <td>${escapeHtml(row.pred_direction || "-")} <span class="text-muted">${formatPercentMetric(row.pred_confidence, 0)}</span></td>
      <td>${formatMetric(row.quality_score, 1)}</td>
      <td><span class="eval-flag-list">${escapeHtml(flags.join(", ") || "none")}</span></td>
      <td><button class="action-btn" data-action="evaluation-inspect" data-index="${index}">Inspect</button></td>
    `;
    tbody.appendChild(tr);
  });

  const total = evaluationState.selected?.worst_predictions_total;
  if (Number.isFinite(total) && total > rows.length) {
    const note = document.createElement("tr");
    note.innerHTML = `<td colspan="7" class="text-muted">Showing the ${rows.length} worst of ${total} scored rows.</td>`;
    tbody.appendChild(note);
  }
}


export function renderRecommendations() {
  const tbody = document.getElementById("eval-recommendations-tbody");
  if (!tbody) return;
  const rows = evaluationState.recommendations.slice(0, 50);
  tbody.innerHTML = "";
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td class="text-muted">No recommendations loaded</td></tr>`;
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.date)}</td>
      <td>${escapeHtml(row.horizon)}</td>
      <td><strong>${escapeHtml(row.recommendation)}</strong></td>
      <td>${formatMetric(row.edge, 3)}</td>
      <td>${formatMetric(row.quality_score, 1)}</td>
    `;
    tbody.appendChild(tr);
  });
}


export function exportWorstPredictionsCsv() {
  const rows = evaluationState.worstRows || [];
  if (rows.length === 0) return alert("No worst predictions to export.");
  // Rows arrive already flattened from predictions_quality.parquet.
  const headers = [
    "date",
    "open_time",
    "horizon",
    "target_date",
    "actual_direction",
    "pred_direction",
    "pred_confidence",
    "quality_score",
    "error_score",
    "flags",
    "reason",
  ];
  const csvRows = [
    headers.join(","),
    ...rows.map((row) =>
      headers
        .map((header) => `"${String(row[header] ?? "").replace(/"/g, '""')}"`)
        .join(","),
    ),
  ];
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  const metrics = evaluationState.selected?.metrics || {};
  link.download = `${metrics.symbol || "evaluation"}-${metrics.model_type || "model"}-${metrics.timestamp || "run"}-worst.csv`;
  link.href = URL.createObjectURL(blob);
  link.click();
  URL.revokeObjectURL(link.href);
}


export function inspectEvaluationPrediction(index) {
  const row = evaluationState.worstRows[index];
  if (!row) return;
  const item = state.globalInventory.find((entry) => entry.symbol === row.symbol);
  if (!item) {
    return alert("Symbol is not present in the data inventory. Refresh data inventory first.");
  }
  openAnalysisDrawer(row.symbol);
  setTimeout(() => {
    const intervalSelect = document.getElementById("analysis-interval");
    if (intervalSelect && [...intervalSelect.options].some((option) => option.value === row.interval)) {
      intervalSelect.value = row.interval;
    }
    const startInput = document.getElementById("analysis-start");
    const endInput = document.getElementById("analysis-end");
    const baseDate = new Date(row.date);
    if (!Number.isNaN(baseDate.getTime())) {
      const start = new Date(baseDate);
      start.setDate(start.getDate() - 14);
      startInput.value = start.toISOString().slice(0, 10);
      endInput.value = row.target_date || row.date;
    }
    loadAnalysisData();
  }, 80);
}
