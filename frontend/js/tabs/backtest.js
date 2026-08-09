/** Backtest tab: run picker, simulation results, equity curve. */

import { api } from "../api.js";
import { animateValue, escapeHtml } from "../format.js";
import { state } from "../state.js";

let chartInstance = null;

export function populateBacktestRuns(models) {
  const select = document.getElementById("backtest-run");
  if (!select) return;

  const current = select.value;
  select.innerHTML = "";
  if (!models || models.length === 0) {
    select.innerHTML = '<option value="">No model runs found</option>';
    onBacktestRunChange();
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
  onBacktestRunChange();
}


export function getSelectedBacktestRun() {
  const select = document.getElementById("backtest-run");
  if (!select || !select.value) return null;
  try {
    return JSON.parse(select.value);
  } catch (e) {
    return null;
  }
}


export function onBacktestRunChange() {
  const run = getSelectedBacktestRun();
  const label = document.getElementById("backtest-run-label");
  if (label) {
    label.innerText = run
      ? `${run.model_type}_${run.interval}_${run.timestamp}`
      : "No run selected";
  }
  if (!run) return;

  // Default the date range to the symbol's available history.
  const item = state.globalInventory.find((i) => i.symbol === run.symbol);
  const endInput = document.getElementById("eval-end");
  if (item && item.end_date !== "N/A" && endInput && !endInput.dataset.userSet) {
    endInput.value = item.end_date;
  }
}


// Beating buy & hold is the weakest of four claims. Showing only that one is
// how a reduced-beta position gets read as a forecast: ETHUSDT tft cleared
// holding by 11.06 pp on one window while its cumulative monthly excess was
// negative in all 40 months.
export function renderBacktestNulls(data) {
  const tbody = document.getElementById("backtest-nulls-tbody");
  if (!tbody) return;

  const nulls = data.nulls || {};
  const signed = (value, unit) =>
    Number.isFinite(value)
      ? `<span class="${value >= 0 ? "text-green" : "text-red"}">${value >= 0 ? "+" : ""}${value.toFixed(2)}${unit}</span>`
      : '<span class="text-muted">--</span>';

  const rows = [];

  const constant = nulls.constant_exposure;
  if (constant) {
    rows.push([
      `Constant ${(100 * (constant.mean_exposure ?? 0)).toFixed(0)}% exposure`,
      signed(constant.excess_roi_pct, " pp"),
      "Rules out simply carrying less of the asset.",
    ]);
  }

  const timing = nulls.mistimed_replay;
  if (timing && Number.isFinite(timing.percentile)) {
    // 50 is chance. 95 is the usual bar, and with many runs examined even that
    // is generous, so anything below is shown as unconvincing rather than good.
    const strong = timing.percentile >= 95;
    rows.push([
      "Vs its own mistimed copies",
      `<span class="${strong ? "text-green" : "text-muted"}">${timing.percentile.toFixed(1)} pctile</span>`,
      `Rules out volatility timing. ${timing.samples} shifts; 50 = chance.`,
    ]);
  }

  const momentum = nulls.naive_momentum;
  if (momentum && Number.isFinite(momentum.roi_pct)) {
    const delta = Number.isFinite(timing?.replay_roi_pct)
      ? timing.replay_roi_pct - momentum.roi_pct
      : null;
    rows.push([
      `${momentum.lookback_bars}-bar momentum rule`,
      delta === null ? '<span class="text-muted">--</span>' : signed(delta, " pp"),
      "Rules out “any trend rule would have done this”. No model in it.",
    ]);
  }

  tbody.innerHTML = rows.length
    ? rows
        .map(
          ([label, result, meaning]) =>
            `<tr><td>${escapeHtml(label)}</td><td>${result}</td><td class="text-muted">${meaning}</td></tr>`,
        )
        .join("")
    : '<tr><td class="text-muted">Run a backtest to see its comparisons.</td></tr>';
}


export function renderBacktestLimitations(limitations) {
  const list = document.getElementById("backtest-limitations-list");
  if (!list) return;
  const items = limitations || [];
  list.innerHTML = items.length
    ? items.map((note) => `<li>${escapeHtml(note)}</li>`).join("")
    : '<li class="text-muted">No limitations reported.</li>';
}



// The assumptions form. Values come from /api/meta/backtest-defaults rather than
// from literals in the HTML, for the reason the epoch field taught: a number
// typed into markup becomes a second definition and the two drift silently.
let backtestDefaults = null;

const PERCENT_FIELDS = [
  ["bt-fee", "fee_rate"],
  ["bt-slippage", "slippage_rate"],
  ["bt-hurdle", "round_trip_cost"],
  ["bt-rebalance", "min_rebalance_fraction"],
];

export async function loadBacktestDefaults() {
  try {
    backtestDefaults = await api.backtestDefaults();
  } catch (error) {
    console.error("backtest defaults unavailable", error);
    return;
  }
  resetBacktestAssumptions();
}

export function resetBacktestAssumptions() {
  if (!backtestDefaults) return;

  const capital = document.getElementById("backtest-capital");
  if (capital) capital.value = backtestDefaults.initial_capital;

  // Stored as fractions, shown as percentages: 0.001 is far harder to read than
  // 0.1%, and a fee field is exactly where a factor-of-100 slip hides.
  for (const [id, key] of PERCENT_FIELDS) {
    const field = document.getElementById(id);
    if (field) field.value = round6(backtestDefaults[key] * 100);
  }
  const momentum = document.getElementById("bt-momentum");
  if (momentum) momentum.value = backtestDefaults.momentum_lookback_bars;

  const container = document.getElementById("backtest-exposure");
  if (container) {
    container.innerHTML = (backtestDefaults.exposure || [])
      .map(
        (grade) => `
        <label class="evaluation-label">
          <span>${escapeHtml(grade.label)} (%)</span>
          <input type="number" id="bt-exp-${escapeHtml(grade.key)}"
                 class="glass-input" step="5" min="0" max="100"
                 value="${round6(grade.default * 100)}" />
        </label>`,
      )
      .join("");
  }
}

function round6(value) {
  return Math.round(Number(value) * 1e6) / 1e6;
}

function fraction(id) {
  const raw = document.getElementById(id)?.value;
  if (raw === undefined || raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value / 100 : null;
}

function exposureParam() {
  if (!backtestDefaults) return null;
  const parts = [];
  for (const grade of backtestDefaults.exposure || []) {
    const share = fraction(`bt-exp-${grade.key}`);
    // Only send what the user actually moved; the server merges onto its own
    // default, so an unchanged grade needs no opinion from the browser.
    if (share !== null && Math.abs(share - grade.default) > 1e-9) {
      parts.push(`${grade.key}=${share}`);
    }
  }
  return parts.length ? parts.join(",") : null;
}

export async function runBacktest() {
  const run = getSelectedBacktestRun();
  const status = document.getElementById("backtest-status");
  if (!run) return alert("Select a prediction run first.");

  const start = document.getElementById("eval-start").value;
  const end = document.getElementById("eval-end").value;
  const horizon = document.getElementById("backtest-horizon")?.value || "short";
  const capital = document.getElementById("backtest-capital")?.value || "10000";

  const params = new URLSearchParams({
    start,
    end,
    model_type: run.model_type,
    timestamp: run.timestamp,
    interval: run.interval,
    horizon,
    capital,
  });

  // Omit anything left at its default, so the request stays the shape a plain
  // CLI invocation would produce and the server's own defaults stay in charge.
  const optional = {
    fee_rate: fraction("bt-fee"),
    slippage_rate: fraction("bt-slippage"),
    round_trip_cost: fraction("bt-hurdle"),
    min_rebalance: fraction("bt-rebalance"),
    momentum_lookback: document.getElementById("bt-momentum")?.value || null,
    exposure: exposureParam(),
  };
  for (const [key, value] of Object.entries(optional)) {
    if (value !== null && value !== "") params.set(key, String(value));
  }

  if (status) status.innerText = `Simulating ${run.model_type} ${run.timestamp}...`;

  try {
    const data = await api.backtest(run.symbol, params);

    animateValue("val-roi", 0, data.roi, 1000, "", "%");
    animateValue("val-capital", data.initial_capital, data.final_capital, 1000, "$", "");
    animateValue("val-drawdown", 0, data.max_drawdown, 1000, "", "%");

    document.getElementById("val-trades").innerText = data.total_trades;

    const roiEl = document.getElementById("val-roi");
    roiEl.className =
      "metric-value fade-in " + (data.roi >= 0 ? "text-green" : "text-red");

    // The benchmark is the point of the whole tab: a positive ROI during a
    // bull market says nothing on its own.
    const excess = data.benchmark?.excess_roi_pct;
    const excessEl = document.getElementById("val-excess");
    excessEl.innerText = Number.isFinite(excess)
      ? `${excess >= 0 ? "+" : ""}${excess.toFixed(2)}%`
      : "--%";
    excessEl.className =
      "metric-value fade-in " + (excess >= 0 ? "text-green" : "text-red");

    document.getElementById("val-hold").innerText = Number.isFinite(
      data.benchmark?.roi_pct,
    )
      ? `${data.benchmark.roi_pct.toFixed(2)}%`
      : "--%";
    document.getElementById("val-sharpe").innerText = Number.isFinite(
      data.sharpe,
    )
      ? data.sharpe.toFixed(2)
      : "--";
    document.getElementById("val-exposure").innerText = Number.isFinite(
      data.time_in_market_pct,
    )
      ? `${data.time_in_market_pct.toFixed(1)}%`
      : "--%";

    renderBacktestNulls(data);
    renderBacktestLimitations(data.assumptions);
    drawChart(data.chart_data);

    if (status) {
      const skipped = data.skipped_untrained
        ? `, ${data.skipped_untrained} untrained rows skipped`
        : "";
      status.innerText =
        `${data.run.slug} · ${data.horizon} horizon · ` +
        `${data.days_tested} bars simulated${skipped}.`;
    }
  } catch (error) {
    console.error(error);
    if (status) status.innerText = `Backtest failed: ${error.message}`;
  }
}


export function drawChart(chartData) {
  const ctx = document.getElementById("equityChart").getContext("2d");
  if (chartInstance) chartInstance.destroy();

  Chart.defaults.color = "#8e9bb0";
  Chart.defaults.font.family = "Inter";

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: chartData.dates,
      datasets: [
        {
          label: "Portfolio Equity ($)",
          data: chartData.capital,
          borderColor: "#00e5ff",
          backgroundColor: "rgba(0, 229, 255, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.2,
          yAxisID: "y",
        },
        {
          label: "Asset Price ($)",
          data: chartData.close,
          borderColor: "#2962ff",
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [5, 5],
          fill: false,
          tension: 0.2,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", labels: { color: "#f0f4f8" } },
        tooltip: {
          backgroundColor: "rgba(20, 24, 39, 0.9)",
          titleColor: "#00e5ff",
          bodyColor: "#f0f4f8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 12,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.03)" },
          ticks: { maxTicksLimit: 10 },
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

// Small popover for prediction details
