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


export function renderBacktestLimitations(limitations) {
  const list = document.getElementById("backtest-limitations-list");
  if (!list) return;
  const items = limitations || [];
  list.innerHTML = items.length
    ? items.map((note) => `<li>${escapeHtml(note)}</li>`).join("")
    : '<li class="text-muted">No limitations reported.</li>';
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
