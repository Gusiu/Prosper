/** Data Manager tab: inventory, pipeline form, aggregation and repair. */

import { api } from "../api.js";
import { EVENTS, emit } from "../events.js";
import { escapeHtml } from "../format.js";
import { getResearchFlags } from "../research.js";
import { submitTask, updateProgress } from "../queue.js";
import { state } from "../state.js";

let symbolMetadata = { symbols: [], base_assets: [], quote_assets: [] };

export async function initializeDates() {
  try {
    const data = await api.dates();

    const startInput = document.getElementById("pipe-start");
    const endInput = document.getElementById("pipe-end");

    // Default to last 3 months if no data
    startInput.value = data.default_start_month
      ? `${data.default_start_month}-01`
      : "2023-01-01";
    endInput.value = data.yesterday
      ? data.yesterday
      : new Date().toISOString().split("T")[0];

    document.getElementById("train-start").value = startInput.value;
    document.getElementById("train-end").value = endInput.value;

    document.getElementById("eval-start").value = "2024-01-01";
    document.getElementById("eval-end").value = endInput.value;

    // Track manual edits so run selection stops overwriting the end date.
    ["eval-start", "eval-end"].forEach((id) => {
      document.getElementById(id)?.addEventListener("change", (event) => {
        event.target.dataset.userSet = "1";
      });
    });
  } catch (e) {
    console.error("Failed to fetch dates", e);
  }
}


export async function loadSymbolMetadata() {
  try {
    symbolMetadata = await api.symbolMetadata();
    populateCoinOptions();
  } catch (e) {
    console.warn("Symbol metadata unavailable", e);
  }
}


export function populateCoinOptions() {
  const datalist = document.getElementById("coin-options");
  if (!datalist) return;

  const fallbackQuotes = ["USDT", "USDC", "FDUSD", "BTC", "ETH", "BNB"];
  const options = [
    ...(symbolMetadata.base_assets || []),
    ...(symbolMetadata.quote_assets || []),
    ...fallbackQuotes,
  ];
  const uniqueOptions = [
    ...new Set(
      options.filter(Boolean).map((item) => String(item).toUpperCase()),
    ),
  ].sort();
  datalist.innerHTML = "";
  uniqueOptions.forEach((asset) => {
    const option = document.createElement("option");
    option.value = asset;
    datalist.appendChild(option);
  });
}


export async function loadInventory(refresh = false) {
  const verifyBtn = document.querySelector(
    '[data-action="inventory-verify"]',
  );
  try {
    if (refresh) {
      const term = document.getElementById("terminal-output");
      term.innerHTML += `<div class="log-line text-muted">Deep scan initiated: verifying all files for gaps...</div>`;
      term.scrollTop = term.scrollHeight;
      if (verifyBtn) verifyBtn.innerHTML = "⌛ Scanning...";
      updateProgress({
        visible: true,
        percent: 45,
        label: "Deep Quality Scan...",
      });
    }

    state.globalInventory = await api.inventory(refresh);

    if (refresh) {
      updateProgress({ visible: true, percent: 100, label: "Scan Complete" });
      setTimeout(() => {
        if (!state.isPipelineRunning) updateProgress({ visible: false });
      }, 3000);
    }
  } catch (e) {
    console.error("Failed to load inventory", e);
  } finally {
    if (verifyBtn) verifyBtn.innerHTML = "🛡️ Verify Quality";
  }

  try {
    const tbody = document.getElementById("inventory-tbody");
    tbody.innerHTML = "";

    const trainSelect = document.getElementById("train-symbol");

    // Save current selection
    const currTrain = trainSelect.value;

    trainSelect.innerHTML = '<option value="">Select Symbol...</option>';

    state.globalInventory.forEach((item) => {
      const tr = document.createElement("tr");

      // Format checkmarks for aggregations with status colors
      const aggs =
        item.aggregations.length > 0
          ? item.aggregations
              .map((a) => {
                let color = "var(--text-muted)"; // default
                if (a.status === "complete") color = "var(--accent-glow)";
                if (a.status === "warning") color = "#ffbd2e";
                if (a.status === "incomplete") color = "var(--red-main)";
                return `<span style="color:${color}" title="${a.message}">${a.interval}</span>`;
              })
              .join(", ")
          : '<span class="text-muted">None</span>';

      // Quality Status Dot + Fix button
      const quality = item.quality || {
        status: "incomplete",
        label: "N/A",
        message: "No data",
      };
      const dotClass = `status-dot quality-${quality.status}`;
      const showFix = quality.status !== "complete";
      const fixBtn = showFix
        ? `<button class="action-btn" style="font-size:0.6rem; padding:2px 6px; margin-left:6px; background:rgba(255,189,46,0.15); color:#ffbd2e; border-color:rgba(255,189,46,0.3);" data-action="symbol-fix" data-symbol="${escapeHtml(item.symbol)}">Fix</button>`
        : "";

      // Build existing intervals set for graying out
      const existingIntervals = item.aggregations.map((a) => a.interval);

      tr.innerHTML = `
                <td><strong>${item.symbol}</strong></td>
                <td><span class="text-muted">${item.start_date}</span> → <span class="text-muted">${item.end_date}</span></td>
                <td>${item.size_mb} MB</td>
                <td style="font-size:0.75rem;">${aggs}</td>
                <td>
                    <div style="display:flex; align-items:center;">
                        <div class="${dotClass}" title="${quality.label}: ${quality.message}"></div>
                        ${fixBtn}
                    </div>
                </td>
                <td style="position: relative;">
                    <button class="action-btn analysis-action-btn" data-action="symbol-analyze" data-symbol="${escapeHtml(item.symbol)}">Analyze</button>
                    <button class="action-btn" style="background: rgba(0, 229, 255, 0.1); color: var(--accent-glow); border-color: rgba(0, 229, 255, 0.2);" data-action="aggregate-panel" data-symbol="${escapeHtml(item.symbol)}">Aggregate</button>
                    <button class="action-btn" data-action="symbol-delete" data-symbol="${escapeHtml(item.symbol)}">Delete</button>
                    <!-- Hidden aggregation panel -->
                    <div id="agg-panel-${item.symbol}" style="display:none; position: absolute; top: 100%; right: 0; background: #161b22; padding: 12px; border: 1px solid rgba(0, 229, 255, 0.4); border-radius: 8px; z-index: 100; box-shadow: 0 8px 30px rgba(0,0,0,0.8); min-width: 200px;">
                        <div class="checkbox-group mb-2" style="flex-wrap: wrap; font-size: 0.7rem;">
                            ${[
                              "3m",
                              "5m",
                              "15m",
                              "30m",
                              "1h",
                              "2h",
                              "4h",
                              "6h",
                              "8h",
                              "12h",
                              "1d",
                              "3d",
                              "1w",
                            ]
                              .map((a) => {
                                const idSuffix = `agg-inline-${a}-${item.symbol}`;
                                const exists = existingIntervals.includes(a);
                                const disabled = exists ? "disabled" : "";
                                const style = exists
                                  ? "opacity: 0.4; text-decoration: line-through;"
                                  : "";
                                return `<label style="${style}"><input type="checkbox" id="${idSuffix}" value="${a}" ${disabled}> ${a}</label>`;
                              })
                              .join("")}
                        </div>
                        <div style="display: flex; gap: 8px; justify-content: space-between;">
                            <button class="outline-btn" style="padding: 4px 12px; font-size: 0.75rem; flex: 1;" data-action="aggregate-queue" data-symbol="${escapeHtml(item.symbol)}" data-start="${escapeHtml(item.start_date)}" data-end="${escapeHtml(item.end_date)}">+ Queue</button>
                            <button class="glow-btn" style="padding: 4px 12px; font-size: 0.75rem; flex: 1;" data-action="aggregate-run" data-symbol="${escapeHtml(item.symbol)}" data-start="${escapeHtml(item.start_date)}" data-end="${escapeHtml(item.end_date)}">Run Now</button>
                        </div>
                    </div>
                </td>
            `;
      tbody.appendChild(tr);

      // Populate dropdowns
      trainSelect.innerHTML += `<option value="${item.symbol}">${item.symbol}</option>`;
    });

    // Restore selection if still valid
    if (currTrain) trainSelect.value = currTrain;
    emit(EVENTS.INVENTORY_LOADED, state.globalInventory);
  } catch (e) {
    console.error("Failed to load inventory", e);
  }
}


export async function deleteSymbol(symbol) {
  if (!confirm(`Are you sure you want to delete all data for ${symbol}?`))
    return;
  try {
    await api.deleteSymbol(symbol);
    loadInventory();
  } catch (e) {
    alert("Failed to delete");
  }
}


export function fixQuality(symbol) {
  const item = state.globalInventory.find((i) => i.symbol === symbol);
  if (!item) return alert("Symbol not found in inventory");

  // The server derives the repair chain from the inventory. This used to be
  // reconstructed here by regex-matching the human-readable quality message,
  // which silently stopped working whenever that wording changed.
  submitTask(`Fix Quality [${symbol}]`, "/api/pipeline/repair", {
    symbol,
    flags: getResearchFlags(),
  });
}

// --- AI Inventory & Calendar ---


export function showAggregatePanel(symbol) {
  const p = document.getElementById(`agg-panel-${symbol}`);
  const isOpening = p.style.display === "none";
  p.style.display = isOpening ? "block" : "none";

  if (isOpening) {
    const item = state.globalInventory.find((i) => i.symbol === symbol);
    if (item) {
      const existing = item.aggregations.map((a) => a.interval);
      const possible = [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
      ];
      possible.forEach((a) => {
        const cb = document.getElementById(`agg-inline-${a}-${symbol}`);
        if (cb) {
          if (existing.includes(a)) {
            cb.checked = false;
            cb.disabled = true;
            cb.parentElement.style.opacity = "0.4";
            cb.parentElement.title = "Already present";
          } else {
            cb.disabled = false;
            cb.parentElement.style.opacity = "1";
            cb.parentElement.title = "";
          }
        }
      });
    }
  }
}


export function getCheckedAggregations(symbol) {
  const aggs = [];
  const possible = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
  ];
  possible.forEach((a) => {
    const cb = document.getElementById(`agg-inline-${a}-${symbol}`);
    if (cb && cb.checked) aggs.push(a);
  });
  return aggs;
}


export function getInlineAggConfig(symbol, start, end) {
  const itemInfo = state.globalInventory.find((i) => i.symbol === symbol);
  // Note: 1m is required for any aggregation
  if (
    !itemInfo.aggregations.some(
      (a) => a.interval === "1m" && a.status !== "incomplete",
    )
  ) {
    return {
      error:
        "Error: Valid 1m source data is missing! You must Backfill 1m data first.",
    };
  }
  const aggs = getCheckedAggregations(symbol);
  if (aggs.length === 0) return { error: "Select at least one aggregation" };

  return {
    taskName: `Agg & Auto-Features [${symbol}] to [${aggs.join(",")}]`,
    endpoint: "/api/pipeline/aggregate",
    body: {
      symbol,
      start,
      end,
      intervals: aggs,
      flags: getResearchFlags(),
    },
  };
}


export function queueInlineAggregation(symbol, start, end) {
  const { taskName, endpoint, body, error } = getInlineAggConfig(
    symbol,
    start,
    end,
  );
  if (error) return alert(error);
  document.getElementById(`agg-panel-${symbol}`).style.display = "none";
  submitTask(taskName, endpoint, body);
}


export function runInlineAggregation(symbol, start, end) {
  const { taskName, endpoint, body, error } = getInlineAggConfig(
    symbol,
    start,
    end,
  );
  if (error) return alert(error);
  document.getElementById(`agg-panel-${symbol}`).style.display = "none";
  submitTask(taskName, endpoint, body);
}
