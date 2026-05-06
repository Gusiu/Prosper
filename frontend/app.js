let chartInstance = null;
let pollInterval = null;
let lastLogIdx = 0;
let pipelineQueue = [];
let isPipelineRunning = false;
let globalInventory = [];

function isValidSymbolPart(value) {
  return /^[A-Z0-9]{1,15}$/.test(value);
}

document.addEventListener("DOMContentLoaded", () => {
  // Check API Status
  fetch("/api/status")
    .then((res) => res.json())
    .then((data) => {
      console.log("API Status:", data);
    })
    .catch((err) => {
      document.getElementById("api-status").innerText = "Engine Offline";
      document.querySelector(".status-dot.online").style.backgroundColor =
        "var(--red-main)";
      document.querySelector(".status-dot.online").style.boxShadow =
        "0 0 10px var(--red-main)";
    });

  // Start background poll
  pollTaskStatus();

  // Load metadata and inventory
  initializeDates();
  loadInventory();
});

// --- TABS LOGIC ---
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

// --- DATES & INVENTORY ---
async function initializeDates() {
  try {
    const res = await fetch("/api/data/dates");
    const data = await res.json();

    document.getElementById("pipe-start").value = data.default_start_month;
    document.getElementById("pipe-end").value = data.yesterday_month;

    document.getElementById("train-start").value =
      data.default_start_month + "-01";
    document.getElementById("train-end").value = data.yesterday;

    document.getElementById("eval-start").value = "2024-01-01";
    document.getElementById("eval-end").value = data.yesterday;
  } catch (e) {
    console.error("Failed to fetch dates", e);
  }
}

async function loadInventory(refresh = false) {
  try {
    const url = refresh ? "/api/data/inventory?refresh=true" : "/api/data/inventory";
    const res = await fetch(url);
    globalInventory = await res.json();

    const tbody = document.getElementById("inventory-tbody");
    tbody.innerHTML = "";

    const trainSelect = document.getElementById("train-symbol");
    const evalSelect = document.getElementById("eval-symbol");

    // Save current selections
    const currTrain = trainSelect.value;
    const currEval = evalSelect.value;

    trainSelect.innerHTML = '<option value="">Select Symbol...</option>';
    evalSelect.innerHTML = '<option value="">Select Symbol...</option>';

    globalInventory.forEach((item) => {
      const tr = document.createElement("tr");

      // Format checkmarks for aggregations
      const aggs =
        item.aggregations.length > 0
          ? item.aggregations
              .map((a) => `<span style="color:var(--accent-glow)">${a}</span>`)
              .join(", ")
          : '<span class="text-muted">None</span>';

      tr.innerHTML = `
                <td><strong>${item.symbol}</strong></td>
                <td><span class="text-muted">${item.start_date}</span> → <span class="text-muted">${item.end_date}</span></td>
                <td>${item.size_mb} MB</td>
                <td style="font-size:0.75rem;">${aggs}</td>
                <td style="position: relative;">
                    <button class="action-btn" style="background: rgba(0, 229, 255, 0.1); color: var(--accent-glow); border-color: rgba(0, 229, 255, 0.2);" onclick="showAggregatePanel('${item.symbol}')">Aggregate</button>
                    <button class="action-btn" onclick="deleteSymbol('${item.symbol}')">Delete</button>
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
                                  const exists = item.aggregations.includes(a);
                                  const disabled = exists ? "disabled" : "";
                                  const style = exists
                                    ? "opacity: 0.4; text-decoration: line-through;"
                                    : "";
                                  return `<label style="${style}"><input type="checkbox" id="${idSuffix}" value="${a}" ${disabled}> ${a}</label>`;
                                })
                              .join("")}
                        </div>
                        <div style="display: flex; gap: 8px; justify-content: space-between;">
                            <button class="outline-btn" style="padding: 4px 12px; font-size: 0.75rem; flex: 1;" onclick="queueInlineAggregation('${item.symbol}', '${item.start_date}', '${item.end_date}')">+ Queue</button>
                            <button class="glow-btn" style="padding: 4px 12px; font-size: 0.75rem; flex: 1;" onclick="runInlineAggregation('${item.symbol}', '${item.start_date}', '${item.end_date}')">Run Now</button>
                        </div>
                    </div>
                </td>
            `;
      tbody.appendChild(tr);

      // Populate dropdowns
      trainSelect.innerHTML += `<option value="${item.symbol}">${item.symbol}</option>`;
      evalSelect.innerHTML += `<option value="${item.symbol}">${item.symbol}</option>`;
    });

    // Restore selections if still valid
    if (currTrain) trainSelect.value = currTrain;
    if (currEval) evalSelect.value = currEval;
  } catch (e) {
    console.error("Failed to load inventory", e);
  }
}

function updateQueueUI() {
  const tbody = document.getElementById("queue-tbody");
  const count = document.getElementById("queue-count");
  count.innerText = `(${pipelineQueue.length})`;

  if (pipelineQueue.length === 0) {
    tbody.innerHTML = `<tr><td class="text-muted">Queue is empty</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  pipelineQueue.forEach((item, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
            <td style="font-size: 0.75rem; word-break: break-all;">${item.name}</td>
            <td style="width: 80px; text-align: right;">
                <button class="action-btn" style="padding: 2px 4px; color: #2962ff;" onclick="moveQueue(${idx}, -1)">▲</button>
                <button class="action-btn" style="padding: 2px 4px; color: #2962ff;" onclick="moveQueue(${idx}, 1)">▼</button>
                <button class="action-btn" style="padding: 2px 4px; color: var(--red-main);" onclick="removeQueue(${idx})">X</button>
            </td>
        `;
    tbody.appendChild(tr);
  });
}

function moveQueue(idx, dir) {
  if (idx + dir < 0 || idx + dir >= pipelineQueue.length) return;
  const temp = pipelineQueue[idx];
  pipelineQueue[idx] = pipelineQueue[idx + dir];
  pipelineQueue[idx + dir] = temp;
  updateQueueUI();
}

function removeQueue(idx) {
  pipelineQueue.splice(idx, 1);
  updateQueueUI();
  const taskTxt = document.getElementById("task-status");
  if (!isPipelineRunning)
    taskTxt.innerText = `Task: Idle (Queue: ${pipelineQueue.length})`;
}

function updateTrainDates() {
  const sym = document.getElementById("train-symbol").value;
  const item = globalInventory.find((i) => i.symbol === sym);
  if (item && item.start_date !== "N/A") {
    document.getElementById("train-start").value = item.start_date;
    document.getElementById("train-end").value = item.end_date;
  }
}

function updateEvalDates() {
  const sym = document.getElementById("eval-symbol").value;
  const item = globalInventory.find((i) => i.symbol === sym);
  if (item && item.start_date !== "N/A") {
    // usually eval is recent
    document.getElementById("eval-start").value = "2024-01-01";
    document.getElementById("eval-end").value = item.end_date;
  }
}

async function deleteSymbol(symbol) {
  if (!confirm(`Are you sure you want to delete all data for ${symbol}?`))
    return;
  try {
    await fetch(`/api/data/${encodeURIComponent(symbol)}`, {
      method: "DELETE",
    });
    loadInventory();
  } catch (e) {
    alert("Failed to delete");
  }
}

function showAggregatePanel(symbol) {
  const p = document.getElementById(`agg-panel-${symbol}`);
  p.style.display = p.style.display === "none" ? "block" : "none";
}

function getCheckedAggregations(symbol) {
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

function getInlineAggConfig(symbol, start, end) {
  const itemInfo = globalInventory.find((i) => i.symbol === symbol);
  if (!itemInfo.aggregations.includes("1m")) {
    return {
      error:
        "Error: 1m source data is missing! You must Backfill 1m data first before you can aggregate 'up'.",
    };
  }
  const aggs = getCheckedAggregations(symbol);
  if (aggs.length === 0) return { error: "Select at least one aggregation" };
  const s = start.substring(0, 7);
  const e = end.substring(0, 7);
  const toArg = aggs.join(",");
  let cmd = `python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${s} --end ${e} --to ${toArg} --root ./data`;
  cmd += ` && python -m prosper.cli features build --symbol ${symbol} --start ${start} --end ${end} --root ./data`;
  cmd += ` && python -m prosper.cli labels build --symbol ${symbol} --root ./data`;
  const taskName = `Agg & Features [${symbol}] to ${aggs.join(",")}`;
  return { cmd, taskName };
}

function queueInlineAggregation(symbol, start, end) {
  const { cmd, taskName, error } = getInlineAggConfig(symbol, start, end);
  if (error) return alert(error);
  if (pipelineQueue.some((i) => i.name === taskName)) {
    if (!confirm(`${taskName} is already in the queue. Add again?`)) return;
  }
  pipelineQueue.push({ name: taskName, cmd });
  document.getElementById(`agg-panel-${symbol}`).style.display = "none";
  updateQueueUI();
  const taskTxt = document.getElementById("task-status");
  if (!isPipelineRunning)
    taskTxt.innerText = `Task: Idle (Queue: ${pipelineQueue.length})`;
  else taskTxt.innerText = `Task: RUNNING (Queue: ${pipelineQueue.length})`;
}

function runInlineAggregation(symbol, start, end) {
  const { cmd, taskName, error } = getInlineAggConfig(symbol, start, end);
  if (error) return alert(error);
  document.getElementById(`agg-panel-${symbol}`).style.display = "none";
  pipelineQueue.unshift({ name: taskName, cmd });
  processQueue();
}

function fmtMonth(val, isEnd) {
  if (!val) return "";
  let base = val + "-01";
  if (isEnd) {
    let [y, m] = val.split("-");
    const lastDay = new Date(Number(y), Number(m), 0).getDate();
    base = `${y}-${m}-${String(lastDay).padStart(2, "0")}`;
  }
  return base;
}

function addToQueue() {
  const base = document.getElementById("base-coin").value.trim().toUpperCase();
  const quote = document
    .getElementById("quote-coin")
    .value.trim()
    .toUpperCase();
  if (!base || !quote) {
    alert("Please enter both Base and Quote coins.");
    return;
  }
  if (!isValidSymbolPart(base) || !isValidSymbolPart(quote)) {
    alert("Symbols can contain only uppercase letters and digits.");
    return;
  }

  const symbol = base + quote;
  const s = document.getElementById("pipe-start").value;
  const e = document.getElementById("pipe-end").value;
  if (!s || !e) {
    alert("Please select both start and end months.");
    return;
  }

  const sFull = fmtMonth(s, false);
  const eFull = fmtMonth(e, true);

  const aggs = [];
  const possible = [
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
    if (
      document.getElementById(`agg-${a}`) &&
      document.getElementById(`agg-${a}`).checked
    )
      aggs.push(a);
  });

  let cmd = `python -m prosper.cli backfill --symbol ${symbol} --start ${s} --end ${e} --root ./data`;

  if (aggs.length > 0) {
    // Pass intervals as a comma-separated string to --to and set explicit root
    const toArg = aggs.join(",");
    cmd += ` && python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${s} --end ${e} --to ${toArg} --root ./data`;
  }

  cmd += ` && python -m prosper.cli features build --symbol ${symbol} --start ${sFull} --end ${eFull} --root ./data`;
  cmd += ` && python -m prosper.cli labels build --symbol ${symbol} --root ./data`;

  const aggNamePart = aggs.length > 0 ? ` + [${aggs.join(",")}]` : "";
  const taskName = `Pipeline [${symbol}] ${s} to ${e} (1m)${aggNamePart}`;
  if (pipelineQueue.some((i) => i.name === taskName)) {
    if (!confirm(`${taskName} is already in the queue. Add again?`)) return;
  }

  pipelineQueue.push({ name: taskName, cmd: cmd });

  // Clear input for next
  document.getElementById("base-coin").value = "";

  // Update queue UI
  updateQueueUI();
  const taskTxt = document.getElementById("task-status");
  if (!isPipelineRunning)
    taskTxt.innerText = `Task: Idle (Queue: ${pipelineQueue.length})`;
  else taskTxt.innerText = `Task: RUNNING (Queue: ${pipelineQueue.length})`;
}

function startQueueProcessing() {
  const baseInput = document.getElementById("base-coin");
  const baseValue = baseInput.value.trim();
  
  // If user typed something, add it to queue first
  if (baseValue) {
    addToQueue();
  }
  
  // If we have something to do (either just added or already there)
  if (pipelineQueue.length > 0) {
    if (isPipelineRunning) {
      // Just inform that it's queued if something is already running
      console.log("Task added to queue and will run automatically.");
    } else {
      processQueue();
    }
  } else if (!baseValue) {
    alert("Queue is empty. Please fill the form or use the 'Aggregate' panel.");
  }
}

async function processQueue() {
  if (isPipelineRunning || pipelineQueue.length === 0) return;

  const item = pipelineQueue.shift();
  updateQueueUI();
  isPipelineRunning = true;

  const term = document.getElementById("terminal-output");
  term.innerHTML += `<div class="log-line text-muted mt-2" style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">--- STARTING TASK: ${item.name} ---</div>`;
  term.scrollTop = term.scrollHeight;

  try {
    const res = await fetch("/api/task/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: item.cmd }),
    });

    if (!res.ok) {
      const data = await res.json();
      alert("Error: " + data.detail);
      isPipelineRunning = false;
    } else {
      lastLogIdx = 0;
      const progText = document.getElementById("progress-text");
      const symMatch = item.name.match(/\[([A-Z0-9]+)\]/);
      const symName = symMatch
        ? symMatch[1]
        : item.name.split(" ")[0] || "Task";

      if (item.name.toLowerCase().includes("pipeline")) {
        progText.innerText = `Full Pipeline: ${symName}...`;
      } else if (item.name.toLowerCase().includes("agg")) {
        progText.innerText = `Aggregating: ${symName}...`;
      } else {
        progText.innerText = `Processing: ${symName}...`;
      }
      updateProgress({ visible: true, percent: 0, label: progText.innerText });
    }
  } catch (e) {
    alert("Failed to execute command.");
    isPipelineRunning = false;
  }
}

// --- MODELS & EVALUATION ---
function runTrain() {
  const symbol = document.getElementById("train-symbol").value;
  if (!symbol) return alert("Select symbol first");

  const model = document.getElementById("model-select").value;
  const s = document.getElementById("train-start").value;
  const e = document.getElementById("train-end").value;
  const ep = document.getElementById("train-epochs").value;

  let cmd = "";
  if (model === "ml") {
    cmd = `python -m prosper.cli predict ml --symbol ${symbol} --start ${s} --end ${e} --root ./data`;
  } else if (model === "gru") {
    cmd = `python -m prosper.cli predict gru --symbol ${symbol} --start ${s} --end ${e} --epochs ${ep} --root ./data`;
  } else {
    cmd = `python -m prosper.cli predict ${model} --symbol ${symbol} --start ${s} --end ${e} --max-epochs ${ep} --root ./data`;
  }

  pipelineQueue.push({
    name: `Train ${model.toUpperCase()} on ${symbol}`,
    cmd: cmd,
  });
  updateQueueUI();
  processQueue();
}

// --- TERMINAL LOGIC ---
function pollTaskStatus() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(checkTaskLogs, 1000);
}

async function checkTaskLogs() {
  try {
    const res = await fetch(`/api/task/logs?start_idx=${lastLogIdx}`);
    const data = await res.json();

    const term = document.getElementById("terminal-output");
    let added = false;

    // Use high-quality progress data from backend if available
    if (data.progress && data.progress.visible) {
      updateProgress(data.progress);
    }

    data.logs.forEach((log) => {
      const div = document.createElement("div");
      div.className = "log-line";
      div.textContent = log;
      term.appendChild(div);
      lastLogIdx++;
      added = true;
    });

    if (added) {
      term.scrollTop = term.scrollHeight;
    }

    const taskDot = document.getElementById("task-dot");
    const taskTxt = document.getElementById("task-status");

    if (data.is_running) {
      taskDot.className = "status-dot active";
      taskTxt.innerText = `Task: RUNNING (Queue: ${pipelineQueue.length})`;
      taskTxt.style.color = "var(--accent-glow)";
    } else {
      taskDot.className = "status-dot";
      taskDot.style.background = "#8e9bb0";
      taskTxt.innerText = `Task: Idle (Queue: ${pipelineQueue.length})`;
      taskTxt.style.color = "var(--text-muted)";

      // Hide progress when idle
      const progContainer = document.getElementById("global-progress");
      if (progContainer.style.display !== "none") {
        progContainer.style.display = "none";
      }

      if (isPipelineRunning) {
        // Task just finished!
        isPipelineRunning = false;
        loadInventory(true); // Refresh inventory after any task finishes
        processQueue(); // Process next in queue
      }
    }
  } catch (e) {
    // silent
  }
}

function updateProgress(progress) {
  if (!progress || !progress.visible) return;
  const progContainer = document.getElementById("global-progress");
  const progFill = document.getElementById("progress-bar-fill");
  const progText = document.getElementById("progress-text");
  const progPct = document.getElementById("progress-pct");
  const pct = Math.max(0, Math.min(100, Number(progress.percent || 0)));

  progContainer.style.display = "flex";
  progFill.style.width = `${pct}%`;
  progText.innerText = progress.label || "Processing";
  progPct.innerText = `${pct.toFixed(1)}%`;
}

// --- CHART RENDERING ---
function animateValue(id, start, end, duration, prefix = "", suffix = "") {
  const obj = document.getElementById(id);
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const current = (progress * (end - start) + start).toFixed(2);
    obj.innerHTML = prefix + current + suffix;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

async function runBacktest() {
  const symbol = document.getElementById("eval-symbol").value;
  if (!symbol) return alert("Select symbol first");

  const start = document.getElementById("eval-start").value;
  const end = document.getElementById("eval-end").value;

  try {
    document.getElementById("terminal-output").innerHTML +=
      `<div class="log-line text-muted">Fetching backtest details from API...</div>`;
    const res = await fetch(
      `/api/backtest/${symbol}?start=${start}&end=${end}`,
    );

    if (!res.ok) {
      const err = await res.json();
      alert("Evaluating Error: " + err.detail);
      return;
    }

    const data = await res.json();

    animateValue("val-roi", 0, data.roi, 1000, "", "%");
    animateValue(
      "val-capital",
      Math.max(0, data.final_capital - 2000),
      data.final_capital,
      1000,
      "$",
      "",
    );
    animateValue("val-drawdown", 0, data.max_drawdown, 1000, "", "%");

    document.getElementById("val-trades").innerText = data.total_trades;

    const roiEl = document.getElementById("val-roi");
    roiEl.className =
      "metric-value fade-in " + (data.roi >= 0 ? "text-green" : "text-red");

    drawChart(data.chart_data);
    document.getElementById("terminal-output").innerHTML +=
      `<div class="log-line text-muted">Backtest complete. Chart rendered.</div>`;
  } catch (e) {
    alert("Failed to connect to Backtest API");
  }
}

function drawChart(chartData) {
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
