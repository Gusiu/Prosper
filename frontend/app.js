let chartInstance = null;
let pollInterval = null;
let lastLogIdx = 0;
let pipelineQueue = [];
let isPipelineRunning = false;
let globalInventory = [];
let researchMode = false;
let symbolMetadata = { symbols: [], base_assets: [], quote_assets: [] };
const ANALYSIS_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"];
let analysisState = {
  symbol: null,
  interval: null,
  data: null,
  isSyncingRange: false,
  chart: null,
  candleSeries: null,
  volumeSeries: null,
  ma10Series: null,
  ma30Series: null,
  bbUpperSeries: null,
  bbMidSeries: null,
  bbLowerSeries: null,
  rsiChart: null,
  rsiSeries: null,
  macdChart: null,
  macdSeries: null,
  atrChart: null,
  atrSeries: null,
  mainRangeHandler: null,
  resizeHandler: null,
  rowsByTime: new Map(),
};

function isValidSymbolPart(value) {
  return /^[A-Z0-9]{1,15}$/.test(value);
}

// --- RESEARCH MODE ---
function initResearchMode() {
  const saved = localStorage.getItem("prosper_research_mode");
  researchMode = saved === "true";
  applyResearchModeUI();

  // Restore individual flag states
  const flags = JSON.parse(localStorage.getItem("prosper_research_flags") || "{}");
  if (flags.strict !== undefined) document.getElementById("research-strict").checked = flags.strict;
  if (flags.deterministic !== undefined) document.getElementById("research-deterministic").checked = flags.deterministic;
  if (flags.seed !== undefined) document.getElementById("research-seed").value = flags.seed;
  if (flags.metadata !== undefined) document.getElementById("research-metadata").checked = flags.metadata;
}

function toggleResearchMode() {
  const switchEl = document.getElementById("research-switch");
  const optionsEl = document.getElementById("research-options");
  const badgeEl = document.getElementById("research-badge");

  const isResearch = switchEl.classList.toggle("active");
  researchMode = isResearch;
  localStorage.setItem("prosper_research_mode", String(researchMode));

  if (isResearch) {
    optionsEl.style.display = "flex";
    badgeEl.innerText = "ON";
    badgeEl.className = "research-badge on";
  } else {
    optionsEl.style.display = "none";
    badgeEl.innerText = "OFF";
    badgeEl.className = "research-badge off";
  }
}

function applyResearchModeUI() {
  const toggle = document.getElementById("research-toggle");
  const sw = document.getElementById("research-switch");
  const badge = document.getElementById("research-badge");
  const options = document.getElementById("research-options");

  if (researchMode) {
    toggle.classList.add("active");
    sw.classList.add("active");
    badge.innerText = "ON";
    badge.className = "research-badge on";
    options.style.display = "flex";
  } else {
    toggle.classList.remove("active");
    sw.classList.remove("active");
    badge.innerText = "OFF";
    badge.className = "research-badge off";
    options.style.display = "none";
  }
}

function saveResearchFlags() {
  const flags = {
    strict: document.getElementById("research-strict").checked,
    deterministic: document.getElementById("research-deterministic").checked,
    seed: document.getElementById("research-seed").value,
    metadata: document.getElementById("research-metadata").checked,
  };
  localStorage.setItem("prosper_research_flags", JSON.stringify(flags));
}

// Save flags on change
["research-strict", "research-deterministic", "research-seed", "research-metadata"].forEach((id) => {
  document.getElementById(id)?.addEventListener("change", saveResearchFlags);
});

/**
 * Build research flag string to append to a CLI command.
 * Only appends flags relevant to the given command type.
 * commandType: 'pipeline' | 'predict' | 'eval' | 'generic'
 */
function getResearchFlags(commandType) {
  if (!researchMode) return "";
  saveResearchFlags();

  const strict = document.getElementById("research-strict").checked;
  const deterministic = document.getElementById("research-deterministic").checked;
  const seed = document.getElementById("research-seed").value;
  const metadata = document.getElementById("research-metadata").checked;

  let flags = "";

  // --strict applies to most commands
  if (strict) flags += " --strict";

  // --deterministic and --seed only for predict commands
  if (commandType === "predict") {
    if (deterministic) flags += " --deterministic";
    if (seed) flags += ` --seed ${seed}`;
  }

  // --save-metadata applies to all artifact-producing commands
  if (metadata) flags += " --save-metadata";

  return flags;
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
  loadSymbolMetadata();
  loadInventory();

  // Initialize research mode from localStorage
  initResearchMode();
  initAnalysisDrawer();
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

    const startInput = document.getElementById("pipe-start");
    const endInput = document.getElementById("pipe-end");
    
    // Default to last 3 months if no data
    startInput.value = data.default_start_month ? `${data.default_start_month}-01` : "2023-01-01";
    endInput.value = data.yesterday ? data.yesterday : new Date().toISOString().split('T')[0];

    document.getElementById("train-start").value = startInput.value;
    document.getElementById("train-end").value = endInput.value;

    document.getElementById("eval-start").value = "2024-01-01";
    document.getElementById("eval-end").value = endInput.value;
  } catch (e) {
    console.error("Failed to fetch dates", e);
  }
}

async function loadSymbolMetadata() {
  try {
    const res = await fetch("/api/meta/symbols");
    if (!res.ok) return;
    symbolMetadata = await res.json();
    populateCoinOptions();
  } catch (e) {
    console.warn("Symbol metadata unavailable", e);
  }
}

function populateCoinOptions() {
  const datalist = document.getElementById("coin-options");
  if (!datalist) return;

  const fallbackQuotes = ["USDT", "USDC", "FDUSD", "BTC", "ETH", "BNB"];
  const options = [
    ...(symbolMetadata.base_assets || []),
    ...(symbolMetadata.quote_assets || []),
    ...fallbackQuotes,
  ];
  const uniqueOptions = [...new Set(options.filter(Boolean).map((item) => String(item).toUpperCase()))].sort();
  datalist.innerHTML = "";
  uniqueOptions.forEach((asset) => {
    const option = document.createElement("option");
    option.value = asset;
    datalist.appendChild(option);
  });
}

async function loadInventory(refresh = false) {
  const verifyBtn = document.querySelector('button[onclick="loadInventory(true)"]');
  try {
    if (refresh) {
      const term = document.getElementById("terminal-output");
      term.innerHTML += `<div class="log-line text-muted">Deep scan initiated: verifying all files for gaps...</div>`;
      term.scrollTop = term.scrollHeight;
      if (verifyBtn) verifyBtn.innerHTML = "⌛ Scanning...";
      updateProgress({ visible: true, percent: 45, label: "Deep Quality Scan..." });
    }

    const url = refresh ? "/api/data/inventory?refresh=true" : "/api/data/inventory";
    const res = await fetch(url);
    globalInventory = await res.json();
    
    if (refresh) {
      updateProgress({ visible: true, percent: 100, label: "Scan Complete" });
      setTimeout(() => { if (!isPipelineRunning) updateProgress({ visible: false }); }, 3000);
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
    const evalSelect = document.getElementById("eval-symbol");

    // Save current selections
    const currTrain = trainSelect.value;
    const currEval = evalSelect.value;

    trainSelect.innerHTML = '<option value="">Select Symbol...</option>';
    evalSelect.innerHTML = '<option value="">Select Symbol...</option>';

    globalInventory.forEach((item) => {
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
      const quality = item.quality || { status: "incomplete", label: "N/A", message: "No data" };
      const dotClass = `status-dot quality-${quality.status}`;
      const showFix = quality.status !== "complete";
      const fixBtn = showFix
        ? `<button class="action-btn" style="font-size:0.6rem; padding:2px 6px; margin-left:6px; background:rgba(255,189,46,0.15); color:#ffbd2e; border-color:rgba(255,189,46,0.3);" onclick="fixQuality('${item.symbol}')">Fix</button>`
        : "";

      // Build existing intervals set for graying out
      const existingIntervals = item.aggregations.map(a => a.interval);
      
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
                    <button class="action-btn analysis-action-btn" onclick="openAnalysisDrawer('${item.symbol}')">Analyze</button>
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

function fixQuality(symbol) {
  const item = globalInventory.find((i) => i.symbol === symbol);
  if (!item) return alert("Symbol not found in inventory");

  const quality = item.quality;
  const rFlags = getResearchFlags("pipeline");
  let cmds = [];
  const sMonth = item.start_date.substring(0, 7);
  const eMonth = item.end_date.substring(0, 7);

  // 1. Fix gaps in 1m data (re-backfill the entire range)
  const agg1m = item.aggregations.find(a => a.interval === "1m");
  if (agg1m && agg1m.gaps > 0) {
    cmds.push(`python -m prosper.cli backfill --symbol ${symbol} --start ${sMonth} --end ${eMonth} --root ./data${rFlags}`);
  }

  // 2. Fix gaps in other intervals (re-aggregate from 1m)
  const gappedOther = item.aggregations
    .filter(a => a.interval !== "1m" && a.gaps > 0)
    .map(a => a.interval);

  if (gappedOther.length > 0) {
    const toArg = gappedOther.join(",");
    cmds.push(`python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${sMonth} --end ${eMonth} --to ${toArg} --root ./data${rFlags}`);
  }

  // 3. Fix broken parquet data (empty folders — re-aggregate)
  const brokenIntervals = item.aggregations
    .filter(a => a.status === "incomplete" && a.interval !== "1m" && (a.gaps || 0) === 0)
    .map(a => a.interval);

  if (brokenIntervals.length > 0) {
    const toArg = brokenIntervals.join(",");
    cmds.push(`python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${sMonth} --end ${eMonth} --to ${toArg} --root ./data${rFlags}`);
  }

  // 4. Fix missing features/labels
  const needsFeatures = new Set();

  // From per-interval status
  item.aggregations.forEach(a => {
    if (a.status === "warning") needsFeatures.add(a.interval);
  });

  // From quality message
  if (quality.message && quality.message.includes("Features/Labels missing")) {
    const match = quality.message.match(/for: (.+)/);
    if (match) {
      match[1].split(",").map(s => s.trim()).forEach(i => needsFeatures.add(i));
    }
  }

  for (const interval of needsFeatures) {
    cmds.push(`python -m prosper.cli features build --symbol ${symbol} --base-interval ${interval} --start ${item.start_date} --end ${item.end_date} --root ./data${rFlags}`);
    cmds.push(`python -m prosper.cli labels build --symbol ${symbol} --base-interval ${interval} --root ./data${rFlags}`);
  }

  if (cmds.length === 0) {
    return alert(`No automatic fix available for: ${quality.label} — ${quality.message}`);
  }

  const fullCmd = cmds.join(" && ");
  const taskName = `Fix Quality [${symbol}]`;
  
  console.log("Fix command:", fullCmd);
  pipelineQueue.push({ name: taskName, cmd: fullCmd });
  updateQueueUI();
  processQueue();
}

function showAggregatePanel(symbol) {
  const p = document.getElementById(`agg-panel-${symbol}`);
  const isOpening = p.style.display === "none";
  p.style.display = isOpening ? "block" : "none";

  if (isOpening) {
    const item = globalInventory.find((i) => i.symbol === symbol);
    if (item) {
      const existing = item.aggregations.map(a => a.interval);
      const possible = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w"];
      possible.forEach(a => {
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
  // Note: 1m is required for any aggregation
  if (!itemInfo.aggregations.some(a => a.interval === "1m" && a.status !== "incomplete")) {
    return {
      error:
        "Error: Valid 1m source data is missing! You must Backfill 1m data first.",
    };
  }
  const aggs = getCheckedAggregations(symbol);
  if (aggs.length === 0) return { error: "Select at least one aggregation" };
  
  const sMonth = start.substring(0, 7);
  const eMonth = end.substring(0, 7);
  const toArg = aggs.join(",");
  const rFlags = getResearchFlags("pipeline");
  
  let cmd = `python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${sMonth} --end ${eMonth} --to ${toArg} --root ./data${rFlags}`;
  
  const allIntervals = ["1m", ...aggs];
  for (const interval of allIntervals) {
      // Use full dates for features, month format for aggregate
      cmd += ` && python -m prosper.cli features build --symbol ${symbol} --base-interval ${interval} --start ${start} --end ${end} --root ./data${rFlags}`;
      cmd += ` && python -m prosper.cli labels build --symbol ${symbol} --base-interval ${interval} --root ./data${rFlags}`;
  }

  const taskName = `Agg & Auto-Features [${symbol}] to [${aggs.join(",")}]`;
  console.log("Full Generated Command:", cmd);
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
  // If we already have a full date (YYYY-MM-DD), use it or truncate to month if needed
  if (val.length === 10) {
    if (isEnd) return val; // Use selected end day
    return val;
  }
  
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
  const s = document.getElementById("pipe-start").value.substring(0, 7); // CLI expects YYYY-MM
  const e = document.getElementById("pipe-end").value.substring(0, 7);
  if (!s || !e) {
    alert("Please select both start and end dates.");
    return;
  }

  const sFull = document.getElementById("pipe-start").value;
  const eFull = document.getElementById("pipe-end").value;

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

  const rFlags = getResearchFlags("pipeline");
  let cmd = `python -m prosper.cli backfill --symbol ${symbol} --start ${s} --end ${e} --root ./data${rFlags}`;

  if (aggs.length > 0) {
    const toArg = aggs.join(",");
    cmd += ` && python -m prosper.cli aggregate --symbol ${symbol} --from 1m --start ${s} --end ${e} --to ${toArg} --root ./data${rFlags}`;
    
    // Automatically build features for ALL selected intervals + 1m
    const allIntervals = ["1m", ...aggs];
    for (const interval of allIntervals) {
        cmd += ` && python -m prosper.cli features build --symbol ${symbol} --base-interval ${interval} --start ${sFull} --end ${eFull} --root ./data${rFlags}`;
        cmd += ` && python -m prosper.cli labels build --symbol ${symbol} --base-interval ${interval} --root ./data${rFlags}`;
    }
  } else {
    // Only 1m if no aggregations selected
    cmd += ` && python -m prosper.cli features build --symbol ${symbol} --base-interval 1m --start ${sFull} --end ${eFull} --root ./data${rFlags}`;
    cmd += ` && python -m prosper.cli labels build --symbol ${symbol} --base-interval 1m --root ./data${rFlags}`;
  }
  
  console.log("Pipeline command generated:", cmd);

  const aggNamePart = aggs.length > 0 ? ` + [${aggs.join(",")}]` : "";
  const taskName = `Pipeline [${symbol}] ${s} to ${e} (Auto-Features)${aggNamePart}`;

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

  const rFlags = getResearchFlags("predict");
  let cmd = "";
  if (model === "ml") {
    cmd = `python -m prosper.cli predict ml --symbol ${symbol} --start ${s} --end ${e} --root ./data${rFlags}`;
  } else if (model === "gru") {
    cmd = `python -m prosper.cli predict gru --symbol ${symbol} --start ${s} --end ${e} --epochs ${ep} --root ./data${rFlags}`;
  } else {
    cmd = `python -m prosper.cli predict ${model} --symbol ${symbol} --start ${s} --end ${e} --max-epochs ${ep} --root ./data${rFlags}`;
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
      
      // Ensure progress bar is shown if running
      if (data.progress && data.progress.visible) {
        updateProgress(data.progress);
      }
    } else {
      taskDot.className = "status-dot";
      taskDot.style.background = "#8e9bb0";
      taskTxt.innerText = `Task: Idle (Queue: ${pipelineQueue.length})`;
      taskTxt.style.color = "var(--text-muted)";

      if (isPipelineRunning) {
        // Task just finished!
        isPipelineRunning = false;
        
        // Ensure we show 100%
        updateProgress({ visible: true, percent: 100, label: "Complete" });

        // Hide after delay
        setTimeout(() => {
          if (!isPipelineRunning) {
            const progContainer = document.getElementById("global-progress");
            if (progContainer) progContainer.style.display = "none";
          }
        }, 3000);

        loadInventory(true);
        processQueue();
      } else {
        // Hide immediately if truly idle
        const progContainer = document.getElementById("global-progress");
        if (progContainer && !isPipelineRunning) {
            progContainer.style.display = "none";
        }
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

// --- DATA ANALYSIS DRAWER ---
function initAnalysisDrawer() {
  [
    "analysis-show-volume",
    "show-ma10",
    "show-ma30",
    "show-bb",
    "show-rsi",
    "show-macd",
    "show-atr",
    "analysis-show-labels",
    "analysis-show-gaps",
  ].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.addEventListener("change", renderAnalysisCharts);
  });

  const intervalSelect = document.getElementById("analysis-interval");
  if (intervalSelect) intervalSelect.addEventListener("change", loadAnalysisData);
}

function openAnalysisDrawer(symbol) {
  const item = globalInventory.find((i) => i.symbol === symbol);
  if (!item) return alert("Symbol not found in inventory");

  analysisState.symbol = symbol;
  const drawer = document.getElementById("analysis-drawer");
  const backdrop = document.getElementById("analysis-backdrop");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.classList.add("open");

  document.getElementById("analysis-title").innerText = symbol;
  document.getElementById("analysis-subtitle").innerText = `${item.start_date} to ${item.end_date}`;
  document.getElementById("analysis-start").value = item.start_date !== "N/A" ? item.start_date : "";
  document.getElementById("analysis-end").value = item.end_date !== "N/A" ? item.end_date : "";
  document.querySelectorAll(".indicator-checkbox").forEach((cb) => { cb.checked = false; });
  document.getElementById("show-rsi").checked = true;
  document.getElementById("show-macd").checked = true;
  document.getElementById("show-atr").checked = true;
  document.getElementById("analysis-show-volume").checked = true;
  document.getElementById("analysis-show-labels").checked = false;
  document.getElementById("analysis-show-gaps").checked = true;
  populateAnalysisIntervals(item);
  loadAnalysisData();
}

function closeAnalysisDrawer() {
  document.getElementById("analysis-drawer").classList.remove("open");
  document.getElementById("analysis-drawer").setAttribute("aria-hidden", "true");
  document.getElementById("analysis-backdrop").classList.remove("open");
  disposeAnalysisCharts();
  analysisState.symbol = null;
  analysisState.data = null;
}

function populateAnalysisIntervals(item) {
  const select = document.getElementById("analysis-interval");
  select.innerHTML = "";
  const intervals = (item.aggregations || [])
    .map((agg) => agg.interval || agg)
    .filter(Boolean)
    .sort(compareAnalysisIntervals);

  const uniqueIntervals = intervals.length > 0 ? [...new Set(intervals)] : ["1m"];
  uniqueIntervals.forEach((interval) => {
    const option = document.createElement("option");
    option.value = interval;
    option.innerText = interval;
    select.appendChild(option);
  });

  const preferred = uniqueIntervals.includes("1d")
    ? "1d"
    : uniqueIntervals.includes("1h")
      ? "1h"
      : uniqueIntervals[0];
  select.value = preferred;
  analysisState.interval = preferred;
}

async function loadAnalysisData() {
  if (!analysisState.symbol) return;

  const interval = document.getElementById("analysis-interval").value || "1d";
  const start = document.getElementById("analysis-start").value;
  const end = document.getElementById("analysis-end").value;
  const limit = document.getElementById("analysis-limit").value || "1000";
  const status = document.getElementById("analysis-status");

  analysisState.interval = interval;
  status.innerText = `Loading ${analysisState.symbol} ${interval}...`;

  const params = new URLSearchParams({ limit });
  if (start) params.set("start", start);
  if (end) params.set("end", end);

  try {
    const res = await fetch(
      `/api/data/klines/${encodeURIComponent(analysisState.symbol)}/${encodeURIComponent(interval)}?${params.toString()}`,
    );
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Chart data request failed");
    }
    analysisState.data = await res.json();
    status.innerText = `${analysisState.data.rows} candles loaded, ${analysisState.data.gaps.length} gap(s) detected.`;
    renderAnalysisCharts();
  } catch (e) {
    disposeAnalysisCharts();
    status.innerText = e.message || "Failed to load chart data.";
  }
}

function renderAnalysisCharts() {
  if (!analysisState.data || !window.LightweightCharts) {
    const status = document.getElementById("analysis-status");
    if (status && !window.LightweightCharts) {
      status.innerText = "Lightweight Charts library is not available.";
    }
    return;
  }

  const candles = analysisState.data.candles || [];
  if (candles.length === 0) return;

  disposeAnalysisCharts();
  analysisState.rowsByTime = new Map(candles.map((row) => [String(row.time), row]));

  const showVolume = document.getElementById("analysis-show-volume").checked;
  const showMa10 = document.getElementById("show-ma10").checked;
  const showMa30 = document.getElementById("show-ma30").checked;
  const showBb = document.getElementById("show-bb").checked;
  const showRsi = document.getElementById("show-rsi").checked;
  const showMacd = document.getElementById("show-macd").checked;
  const showAtr = document.getElementById("show-atr").checked;
  const showLabels = document.getElementById("analysis-show-labels").checked;
  const showGaps = document.getElementById("analysis-show-gaps").checked;

  const container = document.getElementById("chart-container");
  const rsiContainer = document.getElementById("analysis-rsi-chart");
  const macdContainer = document.getElementById("analysis-macd-chart");
  const atrContainer = document.getElementById("analysis-atr-chart");
  const chartHeight = Math.max(360, Math.floor(container.clientHeight || 520));

  analysisState.chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: chartHeight,
    layout: {
      background: { type: "solid", color: "rgba(7, 11, 18, 0)" },
      textColor: "#8e9bb0",
      fontFamily: "Inter",
    },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.04)" },
      horzLines: { color: "rgba(255,255,255,0.04)" },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
    timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true },
  });

  analysisState.candleSeries = analysisState.chart.addCandlestickSeries({
    upColor: "#00e676",
    downColor: "#ff1744",
    borderUpColor: "#00e676",
    borderDownColor: "#ff1744",
    wickUpColor: "#00e676",
    wickDownColor: "#ff1744",
  });
  analysisState.candleSeries.setData(
    candles.map((row) => ({
      time: row.time,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    })),
  );

  if (showVolume) drawVolume(candles);
  overlayIndicators(candles, { showMa10, showMa30, showBb });
  if (showLabels) drawLabels(candles);
  drawRsiChart(candles, rsiContainer, showRsi);
  drawMacdChart(candles, macdContainer, showMacd);
  drawAtrChart(candles, atrContainer, showAtr);

  attachSyncHandlers(analysisState.chart);

  if (showGaps) {
    setTimeout(drawDataHealth, 50);
  }

  analysisState.resizeHandler = () => resizeAnalysisCharts();
  window.addEventListener("resize", analysisState.resizeHandler);
  updateAnalysisLegend({ time: candles[candles.length - 1].time });
}

function drawVolume(candles) {
  analysisState.volumeSeries = analysisState.chart.addHistogramSeries({
    priceFormat: { type: "volume" },
    priceScaleId: "",
  });
  analysisState.volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
  analysisState.volumeSeries.setData(
    candles.map((row) => ({
      time: row.time,
      value: row.volume,
      color: row.close >= row.open ? "rgba(0,230,118,0.28)" : "rgba(255,23,68,0.28)",
    })),
  );
}

function overlayIndicators(candles, options) {
  const ma10Data = candles
    .filter((row) => Number.isFinite(row.ma_10))
    .map((row) => ({ time: row.time, value: row.ma_10 }));
  const ma30Data = candles
    .filter((row) => Number.isFinite(row.ma_30))
    .map((row) => ({ time: row.time, value: row.ma_30 }));
  const bbUpperData = candles
    .filter((row) => Number.isFinite(row.bb_upper))
    .map((row) => ({ time: row.time, value: row.bb_upper }));
  const bbMidData = candles
    .filter((row) => Number.isFinite(row.bb_mid))
    .map((row) => ({ time: row.time, value: row.bb_mid }));
  const bbLowerData = candles
    .filter((row) => Number.isFinite(row.bb_lower))
    .map((row) => ({ time: row.time, value: row.bb_lower }));

  if (options.showMa10 && ma10Data.length > 0) {
    analysisState.ma10Series = analysisState.chart.addLineSeries({
      color: "#00e5ff",
      lineWidth: 2,
      title: "MA 10",
    });
    analysisState.ma10Series.setData(ma10Data);
  }
  if (options.showMa30 && ma30Data.length > 0) {
    analysisState.ma30Series = analysisState.chart.addLineSeries({
      color: "#ffbd2e",
      lineWidth: 2,
      title: "MA 30",
    });
    analysisState.ma30Series.setData(ma30Data);
  }
  if (options.showBb && bbUpperData.length > 0 && bbMidData.length > 0 && bbLowerData.length > 0) {
    analysisState.bbUpperSeries = analysisState.chart.addLineSeries({
      color: "rgba(186, 104, 255, 0.85)",
      lineWidth: 1,
      title: "BB Upper",
    });
    analysisState.bbMidSeries = analysisState.chart.addLineSeries({
      color: "rgba(186, 104, 255, 0.55)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      title: "BB Mid",
    });
    analysisState.bbLowerSeries = analysisState.chart.addLineSeries({
      color: "rgba(186, 104, 255, 0.85)",
      lineWidth: 1,
      title: "BB Lower",
    });
    analysisState.bbUpperSeries.setData(bbUpperData);
    analysisState.bbMidSeries.setData(bbMidData);
    analysisState.bbLowerSeries.setData(bbLowerData);
  }
}

function drawLabels(candles) {
  const markers = candles
    .filter((row) => row.direction)
    .map((row) => {
      if (row.direction === "long") {
        return {
          time: row.time,
          position: "belowBar",
          color: "#00e676",
          shape: "arrowUp",
          text: row.depth_bin === null ? "Long" : `Long ${row.depth_bin}`,
        };
      }
      if (row.direction === "short") {
        return {
          time: row.time,
          position: "aboveBar",
          color: "#ff1744",
          shape: "arrowDown",
          text: row.depth_bin === null ? "Short" : `Short ${row.depth_bin}`,
        };
      }
      return {
        time: row.time,
        position: "belowBar",
        color: "#8e9bb0",
        shape: "circle",
        text: "Flat",
      };
    });
  analysisState.candleSeries.setMarkers(markers);
}

function drawDataHealth() {
  const overlay = document.getElementById("gap-overlay");
  if (!overlay) return;
  overlay.innerHTML = "";
  if (!document.getElementById("analysis-show-gaps")?.checked) return;
  if (!analysisState.chart || !analysisState.data) return;

  (analysisState.data.gaps || []).forEach((gap) => {
    const x1 = analysisState.chart.timeScale().timeToCoordinate(gap.from);
    const x2 = analysisState.chart.timeScale().timeToCoordinate(gap.to);
    if (x1 === null || x2 === null) return;

    const band = document.createElement("div");
    band.className = "analysis-gap-band";
    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const rawWidth = maxX - minX;
    const finalWidth = Math.max(16, rawWidth);
    const center = minX + rawWidth / 2;
    band.style.left = `${center - finalWidth / 2}px`;
    band.style.width = `${finalWidth}px`;
    band.title = `Missing candles: ${gap.missing}`;
    overlay.appendChild(band);
  });
}

function drawRsiChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML = '<div class="analysis-empty-subchart">Indicators disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.rsi_14))
    .map((row) => ({ time: row.time, value: row.rsi_14 }));
  if (data.length === 0) {
    container.innerHTML = '<div class="analysis-empty-subchart">No RSI data</div>';
    return;
  }

  analysisState.rsiChart = LightweightCharts.createChart(container, buildSubChartOptions(container));
  analysisState.rsiSeries = analysisState.rsiChart.addLineSeries({ color: "#7dd3fc", lineWidth: 2 });
  analysisState.rsiSeries.setData(data);
  analysisState.rsiSeries.createPriceLine({ price: 70, color: "rgba(255,23,68,0.55)", lineStyle: LightweightCharts.LineStyle.Dashed });
  analysisState.rsiSeries.createPriceLine({ price: 30, color: "rgba(0,230,118,0.55)", lineStyle: LightweightCharts.LineStyle.Dashed });
  analysisState.rsiChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.rsiChart);
}

function drawMacdChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML = '<div class="analysis-empty-subchart">Indicators disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.macd_hist))
    .map((row) => ({
      time: row.time,
      value: row.macd_hist,
      color: row.macd_hist >= 0 ? "rgba(0,230,118,0.65)" : "rgba(255,23,68,0.65)",
    }));
  if (data.length === 0) {
    container.innerHTML = '<div class="analysis-empty-subchart">No MACD data</div>';
    return;
  }

  analysisState.macdChart = LightweightCharts.createChart(container, buildSubChartOptions(container));
  analysisState.macdSeries = analysisState.macdChart.addHistogramSeries({ priceFormat: { type: "price", precision: 4, minMove: 0.0001 } });
  analysisState.macdSeries.setData(data);
  analysisState.macdChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.macdChart);
}

function drawAtrChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML = '<div class="analysis-empty-subchart">Indicator disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.atr_14))
    .map((row) => ({ time: row.time, value: row.atr_14 }));
  if (data.length === 0) {
    container.innerHTML = '<div class="analysis-empty-subchart">No ATR data</div>';
    return;
  }

  analysisState.atrChart = LightweightCharts.createChart(container, buildSubChartOptions(container));
  analysisState.atrSeries = analysisState.atrChart.addLineSeries({ color: "#fb7185", lineWidth: 2 });
  analysisState.atrSeries.setData(data);
  analysisState.atrChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.atrChart);
}

function buildSubChartOptions(container) {
  return {
    width: container.clientWidth,
    height: 150,
    layout: {
      background: { type: "solid", color: "rgba(7, 11, 18, 0)" },
      textColor: "#8e9bb0",
      fontFamily: "Inter",
    },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.035)" },
      horzLines: { color: "rgba(255,255,255,0.035)" },
    },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
    timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true },
  };
}

function attachSyncHandlers(chart) {
  if (!chart) return;

  chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
    if (!range || analysisState.isSyncingRange) return;
    analysisState.isSyncingRange = true;
    [analysisState.chart, analysisState.rsiChart, analysisState.macdChart, analysisState.atrChart].forEach((c) => {
      if (c && c !== chart) c.timeScale().setVisibleRange(range);
    });
    analysisState.isSyncingRange = false;
    drawDataHealth();
  });

  chart.subscribeCrosshairMove((param) => {
    if (analysisState.isSyncingCrosshair) return;
    analysisState.isSyncingCrosshair = true;
    updateAnalysisLegend(param);

    if (!param || !param.time) {
      [analysisState.chart, analysisState.rsiChart, analysisState.macdChart, analysisState.atrChart].forEach((c) => {
        if (c && c !== chart && typeof c.clearCrosshairPosition === "function") c.clearCrosshairPosition();
      });
    } else {
      const row = analysisState.rowsByTime.get(String(param.time));
      if (row) {
        if (analysisState.chart && chart !== analysisState.chart) setSubchartCrosshair(analysisState.chart, analysisState.candleSeries, row.close, param.time);
        if (analysisState.rsiChart && chart !== analysisState.rsiChart) setSubchartCrosshair(analysisState.rsiChart, analysisState.rsiSeries, row.rsi_14, param.time);
        if (analysisState.macdChart && chart !== analysisState.macdChart) setSubchartCrosshair(analysisState.macdChart, analysisState.macdSeries, row.macd_hist, param.time);
        if (analysisState.atrChart && chart !== analysisState.atrChart) setSubchartCrosshair(analysisState.atrChart, analysisState.atrSeries, row.atr_14, param.time);
      }
    }
    analysisState.isSyncingCrosshair = false;
  });
}

function setSubchartCrosshair(chart, series, value, time) {
  if (!chart || !series || !Number.isFinite(value) || typeof chart.setCrosshairPosition !== "function") {
    return;
  }
  try {
    chart.setCrosshairPosition(value, time, series);
  } catch (e) {
    // Older CDN builds differ slightly; keep legend updates even if sync is unavailable.
  }
}

function clearSubchartCrosshair(chart) {
  if (chart && typeof chart.clearCrosshairPosition === "function") {
    chart.clearCrosshairPosition();
  }
}

function updateAnalysisLegend(param) {
  const legend = document.getElementById("chart-legend");
  if (!param || !param.time) {
    legend.innerText = "Hover over the chart to inspect values.";
    return;
  }

  const row = analysisState.rowsByTime.get(String(param.time));
  if (!row) return;

  const date = new Date(row.time * 1000).toISOString().replace("T", " ").slice(0, 16);
  const parts = [
    `${analysisState.symbol} ${analysisState.interval}`,
    date,
    `O ${formatNumber(row.open)}`,
    `H ${formatNumber(row.high)}`,
    `L ${formatNumber(row.low)}`,
    `C ${formatNumber(row.close)}`,
    `V ${formatNumber(row.volume)}`,
  ];

  if (Number.isFinite(row.ma_10)) parts.push(`MA10 ${formatNumber(row.ma_10)}`);
  if (Number.isFinite(row.ma_30)) parts.push(`MA30 ${formatNumber(row.ma_30)}`);
  if (Number.isFinite(row.bb_upper)) parts.push(`BBU ${formatNumber(row.bb_upper)}`);
  if (Number.isFinite(row.bb_lower)) parts.push(`BBL ${formatNumber(row.bb_lower)}`);
  if (Number.isFinite(row.rsi_14)) parts.push(`RSI ${formatNumber(row.rsi_14)}`);
  if (Number.isFinite(row.macd_hist)) parts.push(`MACD ${formatNumber(row.macd_hist)}`);
  if (Number.isFinite(row.atr_14)) parts.push(`ATR ${formatNumber(row.atr_14)}`);
  if (Number.isFinite(row.obv)) parts.push(`OBV ${formatNumber(row.obv)}`);
  if (row.direction) parts.push(`Label ${row.direction}`);
  legend.innerText = parts.join(" | ");
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return value.toFixed(2);
  if (Math.abs(value) >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}

function compareAnalysisIntervals(a, b) {
  const aIdx = ANALYSIS_INTERVALS.indexOf(a);
  const bIdx = ANALYSIS_INTERVALS.indexOf(b);
  const safeA = aIdx === -1 ? ANALYSIS_INTERVALS.length : aIdx;
  const safeB = bIdx === -1 ? ANALYSIS_INTERVALS.length : bIdx;
  return safeA - safeB || String(a).localeCompare(String(b));
}

function resizeAnalysisCharts() {
  const container = document.getElementById("chart-container");
  const rsiContainer = document.getElementById("analysis-rsi-chart");
  const macdContainer = document.getElementById("analysis-macd-chart");
  const atrContainer = document.getElementById("analysis-atr-chart");
  if (analysisState.chart) {
    analysisState.chart.applyOptions({ width: container.clientWidth });
    drawDataHealth();
  }
  if (analysisState.rsiChart) analysisState.rsiChart.applyOptions({ width: rsiContainer.clientWidth });
  if (analysisState.macdChart) analysisState.macdChart.applyOptions({ width: macdContainer.clientWidth });
  if (analysisState.atrChart) analysisState.atrChart.applyOptions({ width: atrContainer.clientWidth });
}

function disposeAnalysisCharts() {
  if (analysisState.resizeHandler) {
    window.removeEventListener("resize", analysisState.resizeHandler);
    analysisState.resizeHandler = null;
  }
  if (analysisState.chart && analysisState.mainRangeHandler) {
    analysisState.chart.timeScale().unsubscribeVisibleTimeRangeChange(analysisState.mainRangeHandler);
    analysisState.mainRangeHandler = null;
  }
  if (analysisState.chart) analysisState.chart.remove();
  if (analysisState.rsiChart) analysisState.rsiChart.remove();
  if (analysisState.macdChart) analysisState.macdChart.remove();
  if (analysisState.atrChart) analysisState.atrChart.remove();

  analysisState.chart = null;
  analysisState.candleSeries = null;
  analysisState.volumeSeries = null;
  analysisState.ma10Series = null;
  analysisState.ma30Series = null;
  analysisState.bbUpperSeries = null;
  analysisState.bbMidSeries = null;
  analysisState.bbLowerSeries = null;
  analysisState.rsiChart = null;
  analysisState.rsiSeries = null;
  analysisState.macdChart = null;
  analysisState.macdSeries = null;
  analysisState.atrChart = null;
  analysisState.atrSeries = null;
  analysisState.isSyncingRange = false;
  analysisState.rowsByTime = new Map();

  const gapOverlay = document.getElementById("gap-overlay");
  if (gapOverlay) gapOverlay.innerHTML = "";
}

function exportAnalysisPng() {
  if (!analysisState.chart) return alert("Load chart data first.");
  const canvas = analysisState.chart.takeScreenshot();
  const link = document.createElement("a");
  const interval = analysisState.interval || "chart";
  link.download = `${analysisState.symbol}-${interval}-analysis.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
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
