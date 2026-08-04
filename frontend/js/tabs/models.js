/** AI Models tab: run inventory, training form, prediction calendar. */

import { api } from "../api.js";
import { EVENTS, emit } from "../events.js";
import { disposeAnalysisCharts } from "../analysis.js";
import { getResearchFlags } from "../research.js";
import { submitTask } from "../queue.js";
import { ANALYSIS_INTERVALS, state } from "../state.js";

let aiModelRuns = [];
let analysisDrawerContext = { symbol: null, model_type: null, timestamp: null };
// The calendar shows one month, so it fetches one month. Loading the whole
// run to page through it client-side pulled ~5 MB for a 1d run and 24x that
// for 1h — the endpoint has always supported year/month narrowing.
const predictionCalendarState = {
  rowsByDate: new Map(),
  year: 0,
  month: 0,
  context: null,
  months: [],
};

export function updateTrainDates() {
  const sym = document.getElementById("train-symbol").value;
  const item = state.globalInventory.find((i) => i.symbol === sym);
  if (item && item.start_date !== "N/A") {
    document.getElementById("train-start").value = item.start_date;
    document.getElementById("train-end").value = item.end_date;
  }
  // Ensure interval select reflects available prepared aggregations
  try {
    populateTrainIntervals(sym);
  } catch (e) {
    console.warn("populateTrainIntervals failed", e);
  }
  // Update epochs visibility based on currently selected model
  try {
    updateTrainModelUI();
  } catch (e) {
    /* ignore */
  }
}

// Populate `#train-interval` based on prepared aggregations for the symbol


export function populateTrainIntervals(symbol) {
  const sel = document.getElementById("train-interval");
  if (!sel) return;
  sel.innerHTML = "";
  const item = state.globalInventory.find((i) => i.symbol === symbol);
  if (
    !item ||
    !Array.isArray(item.aggregations) ||
    item.aggregations.length === 0
  ) {
    ["1d", "1h", "1m"].forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.text = v;
      sel.appendChild(opt);
    });
    sel.value = "1d";
    return;
  }

  const available = item.aggregations
    .filter((a) => a && a.interval && a.status !== "incomplete")
    .map((a) => a.interval);

  const ordered = ANALYSIS_INTERVALS.filter((i) => available.includes(i));
  const toUse =
    ordered.length > 0
      ? ordered
      : [...new Set(item.aggregations.map((a) => a.interval))];

  toUse.forEach((iv) => {
    if (!iv) return;
    const opt = document.createElement("option");
    opt.value = iv;
    opt.text = iv;
    sel.appendChild(opt);
  });

  if ([...sel.options].some((o) => o.value === "1d")) sel.value = "1d";
  else if (sel.options.length > 0) sel.selectedIndex = 0;
}

// Show epochs input only for DL models (gru, tft)


export function updateTrainModelUI() {
  const model = document.getElementById("model-select")?.value;
  const wrapper = document.getElementById("train-epochs-wrapper");
  if (!wrapper) return;
  if (model === "gru" || model === "tft") wrapper.style.display = "flex";
  else wrapper.style.display = "none";
}


export async function loadAIInventory() {
  try {
    const data = await api.models();
    aiModelRuns = data.models || [];
    renderAIInventory(aiModelRuns);
    // The evaluation and backtest tabs own their run pickers; tell them rather
    // than reaching into their modules.
    emit(EVENTS.RUNS_LOADED, aiModelRuns);
  } catch (e) {
    console.error("Failed to load AI inventory", e);
  }
}


export function renderAIInventory(models) {
  const tbody = document.getElementById("ai-inventory-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (!models || models.length === 0) {
    tbody.innerHTML = `<tr><td class="text-muted">No model runs</td></tr>`;
    return;
  }

  models.forEach((m) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${m.symbol}</td>
      <td>${m.model_type}</td>
      <td>${m.interval || "1d"}</td>
      <td>${m.timestamp}</td>
      <td>${m.predictions_files || 0}</td>
      <td></td>
    `;
    const btnCell = tr.querySelector("td:last-child");
    const viewBtn = document.createElement("button");
    viewBtn.className = "action-btn";
    viewBtn.textContent = "View";
    viewBtn.addEventListener("click", () =>
      openPredictionCalendar(
        m.symbol,
        m.model_type,
        m.timestamp,
        m.interval || "1d",
      ),
    );
    const delBtn = document.createElement("button");
    delBtn.className = "action-btn danger-btn";
    delBtn.style.marginLeft = "6px";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", () =>
      deleteModelRun(m.symbol, m.model_type, m.timestamp, m.interval || "1d"),
    );
    btnCell.appendChild(viewBtn);
    btnCell.appendChild(delBtn);
    tbody.appendChild(tr);
  });
}


export async function deleteModelRun(symbol, model_type, timestamp, interval = "1d") {
  if (
    !confirm(
      `Delete model run ${model_type} (${interval}) for ${symbol} @ ${timestamp}?`,
    )
  ) {
    return;
  }
  try {
    await api.deleteRun(symbol, model_type, timestamp, interval);
    loadAIInventory();
  } catch (error) {
    console.error(error);
    alert(`Delete failed: ${error.message}`);
  }
}


export function runTrain() {
  const symbol = document.getElementById("train-symbol").value;
  if (!symbol) return alert("Select symbol first");

  const model = document.getElementById("model-select").value;
  const interval = document.getElementById("train-interval")?.value || "1d";

  submitTask(`Train ${model.toUpperCase()} on ${symbol}`, "/api/train/run", {
    symbol,
    model_type: model,
    interval,
    start: document.getElementById("train-start").value,
    end: document.getElementById("train-end").value,
    epochs: Number(document.getElementById("train-epochs").value) || 5,
    flags: getResearchFlags(),
  });
}



export function showPredictionCalendarLoading() {
  const chartContainer = document.getElementById("chart-container");
  if (chartContainer) chartContainer.style.display = "none";
  const subcharts = document.querySelector(".analysis-subcharts");
  if (subcharts) subcharts.style.display = "none";
  const toolbar = document.querySelector(".analysis-toolbar");
  if (toolbar) toolbar.style.display = "none";
  const layers = document.querySelector(".analysis-layers");
  if (layers) layers.style.display = "none";

  const cc = document.getElementById("prediction-calendar-container");
  if (!cc) return;
  cc.style.display = "block";
  cc.innerHTML = `
    <div class="calendar-loading">
      <div class="calendar-spinner"></div>
      <div class="calendar-loading-text">Loading predictions...</div>
    </div>
  `;
}


export async function openPredictionCalendar(
  symbol,
  model_type,
  timestamp,
  interval = "1d",
) {
  const title = document.getElementById("analysis-title");
  const subtitle = document.getElementById("analysis-subtitle");
  if (title) title.innerText = `${model_type.toUpperCase()} - ${symbol}`;
  if (subtitle) subtitle.innerText = `Run: ${timestamp} (${interval})`;
  document.getElementById("analysis-backdrop").classList.add("open");
  const drawer = document.getElementById("analysis-drawer");
  if (drawer) {
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
  }

  disposeAnalysisCharts();
  showPredictionCalendarLoading();
  document.getElementById("analysis-status").innerText =
    "Loading predictions...";

  try {
    const run = { symbol, model_type, timestamp, interval };
    const { months = [] } = await api.predictionMonths(run);
    predictionCalendarState.context = run;
    predictionCalendarState.months = months;

    // Save context for download action
    analysisDrawerContext = run;

    // Check whether feature_importances.json exists for this run and toggle download button
    const fiButton = document.getElementById("download-feature-imp-btn");
    if (fiButton) {
      try {
        await api.featureImportances({ symbol, model_type, timestamp, interval });
        fiButton.style.display = "inline-block";
      } catch {
        fiButton.style.display = "none";
      }
    }

    if (months.length === 0) {
      predictionCalendarState.rowsByDate = new Map();
      document.getElementById("analysis-status").innerText = "No predictions in this run";
      renderPredictionCalendar();
      return;
    }

    const [lastYear, lastMonth] = months[months.length - 1].split("-").map(Number);
    predictionCalendarState.year = lastYear;
    predictionCalendarState.month = lastMonth - 1;
    await loadPredictionMonth();
  } catch (e) {
    console.error(e);
    document.getElementById("analysis-status").innerText =
      `Error loading predictions`;
    const cc = document.getElementById("prediction-calendar-container");
    if (cc) {
      cc.innerHTML =
        '<div class="analysis-empty-subchart">Failed to load predictions.</div>';
    }
  }
}


async function loadPredictionMonth() {
  const { context, year, month } = predictionCalendarState;
  if (!context) return;
  const status = document.getElementById("analysis-status");
  try {
    const data = await api.predictions({ ...context, year, month: month + 1 });
    const rows = data.predictions || [];
    predictionCalendarState.rowsByDate = new Map(rows.map((r) => [r.date, r]));
    if (status) status.innerText = `Loaded ${rows.length} rows for ${year}-${String(month + 1).padStart(2, "0")}`;
  } catch (e) {
    console.error(e);
    predictionCalendarState.rowsByDate = new Map();
    if (status) status.innerText = "Error loading predictions";
  }
  renderPredictionCalendar();
}


export async function changePredictionMonth(delta) {
  let { year, month } = predictionCalendarState;
  month += delta;
  if (month > 11) {
    month = 0;
    year++;
  } else if (month < 0) {
    month = 11;
    year--;
  }
  predictionCalendarState.month = month;
  predictionCalendarState.year = year;
  await loadPredictionMonth();
}


export function renderPredictionCalendar() {
  const { rowsByDate, year, month } = predictionCalendarState;
  const cc = document.getElementById("prediction-calendar-container");
  if (!cc) return;
  cc.style.display = "block";
  cc.innerHTML = "";

  if (rowsByDate.size === 0) {
    cc.innerHTML =
      '<div class="analysis-empty-subchart">No predictions available for this run.</div>';
    return;
  }

  const cal = document.createElement("div");
  cal.style.padding = "12px";

  // Navigation Header
  const navHeader = document.createElement("div");
  navHeader.style.display = "flex";
  navHeader.style.justifyContent = "space-between";
  navHeader.style.alignItems = "center";
  navHeader.style.marginBottom = "16px";
  navHeader.style.background = "rgba(255,255,255,0.03)";
  navHeader.style.padding = "8px 12px";
  navHeader.style.borderRadius = "8px";
  navHeader.style.border = "1px solid rgba(255,255,255,0.05)";

  const leftPart = document.createElement("div");
  leftPart.style.display = "flex";
  leftPart.style.gap = "8px";
  leftPart.style.alignItems = "center";

  const prevBtn = document.createElement("button");
  prevBtn.className = "outline-btn";
  prevBtn.style.padding = "4px 10px";
  prevBtn.innerHTML = "&larr;";
  prevBtn.onclick = () => changePredictionMonth(-1);

  const nextBtn = document.createElement("button");
  nextBtn.className = "outline-btn";
  nextBtn.style.padding = "4px 10px";
  nextBtn.innerHTML = "&rarr;";
  nextBtn.onclick = () => changePredictionMonth(1);

  const titleH = document.createElement("div");
  titleH.style.fontWeight = "700";
  titleH.style.fontSize = "1.1rem";
  titleH.style.color = "var(--accent-glow)";
  titleH.style.marginLeft = "8px";
  const d = new Date(year, month, 1);
  titleH.innerText = d.toLocaleString(undefined, {
    month: "long",
    year: "numeric",
  });

  leftPart.appendChild(prevBtn);
  leftPart.appendChild(nextBtn);
  leftPart.appendChild(titleH);

  const rightPart = document.createElement("div");
  rightPart.style.display = "flex";
  rightPart.style.gap = "8px";
  rightPart.style.alignItems = "center";

  const monthInput = document.createElement("input");
  monthInput.type = "number";
  monthInput.className = "glass-input";
  monthInput.style.width = "65px";
  monthInput.style.textAlign = "center";
  monthInput.value = month + 1;
  monthInput.min = 1;
  monthInput.max = 12;

  const yearInput = document.createElement("input");
  yearInput.type = "number";
  yearInput.className = "glass-input";
  yearInput.style.width = "85px";
  yearInput.style.textAlign = "center";
  yearInput.value = year;

  const jumpBtn = document.createElement("button");
  jumpBtn.className = "glow-btn";
  jumpBtn.style.padding = "4px 12px";
  jumpBtn.innerText = "Go";
  jumpBtn.onclick = () => {
    const m = parseInt(monthInput.value) - 1;
    const y = parseInt(yearInput.value);
    if (!isNaN(m) && !isNaN(y)) {
      predictionCalendarState.month = Math.max(0, Math.min(11, m));
      predictionCalendarState.year = y;
      // Jumping to a month must fetch it — only the current month is in memory.
      loadPredictionMonth();
    }
  };

  rightPart.appendChild(monthInput);
  rightPart.appendChild(yearInput);
  rightPart.appendChild(jumpBtn);

  navHeader.appendChild(leftPart);
  navHeader.appendChild(rightPart);
  cal.appendChild(navHeader);

  const table = document.createElement("table");
  table.style.width = "100%";
  table.style.borderCollapse = "collapse";

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].forEach((day) => {
    const th = document.createElement("th");
    th.style.padding = "10px";
    th.style.color = "#94a3b8";
    th.style.fontSize = "0.8rem";
    th.style.textTransform = "uppercase";
    th.style.letterSpacing = "0.05em";
    th.innerText = day;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  let curDay = 1 - firstDay;

  for (let w = 0; w < 6; w++) {
    const tr = document.createElement("tr");
    let hasDays = false;
    for (let d = 0; d < 7; d++) {
      const td = document.createElement("td");
      td.style.padding = "4px";
      td.style.width = "14.28%";
      td.style.verticalAlign = "top";

      if (curDay >= 1 && curDay <= daysInMonth) {
        hasDays = true;
        const dateStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(curDay).padStart(2, "0")}`;
        const dayDiv = document.createElement("div");
        dayDiv.style.padding = "8px";
        dayDiv.style.borderRadius = "8px";
        dayDiv.style.minHeight = "65px";
        dayDiv.style.border = "1px solid rgba(255,255,255,0.03)";
        dayDiv.style.transition = "all 0.2s ease";

        const label = document.createElement("div");
        label.style.fontWeight = "700";
        label.style.fontSize = "0.9rem";
        label.style.marginBottom = "4px";
        label.innerText = curDay;
        dayDiv.appendChild(label);

        const rowData = rowsByDate.get(dateStr);
        if (rowData) {
          // One horizon, not a maximum across three. Taking max(P_long) and
          // max(P_short) separately pulled the two numbers from different
          // horizons, so the pair never summed to 100 and a day could be
          // coloured bullish while another horizon read 99% short. The cell is
          // one day wide, so the shortest horizon is the one it can speak for;
          // the popover still shows all three.
          const summary = rowData.short || rowData.medium || rowData.long || {};
          const pLong = summary.P_long || 0;
          const pShort = summary.P_short || 0;

          if (pLong >= 0.6) {
            dayDiv.style.background = "rgba(0,230,118,0.08)";
            dayDiv.style.borderColor = "rgba(0,230,118,0.2)";
          } else if (pShort >= 0.6) {
            dayDiv.style.background = "rgba(255,23,68,0.08)";
            dayDiv.style.borderColor = "rgba(255,23,68,0.2)";
          } else {
            dayDiv.style.background = "rgba(255,255,255,0.02)";
          }

          const mini = document.createElement("div");
          mini.style.fontSize = "0.7rem";
          mini.style.color = "#94a3b8";
          mini.innerText = `4w  L:${Math.round(pLong * 100)}% S:${Math.round(pShort * 100)}%`;
          dayDiv.appendChild(mini);

          dayDiv.style.cursor = "pointer";
          dayDiv.addEventListener("mouseenter", (e) => {
            dayDiv.style.transform = "translateY(-2px)";
            dayDiv.style.boxShadow = "0 4px 12px rgba(0,0,0,0.3)";
            showPredictionPopover(e, rowData);
          });
          dayDiv.addEventListener("mouseleave", (e) => {
            dayDiv.style.transform = "";
            dayDiv.style.boxShadow = "";
            hidePredictionPopover();
          });
        } else {
          dayDiv.style.opacity = "0.15";
          dayDiv.style.background = "rgba(255,255,255,0.01)";
        }
        td.appendChild(dayDiv);
      }
      tr.appendChild(td);
      curDay++;
    }
    if (hasDays) tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  cal.appendChild(table);
  cc.appendChild(cal);
}


export function showPredictionPopover(e, row) {
  let pop = document.getElementById("prediction-popover");
  if (!pop) {
    pop = document.createElement("div");
    pop.id = "prediction-popover";
    pop.style.position = "absolute";
    pop.style.zIndex = "9999";
    pop.style.background = "rgba(7,11,18,0.95)";
    pop.style.border = "1px solid rgba(255,255,255,0.06)";
    pop.style.padding = "10px";
    pop.style.borderRadius = "8px";
    pop.style.minWidth = "240px";
    pop.style.color = "#e6eef8";
    document.body.appendChild(pop);
  }
  try {
    function _horizonHtml(label, obj, durationTxt) {
      // Read whichever direction classes the run carries. Two-class runs have
      // no P_flat; a legacy three-class run still does, and its middle band
      // should keep rendering rather than silently vanish into the bar.
      const P_long = (obj && obj.P_long) || 0;
      const P_short = (obj && obj.P_short) || 0;
      const P_flat = (obj && obj.P_flat) || 0;
      const pL = Math.round(P_long * 100);
      const pF = Math.round(P_flat * 100);
      const pS = Math.round(P_short * 100);
      const hasFlat = obj && obj.P_flat != null;

      // Depth bins visualization (vertical micro-bars + labels)
      let depthHtml = "";
      try {
        const depthBins = obj && (obj.depth_long_bins || obj.depth_short_bins);
        if (depthBins && typeof depthBins === "object") {
          // Order the run's own bins by magnitude. The labels are not
          // hardcoded here on purpose: runs written before the tail was
          // extended carry a `34+` bin that the current scheme does not have,
          // and a fixed list would render them as three empty bars plus a
          // silently dropped one.
          const entries = Object.entries(depthBins).sort(
            (a, b) => parseFloat(a[0]) - parseFloat(b[0]),
          );
          const bars = entries
            .map(([k, v]) => {
              const h = Math.max(6, Math.round((v || 0) * 100));
              return `<div title="${k}: ${Math.round((v || 0) * 100)}%" style="width:10px; background:linear-gradient(to top,#60ffb1,#0bbf53); height:${h}%; border-radius:3px;"></div>`;
            })
            .join("");
          const labels = entries
            .map(
              ([k]) =>
                `<div style="width:10px; font-size:0.62rem; color:#94a3b8; text-align:center; margin-top:4px;">${k}</div>`,
            )
            .join("");
          depthHtml = `
            <div style="display:flex; gap:6px; align-items:flex-end; height:30px; margin-top:6px;">${bars}</div>
            <div style="display:flex; gap:6px; margin-top:6px;">${labels}</div>
          `;
        }
      } catch (err) {
        depthHtml = "";
      }

      const durationHtml = durationTxt
        ? `<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;">${durationTxt}</div>`
        : "";

      return `
        <div style="margin-bottom:6px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; font-size:0.92rem;">${label}</div>
            ${durationHtml}
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-top:4px;">
            <div style="flex:1; max-width:160px; background:rgba(255,255,255,0.06); height:8px; border-radius:6px; overflow:hidden; position:relative;">
              <div style="position:absolute; left:0; top:0; height:8px; background:rgba(0,230,118,0.85); width:${pL}%;"></div>
              ${hasFlat ? `<div style="position:absolute; left:${pL}%; top:0; height:8px; background:rgba(255,189,46,0.85); width:${pF}%;"></div>` : ""}
              <div style="position:absolute; left:${pL + (hasFlat ? pF : 0)}%; top:0; height:8px; background:rgba(255,23,68,0.85); width:${pS}%;"></div>
            </div>
            <div style="width:86px; font-size:0.82rem; color:#cbd5e1; text-align:right;">L:${pL}%${hasFlat ? ` F:${pF}%` : ""} S:${pS}%</div>
          </div>
          ${depthHtml}
        </div>`;
    }

    const content = `
      <div style="font-weight:700;margin-bottom:6px;">${row.date}</div>
      ${_horizonHtml("Short", row.short || {}, "≈4w (28d)")}
      ${_horizonHtml("Medium", row.medium || {}, "≈26w (182d)")}
      ${_horizonHtml("Long", row.long || {}, "≈52w (365d)")}
    `;
    pop.innerHTML = content;
    const rect = e.currentTarget.getBoundingClientRect();
    const left = Math.min(window.innerWidth - 280, rect.right + 8);
    const top = Math.max(8, rect.top + window.scrollY - 8);
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.display = "block";
  } catch (err) {
    console.error("Popover error", err);
  }
}


export function hidePredictionPopover() {
  const pop = document.getElementById("prediction-popover");
  if (pop) pop.style.display = "none";
}


export async function downloadFeatureImportances() {
  const ctx = analysisDrawerContext || {};
  if (!ctx.symbol || !ctx.model_type || !ctx.timestamp)
    return alert("No run selected");
  try {
    const interval = ctx.interval || "1d";
    const data = await api.featureImportances({
      symbol: ctx.symbol,
      model_type: ctx.model_type,
      timestamp: ctx.timestamp,
      interval,
    });
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${ctx.symbol}_${ctx.model_type}_${ctx.timestamp}_feature_importances.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    alert("Download failed");
  }
}
