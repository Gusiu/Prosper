/**
 * The client-side work queue and the runner log poller.
 *
 * Queue items are `{name, endpoint, body}`: the browser never builds CLI
 * strings, it posts typed JSON and the server decides what to run.
 */

import { api } from "./api.js";
import { escapeHtml, isValidSymbolPart } from "./format.js";
import { getResearchFlags } from "./research.js";
import { EVENTS, emit } from "./events.js";
import { state } from "./state.js";

let pollInterval = null;

/**
 * Render the queue from the server's snapshot.
 *
 * The browser used to keep its own array, so a page reload lost every pending
 * job while the running one carried on — the two views disagreed. The server
 * is now the only place the queue exists.
 */
export function renderQueue(snapshot) {
  const pending = snapshot?.pending ?? [];
  const current = snapshot?.current ?? null;
  const rows = current ? [{ ...current, running: true }, ...pending] : pending;

  for (const [tbodyId, countId] of [
    ["queue-tbody", "queue-count"],
    ["ai-queue-tbody", "ai-queue-count"],
  ]) {
    const tbody = document.getElementById(tbodyId);
    const count = document.getElementById(countId);
    if (count) count.innerText = `(${rows.length})`;
    if (!tbody) continue;

    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td class="text-muted">Queue is empty</td></tr>`;
      continue;
    }

    tbody.innerHTML = "";
    for (const item of rows) {
      const tr = document.createElement("tr");
      const controls = item.running
        ? `<span class="text-muted" style="font-size:0.7rem;">running</span>`
        : `
            <button class="action-btn" style="padding: 2px 4px; color: #2962ff;" data-action="queue-move" data-id="${item.id}" data-direction="-1">▲</button>
            <button class="action-btn" style="padding: 2px 4px; color: #2962ff;" data-action="queue-move" data-id="${item.id}" data-direction="1">▼</button>
            <button class="action-btn" style="padding: 2px 4px; color: var(--red-main);" data-action="queue-remove" data-id="${item.id}">X</button>`;
      tr.innerHTML = `
            <td style="font-size: 0.75rem; word-break: break-all;">${escapeHtml(item.name)}</td>
            <td style="width: 80px; text-align: right;">${controls}</td>
        `;
      tbody.appendChild(tr);
    }
  }
}

/** Backwards-compatible alias used while a caller still has no snapshot. */
export async function updateQueueUI() {
  try {
    renderQueue(await api.taskQueue());
  } catch (error) {
    console.error("queue refresh failed", error);
  }
}


export async function moveQueue(taskId, direction) {
  try {
    renderQueue(await api.moveTask(taskId, direction));
  } catch (error) {
    console.error(error);
  }
}


export async function removeQueue(taskId) {
  try {
    renderQueue(await api.cancelTask(taskId));
  } catch (error) {
    console.error(error);
  }
}


export function addToQueue() {
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
  // Month-precision copies, used only for the queue label. The server
  // truncates the full dates itself for the commands that need YYYY-MM.
  const s = document.getElementById("pipe-start").value.substring(0, 7);
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

  const aggNamePart = aggs.length > 0 ? ` + [${aggs.join(",")}]` : "";

  submitTask(
    `Pipeline [${symbol}] ${s} to ${e}${aggNamePart}`,
    "/api/pipeline/run",
    {
      symbol,
      start: sFull,
      end: eFull,
      intervals: aggs,
      flags: getResearchFlags(),
    },
  );

  document.getElementById("base-coin").value = "";
}


export function startQueueProcessing() {
  // The server drains its own queue, so "start" only has to make sure whatever
  // is in the form gets submitted.
  if (document.getElementById("base-coin").value.trim()) {
    addToQueue();
    return;
  }
  alert("Nothing to submit. Fill the form or use the 'Aggregate' panel.");
}

/**
 * Enqueue a task on the server and reflect the returned snapshot.
 *
 * Every caller goes through here, so there is one place that knows how a
 * submission is announced in the terminal and how failures surface.
 */
export async function submitTask(name, endpoint, body) {
  const terminal = document.getElementById("terminal-output");
  try {
    const snapshot = await api.submit(endpoint, body);
    renderQueue(snapshot);
    if (terminal) {
      terminal.innerHTML += `<div class="log-line text-muted mt-2" style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">--- QUEUED: ${escapeHtml(name)} ---</div>`;
      terminal.scrollTop = terminal.scrollHeight;
    }
    return snapshot;
  } catch (error) {
    console.error(error);
    alert(`Could not queue "${name}": ${error.message}`);
    return null;
  }
}


export function pollTaskStatus() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(checkTaskLogs, 1000);
}


export async function checkTaskLogs() {
  try {
    const data = await api.taskLogs(state.lastLogIdx);
    const terminal = document.getElementById("terminal-output");

    for (const log of data.logs) {
      const div = document.createElement("div");
      div.className = "log-line";
      div.textContent = log;
      terminal.appendChild(div);
    }
    if (data.logs.length) terminal.scrollTop = terminal.scrollHeight;

    // The server trims its buffer, so trust its absolute cursor rather than
    // counting locally.
    if (Number.isFinite(data.next_idx)) state.lastLogIdx = data.next_idx;

    renderQueue(data);
    if (data.progress?.visible) updateProgress(data.progress);

    const dot = document.getElementById("task-dot");
    const label = document.getElementById("task-status");
    const queued = data.pending?.length ?? 0;

    if (data.is_running) {
      dot.className = "status-dot active";
      label.innerText = `Task: RUNNING (Queue: ${queued})`;
      label.style.color = "var(--accent-glow)";
      state.isPipelineRunning = true;
    } else {
      dot.className = "status-dot";
      dot.style.background = "#8e9bb0";
      label.innerText = `Task: Idle (Queue: ${queued})`;
      label.style.color = "var(--text-muted)";

      if (state.isPipelineRunning) {
        // The worker just went idle: everything queued has drained.
        state.isPipelineRunning = false;
        updateProgress({ visible: true, percent: 100, label: "Complete" });
        setTimeout(() => {
          if (!state.isPipelineRunning) {
            const bar = document.getElementById("global-progress");
            if (bar) bar.style.display = "none";
          }
        }, 3000);
        emit(EVENTS.TASK_FINISHED);
      } else {
        const bar = document.getElementById("global-progress");
        if (bar) bar.style.display = "none";
      }
    }
  } catch {
    // The poller runs every second; a transient failure is not worth reporting.
  }
}


export function updateProgress(progress) {
  const progContainer = document.getElementById("global-progress");
  if (!progContainer) return;

  if (!progress || progress.visible === false) {
    progContainer.style.display = "none";
    return;
  }

  const progFill = document.getElementById("progress-bar-fill");
  const progText = document.getElementById("progress-text");
  const progPct = document.getElementById("progress-pct");
  const pct = Math.max(0, Math.min(100, Number(progress.percent || 0)));

  progContainer.style.display = "flex";
  if (progFill) progFill.style.width = `${pct}%`;
  if (progText) progText.innerText = progress.label || "Processing";
  if (progPct) progPct.innerText = `${pct.toFixed(1)}%`;
}

// --- DATA ANALYSIS DRAWER ---
