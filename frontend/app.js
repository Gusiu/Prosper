let chartInstance = null;
let pollInterval = null;
let lastLogIdx = 0;
let pipelineQueue = [];
let isPipelineRunning = false;
let globalInventory = [];

document.addEventListener("DOMContentLoaded", () => {
    // Check API Status
    fetch("/api/status")
        .then(res => res.json())
        .then(data => {
            console.log("API Status:", data);
        })
        .catch(err => {
            document.getElementById("api-status").innerText = "Engine Offline";
            document.querySelector(".status-dot.online").style.backgroundColor = "var(--red-main)";
            document.querySelector(".status-dot.online").style.boxShadow = "0 0 10px var(--red-main)";
        });

    // Start background poll
    pollTaskStatus();
    
    // Load metadata and inventory
    initializeDates();
    loadInventory();
});

// --- TABS LOGIC ---
function switchTab(tabId, element) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    
    document.querySelectorAll('.tab-link').forEach(l => l.classList.remove('active'));
    element.classList.add('active');

    document.getElementById('topbar-title').innerText = element.innerText;
}

// --- DATES & INVENTORY ---
async function initializeDates() {
    try {
        const res = await fetch("/api/data/dates");
        const data = await res.json();
        
        document.getElementById("pipe-start").value = data.default_start_month;
        document.getElementById("pipe-end").value = data.yesterday_month;
        
        document.getElementById("train-start").value = data.default_start_month + "-01";
        document.getElementById("train-end").value = data.yesterday;
        
        document.getElementById("eval-start").value = "2024-01-01";
        document.getElementById("eval-end").value = data.yesterday;
    } catch(e) { console.error("Failed to fetch dates", e); }
}

async function loadInventory() {
    try {
        const res = await fetch("/api/data/inventory");
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

        globalInventory.forEach(item => {
            const tr = document.createElement("tr");
            
            // Format checkmarks for aggregations
            const aggs = ["1m", "1h", "1d", "1w"].map(a => 
                item.aggregations.includes(a) ? `<span style="color:var(--green-main)">✓ ${a}</span>` : `<span class="text-muted">✗ ${a}</span>`
            ).join(" | ");

            tr.innerHTML = `
                <td><strong>${item.symbol}</strong></td>
                <td><span class="text-muted">${item.start_date}</span> → <span class="text-muted">${item.end_date}</span></td>
                <td>${item.size_mb} MB</td>
                <td style="font-size:0.75rem;">${aggs}</td>
                <td>
                    <button class="action-btn" style="background: rgba(0, 229, 255, 0.1); color: var(--accent-glow); border-color: rgba(0, 229, 255, 0.2);" onclick="editSymbol('${item.symbol}')">Edit / Re-build</button>
                    <button class="action-btn" onclick="deleteSymbol('${item.symbol}')">Delete</button>
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

    } catch(e) { console.error("Failed to load inventory", e); }
}

function updateTrainDates() {
    const sym = document.getElementById("train-symbol").value;
    const item = globalInventory.find(i => i.symbol === sym);
    if (item && item.start_date !== "N/A") {
        document.getElementById("train-start").value = item.start_date;
        document.getElementById("train-end").value = item.end_date;
    }
}

function updateEvalDates() {
    const sym = document.getElementById("eval-symbol").value;
    const item = globalInventory.find(i => i.symbol === sym);
    if (item && item.start_date !== "N/A") {
        // usually eval is recent
        document.getElementById("eval-start").value = "2024-01-01";
        document.getElementById("eval-end").value = item.end_date;
    }
}

async function deleteSymbol(symbol) {
    if (!confirm(`Are you sure you want to delete all data for ${symbol}?`)) return;
    try {
        await fetch(`/api/data/${symbol}`, { method: 'DELETE' });
        loadInventory();
    } catch(e) { alert("Failed to delete"); }
}

function editSymbol(symbol) {
    // Fill the Data Manager input with the symbol to quickly re-run aggregations or features
    let base = symbol;
    let quote = "USDT";
    if (symbol.endsWith("USDT")) {
        base = symbol.replace("USDT", "");
    }
    document.getElementById("base-coin").value = base;
    document.getElementById("quote-coin").value = quote;
    window.scrollTo({top: 0, behavior: 'smooth'});
}

// --- QUEUE PIPELINE LOGIC ---
function fmtMonth(val, isEnd) {
    if (!val) return "";
    let base = val + "-01";
    if (isEnd) {
        let [y, m] = val.split("-");
        base = `${y}-${m}-28`;
    }
    return base;
}

function startDataPipeline() {
    const base = document.getElementById("base-coin").value.trim().toUpperCase();
    const quote = document.getElementById("quote-coin").value.trim().toUpperCase();
    if (!base || !quote) {
        alert("Please enter both Base and Quote coins.");
        return;
    }
    
    const symbol = base + quote;
    const s = document.getElementById("pipe-start").value;
    const e = document.getElementById("pipe-end").value;
    
    const aggs = [];
    if(document.getElementById("agg-1m").checked) aggs.push("1m");
    if(document.getElementById("agg-1h").checked) aggs.push("1h");
    if(document.getElementById("agg-1d").checked) aggs.push("1d");
    if(document.getElementById("agg-1w").checked) aggs.push("1w");
    
    const sFull = fmtMonth(s, false);
    const eFull = fmtMonth(e, true);
    
    let cmd = `poetry run prosper backfill --symbol ${symbol} --start ${s} --end ${e}`;
    
    // Auto add aggregate if checked anything other than 1m
    const targetAggs = aggs.filter(a => a !== "1m");
    if (targetAggs.length > 0) {
        cmd += ` && poetry run prosper aggregate --symbol ${symbol} --from 1m --to ${targetAggs.join(',')} --start ${sFull} --end ${eFull}`;
    }
    
    // Auto add features and labels
    cmd += ` && poetry run prosper features build --symbol ${symbol} --start ${sFull} --end ${eFull}`;
    cmd += ` && poetry run prosper labels build --symbol ${symbol}`;

    pipelineQueue.push(cmd);
    
    // Clear input for next
    document.getElementById("base-coin").value = "";
    
    processQueue();
}

async function processQueue() {
    if (isPipelineRunning || pipelineQueue.length === 0) return;
    
    const cmd = pipelineQueue.shift();
    isPipelineRunning = true;
    
    try {
        const res = await fetch("/api/task/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command: cmd })
        });
        
        if (!res.ok) {
            const data = await res.json();
            alert("Error: " + data.detail);
            isPipelineRunning = false;
        } else {
            lastLogIdx = 0;
            document.getElementById("terminal-output").innerHTML = "";
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
    if (model === 'ml') {
        cmd = `poetry run prosper predict ml --symbol ${symbol} --start ${s} --end ${e}`;
    } else {
        cmd = `poetry run prosper predict ${model} --symbol ${symbol} --start ${s} --end ${e} --max-epochs ${ep}`;
    }
    
    pipelineQueue.push(cmd);
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

        data.logs.forEach(log => {
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
            taskTxt.innerText = "Task: Idle";
            taskTxt.style.color = "var(--text-muted)";
            
            if (isPipelineRunning) {
                // Task just finished!
                isPipelineRunning = false;
                loadInventory(); // Refresh inventory after any task finishes
                processQueue();  // Process next in queue
            }
        }
    } catch (e) {
        // silent
    }
}

// --- CHART RENDERING ---
function animateValue(id, start, end, duration, prefix="", suffix="") {
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
    if(!symbol) return alert("Select symbol first");
    
    const start = document.getElementById("eval-start").value;
    const end = document.getElementById("eval-end").value;
    
    try {
        document.getElementById("terminal-output").innerHTML += `<div class="log-line text-muted">Fetching backtest details from API...</div>`;
        const res = await fetch(`/api/backtest/${symbol}?start=${start}&end=${end}`);
        
        if (!res.ok) {
            const err = await res.json();
            alert("Evaluating Error: " + err.detail);
            return;
        }

        const data = await res.json();
        
        animateValue("val-roi", 0, data.roi, 1000, "", "%");
        animateValue("val-capital", Math.max(0, data.final_capital - 2000), data.final_capital, 1000, "$", "");
        animateValue("val-drawdown", 0, data.max_drawdown, 1000, "", "%");
        
        document.getElementById("val-trades").innerText = data.total_trades;

        const roiEl = document.getElementById("val-roi");
        roiEl.className = "metric-value fade-in " + (data.roi >= 0 ? "text-green" : "text-red");

        drawChart(data.chart_data);
        document.getElementById("terminal-output").innerHTML += `<div class="log-line text-muted">Backtest complete. Chart rendered.</div>`;
        
    } catch (e) {
        alert("Failed to connect to Backtest API");
    }
}

function drawChart(chartData) {
    const ctx = document.getElementById('equityChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    Chart.defaults.color = "#8e9bb0";
    Chart.defaults.font.family = "Inter";

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: [
                {
                    label: 'Portfolio Equity ($)',
                    data: chartData.capital,
                    borderColor: '#00e5ff',
                    backgroundColor: 'rgba(0, 229, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.2,
                    yAxisID: 'y'
                },
                {
                    label: 'Asset Price ($)',
                    data: chartData.close,
                    borderColor: '#2962ff',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { color: '#f0f4f8' } },
                tooltip: { backgroundColor: 'rgba(20, 24, 39, 0.9)', titleColor: '#00e5ff', bodyColor: '#f0f4f8', borderColor: 'rgba(255, 255, 255, 0.1)', borderWidth: 1, padding: 12 }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { maxTicksLimit: 10 } },
                y: { type: 'linear', display: true, position: 'left', grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
            }
        }
    });
}
