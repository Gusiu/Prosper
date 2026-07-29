/** Formatting and validation helpers shared across tabs. */

export function isValidSymbolPart(value) {
  return /^[A-Z0-9]{1,15}$/.test(value);
}

// --- RESEARCH MODE ---


export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


export function formatMetric(value, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "--";
}


export function formatPercentMetric(value, digits = 1) {
  const num = Number(value);
  return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : "--%";
}


export function formatNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return value.toFixed(2);
  if (Math.abs(value) >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}


export function animateValue(id, start, end, duration, prefix = "", suffix = "") {
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

// --- BACKTEST TAB ---
