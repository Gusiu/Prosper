/**
 * Every HTTP call the dashboard makes.
 *
 * Endpoint paths used to be string literals scattered across ~2900 lines, so a
 * renamed route meant hunting through the UI. Keeping them here also makes the
 * server's surface readable in one screen.
 */

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || response.statusText);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function postJson(url, body) {
  return request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const q = encodeURIComponent;

export const api = {
  status: () => request("/api/status"),

  // --- data lake ---
  dates: () => request("/api/data/dates"),
  inventory: (refresh = false) =>
    request(`/api/data/inventory${refresh ? "?refresh=true" : ""}`),
  deleteSymbol: (symbol) =>
    request(`/api/data/${q(symbol)}`, { method: "DELETE" }),
  klines: (symbol, interval, params) =>
    request(`/api/data/klines/${q(symbol)}/${q(interval)}?${params}`),
  symbolMetadata: () => request("/api/meta/symbols"),

  // --- background work ---
  taskLogs: (startIdx) => request(`/api/task/logs?start_idx=${startIdx}`),
  taskQueue: () => request("/api/task/queue"),
  cancelTask: (id) => request(`/api/task/queue/${id}`, { method: "DELETE" }),
  moveTask: (id, direction) =>
    postJson(`/api/task/queue/${id}/move?direction=${direction}`, {}),
  submit: (endpoint, body) => postJson(endpoint, body),

  // --- prediction runs ---
  models: () => request("/api/ai/models"),
  deleteRun: (symbol, modelType, timestamp, interval) =>
    request(
      `/api/ai/models/${q(symbol)}/${q(modelType)}/${q(timestamp)}?interval=${q(interval)}`,
      { method: "DELETE" },
    ),
  predictions: ({ symbol, model_type, timestamp, interval, year, month }) => {
    const params = new URLSearchParams({ symbol, model_type, timestamp, interval });
    if (year != null && month != null) {
      params.set("year", String(year));
      params.set("month", String(month));
    }
    return request(`/api/ai/predictions?${params}`);
  },
  trainingDefaults: () => request(`/api/meta/training-defaults`),
  predictionMonths: ({ symbol, model_type, timestamp, interval }) =>
    request(
      `/api/ai/predictions/months?symbol=${q(symbol)}&model_type=${q(model_type)}` +
        `&timestamp=${q(timestamp)}&interval=${q(interval)}`,
    ),
  featureImportances: ({ symbol, model_type, timestamp, interval }) =>
    request(
      `/api/ai/feature_importances?symbol=${q(symbol)}&model_type=${q(model_type)}` +
        `&timestamp=${q(timestamp)}&interval=${q(interval)}`,
    ),

  // --- evaluation ---
  evaluations: () => request("/api/ai/evaluations"),
  evaluationDetail: ({ symbol, model_type, timestamp, interval, limit, flag }) => {
    const params = new URLSearchParams({ interval, limit: String(limit) });
    if (flag) params.set("flag", flag);
    return request(
      `/api/ai/evaluations/${q(symbol)}/${q(model_type)}/${q(timestamp)}?${params}`,
    );
  },
  evaluationFlags: ({ symbol, model_type, timestamp, interval }) =>
    request(
      `/api/ai/evaluations/${q(symbol)}/${q(model_type)}/${q(timestamp)}/flags?interval=${q(interval)}`,
    ),

  // --- simulation ---
  backtest: (symbol, params) => request(`/api/backtest/${q(symbol)}?${params}`),
};
