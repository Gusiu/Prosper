/** The historical inspector drawer: candles, indicators, gaps. */

import { api } from "./api.js";
import { formatNumber } from "./format.js";
import { ANALYSIS_INTERVALS, state } from "./state.js";

const analysisState = {
  symbol: null,
  interval: null,
  data: null,
  isSyncingRange: false,
  isSyncingCrosshair: false,
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

export function initAnalysisDrawer() {
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
  if (intervalSelect)
    intervalSelect.addEventListener("change", () => {
      // Re-scale the default window: 1000 bars of 1m is days, of 1d is years.
      const item = state.globalInventory.find((i) => i.symbol === analysisState.symbol);
      const endInput = document.getElementById("analysis-end");
      const startInput = document.getElementById("analysis-start");
      if (item && endInput && startInput) {
        startInput.value = defaultAnalysisStart(item, endInput.value);
      }
      loadAnalysisData();
    });
}


export function openAnalysisDrawer(symbol) {
  const item = state.globalInventory.find((i) => i.symbol === symbol);
  if (!item) return alert("Symbol not found in inventory");

  analysisState.symbol = symbol;
  const drawer = document.getElementById("analysis-drawer");
  const backdrop = document.getElementById("analysis-backdrop");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  backdrop.classList.add("open");

  document.getElementById("analysis-title").innerText = symbol;
  document.getElementById("analysis-subtitle").innerText =
    `${item.start_date} to ${item.end_date}`;

  // Ensure we show the proper elements (might be hidden by prediction calendar)
  const chartContainer = document.getElementById("chart-container");
  if (chartContainer) chartContainer.style.display = "";
  const predictionContainer = document.getElementById(
    "prediction-calendar-container",
  );
  if (predictionContainer) predictionContainer.style.display = "none";
  const subcharts = document.querySelector(".analysis-subcharts");
  if (subcharts) subcharts.style.display = "";
  const toolbar = document.querySelector(".analysis-toolbar");
  if (toolbar) toolbar.style.display = "";
  const layers = document.querySelector(".analysis-layers");
  if (layers) layers.style.display = "";
  // Default to a recent window rather than the full history. The API returns
  // the tail of the range anyway, so spanning nine years only meant scanning
  // every partition to throw almost all of it away.
  const endValue = item.end_date !== "N/A" ? item.end_date : "";
  document.getElementById("analysis-end").value = endValue;
  document.getElementById("analysis-start").value = defaultAnalysisStart(
    item,
    endValue,
  );
  document.querySelectorAll(".indicator-checkbox").forEach((cb) => {
    cb.checked = false;
  });
  document.getElementById("show-rsi").checked = true;
  document.getElementById("show-macd").checked = true;
  document.getElementById("show-atr").checked = true;
  document.getElementById("analysis-show-volume").checked = true;
  document.getElementById("analysis-show-labels").checked = false;
  document.getElementById("analysis-show-gaps").checked = true;
  populateAnalysisIntervals(item);
  loadAnalysisData();
}

// Roughly how much history the default candle limit covers, per interval.


const ANALYSIS_DEFAULT_SPAN_DAYS = {
  "1m": 3,
  "3m": 7,
  "5m": 10,
  "15m": 30,
  "30m": 60,
  "1h": 120,
  "2h": 180,
  "4h": 365,
  "6h": 400,
  "8h": 500,
  "12h": 700,
  "1d": 1000,
  "3d": 1500,
  "1w": 3000,
};


export function defaultAnalysisStart(item, endValue) {
  if (!endValue || item.start_date === "N/A") return "";
  const interval = document.getElementById("analysis-interval")?.value || "1d";
  const span = ANALYSIS_DEFAULT_SPAN_DAYS[interval] ?? 365;
  const end = new Date(endValue);
  if (Number.isNaN(end.getTime())) return item.start_date;

  const start = new Date(end);
  start.setDate(start.getDate() - span);
  const earliest = new Date(item.start_date);
  if (!Number.isNaN(earliest.getTime()) && start < earliest) {
    return item.start_date;
  }
  return start.toISOString().slice(0, 10);
}


export function closeAnalysisDrawer() {
  document.getElementById("analysis-drawer").classList.remove("open");
  document
    .getElementById("analysis-drawer")
    .setAttribute("aria-hidden", "true");
  document.getElementById("analysis-backdrop").classList.remove("open");
  disposeAnalysisCharts();
  analysisState.symbol = null;
  analysisState.data = null;

  // Clear any inline styles that might have been set by prediction calendar
  const chartContainer = document.getElementById("chart-container");
  if (chartContainer) chartContainer.style.display = "";
  const subcharts = document.querySelector(".analysis-subcharts");
  if (subcharts) subcharts.style.display = "";
  const toolbar = document.querySelector(".analysis-toolbar");
  if (toolbar) toolbar.style.display = "";
  const layers = document.querySelector(".analysis-layers");
  if (layers) layers.style.display = "";
}


export function populateAnalysisIntervals(item) {
  const select = document.getElementById("analysis-interval");
  select.innerHTML = "";
  const intervals = (item.aggregations || [])
    .map((agg) => agg.interval || agg)
    .filter(Boolean)
    .sort(compareAnalysisIntervals);

  const uniqueIntervals =
    intervals.length > 0 ? [...new Set(intervals)] : ["1m"];
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


export async function loadAnalysisData() {
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
    analysisState.data = await api.klines(
      analysisState.symbol,
      interval,
      params,
    );
    status.innerText = `${analysisState.data.rows} candles loaded, ${analysisState.data.gaps.length} gap(s) detected.`;
    renderAnalysisCharts();
  } catch (e) {
    disposeAnalysisCharts();
    status.innerText = e.message || "Failed to load chart data.";
  }
}


export function renderAnalysisCharts() {
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
  analysisState.rowsByTime = new Map(
    candles.map((row) => [String(row.time), row]),
  );

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


export function drawVolume(candles) {
  analysisState.volumeSeries = analysisState.chart.addHistogramSeries({
    priceFormat: { type: "volume" },
    priceScaleId: "",
  });
  analysisState.volumeSeries
    .priceScale()
    .applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
  analysisState.volumeSeries.setData(
    candles.map((row) => ({
      time: row.time,
      value: row.volume,
      color:
        row.close >= row.open ? "rgba(0,230,118,0.28)" : "rgba(255,23,68,0.28)",
    })),
  );
}


export function overlayIndicators(candles, options) {
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
  if (
    options.showBb &&
    bbUpperData.length > 0 &&
    bbMidData.length > 0 &&
    bbLowerData.length > 0
  ) {
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


export function drawLabels(candles) {
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


export function drawDataHealth() {
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


export function drawRsiChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">Indicators disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.rsi_14))
    .map((row) => ({ time: row.time, value: row.rsi_14 }));
  if (data.length === 0) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">No RSI data</div>';
    return;
  }

  analysisState.rsiChart = LightweightCharts.createChart(
    container,
    buildSubChartOptions(container),
  );
  analysisState.rsiSeries = analysisState.rsiChart.addLineSeries({
    color: "#7dd3fc",
    lineWidth: 2,
  });
  analysisState.rsiSeries.setData(data);
  analysisState.rsiSeries.createPriceLine({
    price: 70,
    color: "rgba(255,23,68,0.55)",
    lineStyle: LightweightCharts.LineStyle.Dashed,
  });
  analysisState.rsiSeries.createPriceLine({
    price: 30,
    color: "rgba(0,230,118,0.55)",
    lineStyle: LightweightCharts.LineStyle.Dashed,
  });
  analysisState.rsiChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.rsiChart);
}


export function drawMacdChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">Indicators disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.macd_hist))
    .map((row) => ({
      time: row.time,
      value: row.macd_hist,
      color:
        row.macd_hist >= 0 ? "rgba(0,230,118,0.65)" : "rgba(255,23,68,0.65)",
    }));
  if (data.length === 0) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">No MACD data</div>';
    return;
  }

  analysisState.macdChart = LightweightCharts.createChart(
    container,
    buildSubChartOptions(container),
  );
  analysisState.macdSeries = analysisState.macdChart.addHistogramSeries({
    priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
  });
  analysisState.macdSeries.setData(data);
  analysisState.macdChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.macdChart);
}


export function drawAtrChart(candles, container, visible) {
  container.innerHTML = "";
  if (!visible) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">Indicator disabled</div>';
    return;
  }

  const data = candles
    .filter((row) => Number.isFinite(row.atr_14))
    .map((row) => ({ time: row.time, value: row.atr_14 }));
  if (data.length === 0) {
    container.innerHTML =
      '<div class="analysis-empty-subchart">No ATR data</div>';
    return;
  }

  analysisState.atrChart = LightweightCharts.createChart(
    container,
    buildSubChartOptions(container),
  );
  analysisState.atrSeries = analysisState.atrChart.addLineSeries({
    color: "#fb7185",
    lineWidth: 2,
  });
  analysisState.atrSeries.setData(data);
  analysisState.atrChart.timeScale().fitContent();
  attachSyncHandlers(analysisState.atrChart);
}


export function buildSubChartOptions(container) {
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


export function attachSyncHandlers(chart) {
  if (!chart) return;

  chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
    if (!range || analysisState.isSyncingRange) return;
    analysisState.isSyncingRange = true;
    [
      analysisState.chart,
      analysisState.rsiChart,
      analysisState.macdChart,
      analysisState.atrChart,
    ].forEach((c) => {
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
      [
        analysisState.chart,
        analysisState.rsiChart,
        analysisState.macdChart,
        analysisState.atrChart,
      ].forEach((c) => {
        if (c && c !== chart && typeof c.clearCrosshairPosition === "function")
          c.clearCrosshairPosition();
      });
    } else {
      const row = analysisState.rowsByTime.get(String(param.time));
      if (row) {
        if (analysisState.chart && chart !== analysisState.chart)
          setSubchartCrosshair(
            analysisState.chart,
            analysisState.candleSeries,
            row.close,
            param.time,
          );
        if (analysisState.rsiChart && chart !== analysisState.rsiChart)
          setSubchartCrosshair(
            analysisState.rsiChart,
            analysisState.rsiSeries,
            row.rsi_14,
            param.time,
          );
        if (analysisState.macdChart && chart !== analysisState.macdChart)
          setSubchartCrosshair(
            analysisState.macdChart,
            analysisState.macdSeries,
            row.macd_hist,
            param.time,
          );
        if (analysisState.atrChart && chart !== analysisState.atrChart)
          setSubchartCrosshair(
            analysisState.atrChart,
            analysisState.atrSeries,
            row.atr_14,
            param.time,
          );
      }
    }
    analysisState.isSyncingCrosshair = false;
  });
}


export function setSubchartCrosshair(chart, series, value, time) {
  if (
    !chart ||
    !series ||
    !Number.isFinite(value) ||
    typeof chart.setCrosshairPosition !== "function"
  ) {
    return;
  }
  try {
    chart.setCrosshairPosition(value, time, series);
  } catch (e) {
    // Older CDN builds differ slightly; keep legend updates even if sync is unavailable.
  }
}


export function updateAnalysisLegend(param) {
  const legend = document.getElementById("chart-legend");
  if (!param || !param.time) {
    legend.innerText = "Hover over the chart to inspect values.";
    return;
  }

  const row = analysisState.rowsByTime.get(String(param.time));
  if (!row) return;

  const date = new Date(row.time * 1000)
    .toISOString()
    .replace("T", " ")
    .slice(0, 16);
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
  if (Number.isFinite(row.bb_upper))
    parts.push(`BBU ${formatNumber(row.bb_upper)}`);
  if (Number.isFinite(row.bb_lower))
    parts.push(`BBL ${formatNumber(row.bb_lower)}`);
  if (Number.isFinite(row.rsi_14))
    parts.push(`RSI ${formatNumber(row.rsi_14)}`);
  if (Number.isFinite(row.macd_hist))
    parts.push(`MACD ${formatNumber(row.macd_hist)}`);
  if (Number.isFinite(row.atr_14))
    parts.push(`ATR ${formatNumber(row.atr_14)}`);
  if (Number.isFinite(row.obv)) parts.push(`OBV ${formatNumber(row.obv)}`);
  if (row.direction) parts.push(`Label ${row.direction}`);
  legend.innerText = parts.join(" | ");
}


export function compareAnalysisIntervals(a, b) {
  const aIdx = ANALYSIS_INTERVALS.indexOf(a);
  const bIdx = ANALYSIS_INTERVALS.indexOf(b);
  const safeA = aIdx === -1 ? ANALYSIS_INTERVALS.length : aIdx;
  const safeB = bIdx === -1 ? ANALYSIS_INTERVALS.length : bIdx;
  return safeA - safeB || String(a).localeCompare(String(b));
}


export function resizeAnalysisCharts() {
  const container = document.getElementById("chart-container");
  const rsiContainer = document.getElementById("analysis-rsi-chart");
  const macdContainer = document.getElementById("analysis-macd-chart");
  const atrContainer = document.getElementById("analysis-atr-chart");
  if (analysisState.chart) {
    analysisState.chart.applyOptions({ width: container.clientWidth });
    drawDataHealth();
  }
  if (analysisState.rsiChart)
    analysisState.rsiChart.applyOptions({ width: rsiContainer.clientWidth });
  if (analysisState.macdChart)
    analysisState.macdChart.applyOptions({ width: macdContainer.clientWidth });
  if (analysisState.atrChart)
    analysisState.atrChart.applyOptions({ width: atrContainer.clientWidth });
}


export function disposeAnalysisCharts() {
  if (analysisState.resizeHandler) {
    window.removeEventListener("resize", analysisState.resizeHandler);
    analysisState.resizeHandler = null;
  }
  if (analysisState.chart && analysisState.mainRangeHandler) {
    analysisState.chart
      .timeScale()
      .unsubscribeVisibleTimeRangeChange(analysisState.mainRangeHandler);
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


export function exportAnalysisPng() {
  if (!analysisState.chart) return alert("Load chart data first.");
  const canvas = analysisState.chart.takeScreenshot();
  const link = document.createElement("a");
  const interval = analysisState.interval || "chart";
  link.download = `${analysisState.symbol}-${interval}-analysis.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

// --- CHART RENDERING ---
