/**
 * MANA — Charts Module
 * All chart/visualization rendering: histogram, line chart, bar chart, donut.
 * Also owns analytics mock data and the analytics DataService shim.
 *
 * API endpoints (backend must implement):
 *   GET /api/analytics/sentiment-histogram   ?date_range=14d → { histogram: [{label,value,tone}] }
 *   GET /api/analytics/sentiment-trend       ?date_range=14d → { labels, positive, neutral, negative }
 *   GET /api/analytics/cluster-activity      ?date_range=14d → { clusterActivity: [{label,value,color}] }
 *   GET /api/analytics/priority-distribution ?date_range=14d → { priority: [{label,value,color}] }
 */

// ─── Mock Data ────────────────────────────────────────────────────────────────
const MOCK_ANALYTICS = {
  "7d": {
    histogram:       [{ label:"0-20",  value:26,  tone:"negative" },{ label:"21-40", value:48,  tone:"negative" },{ label:"41-60", value:72,  tone:"neutral"  },{ label:"61-80", value:61,  tone:"positive" },{ label:"81-100",value:33,  tone:"positive" }],
    trend:           { labels:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], positive:[18,20,19,24,26,28,30], neutral:[34,31,36,33,37,35,39], negative:[22,26,29,35,41,47,53] },
    clusterActivity: [{ label:"Food and NFIs",value:120,color:"#f59e0b" },{ label:"Health",    value:85, color:"#3b82f6" },{ label:"CCCM",     value:64, color:"#8b5cf6" },{ label:"Logistics",value:74, color:"#f97316" },{ label:"ETC",      value:42, color:"#06b6d4" },{ label:"Education",value:33, color:"#10b981" },{ label:"SRR",      value:96, color:"#ef4444" },{ label:"MDM",      value:28, color:"#64748b" }],
    priority:        [{ label:"Low",value:44,color:"#34d399" },{ label:"Medium",value:71,color:"#38bdf8" },{ label:"High",value:94,color:"#f59e0b" }],
  },
  "14d": {
    histogram:       [{ label:"0-20",  value:52,  tone:"negative" },{ label:"21-40", value:96,  tone:"negative" },{ label:"41-60", value:128, tone:"neutral"  },{ label:"61-80", value:116, tone:"positive" },{ label:"81-100",value:68,  tone:"positive" }],
    trend:           { labels:["W1","W2","W3","W4","W5","W6","W7"], positive:[28,30,29,34,38,40,43], neutral:[49,52,50,54,56,58,61], negative:[34,39,43,52,61,70,78] },
    clusterActivity: [{ label:"Food and NFIs",value:226,color:"#f59e0b" },{ label:"Health",    value:181,color:"#3b82f6" },{ label:"CCCM",     value:128,color:"#8b5cf6" },{ label:"Logistics",value:149,color:"#f97316" },{ label:"ETC",      value:86, color:"#06b6d4" },{ label:"Education",value:67, color:"#10b981" },{ label:"SRR",      value:194,color:"#ef4444" },{ label:"MDM",      value:54, color:"#64748b" }],
    priority:        [{ label:"Low",value:88,color:"#34d399" },{ label:"Medium",value:132,color:"#38bdf8" },{ label:"High",value:188,color:"#f59e0b" }],
  },
  "30d": {
    histogram:       [{ label:"0-20",  value:110, tone:"negative" },{ label:"21-40", value:184, tone:"negative" },{ label:"41-60", value:248, tone:"neutral"  },{ label:"61-80", value:222, tone:"positive" },{ label:"81-100",value:144, tone:"positive" }],
    trend:           { labels:["P1","P2","P3","P4","P5","P6","P7"], positive:[51,48,52,56,61,67,72], neutral:[82,86,84,90,94,101,110], negative:[60,69,78,88,103,118,132] },
    clusterActivity: [{ label:"Food and NFIs",value:428,color:"#f59e0b" },{ label:"Health",    value:355,color:"#3b82f6" },{ label:"CCCM",     value:246,color:"#8b5cf6" },{ label:"Logistics",value:271,color:"#f97316" },{ label:"ETC",      value:163,color:"#06b6d4" },{ label:"Education",value:122,color:"#10b981" },{ label:"SRR",      value:377,color:"#ef4444" },{ label:"MDM",      value:107,color:"#64748b" }],
    priority:        [{ label:"Low",value:170,color:"#34d399" },{ label:"Medium",value:261,color:"#38bdf8" },{ label:"High",value:336,color:"#f59e0b" }],
  },
};

// ─── API Calls ────────────────────────────────────────────────────────────────
async function apiGetSentimentHistogram(r)   { return apiFetch(`/analytics/sentiment-histogram?date_range=${r}`); }
async function apiGetSentimentTrend(r)       { return apiFetch(`/analytics/sentiment-trend?date_range=${r}`); }
async function apiGetClusterActivity(r)      { return apiFetch(`/analytics/cluster-activity?date_range=${r}`); }
async function apiGetPriorityDistribution(r) { return apiFetch(`/analytics/priority-distribution?date_range=${r}`); }

// ─── DataService shim ─────────────────────────────────────────────────────────
const ChartsService = {
  async getAnalytics(dateRange) {
    if (USE_MOCK) return MOCK_ANALYTICS[dateRange] || MOCK_ANALYTICS["30d"];
    const [h, t, c, p] = await Promise.all([
      apiGetSentimentHistogram(dateRange),
      apiGetSentimentTrend(dateRange),
      apiGetClusterActivity(dateRange),
      apiGetPriorityDistribution(dateRange),
    ]);
    return { histogram: h.histogram, trend: t, clusterActivity: c.clusterActivity, priority: p.priority };
  },
};

// ─── Render: Full Analytics Page ──────────────────────────────────────────────
function renderAnalytics() {
  if (!state.analytics?.histogram?.length) {
    const loadingMarkup = `<div class="watch-empty"><strong>${state.loading.analytics ? "Loading analytics..." : "Analytics will load when this page is opened."}</strong></div>`;
    document.getElementById("sentimentHistogram").innerHTML = loadingMarkup;
    document.getElementById("sentimentTrendLegend").innerHTML = "";
    document.getElementById("sentimentTrendChart").innerHTML = loadingMarkup;
    document.getElementById("clusterActivityChart").innerHTML = loadingMarkup;
    document.getElementById("priorityDistributionChart").innerHTML = loadingMarkup;
    return;
  }

  const analytics = state.analytics;

  renderSentimentHistogram(analytics.histogram);
  renderSentimentTrend(analytics.trend);
  renderClusterActivityBars(analytics.clusterActivity);
  renderPriorityDonut(analytics.priority);
}

// ─── Sentiment Histogram ──────────────────────────────────────────────────────
function renderSentimentHistogram(data) {
  const maxVal = Math.max(...data.map(i => i.value));
  document.getElementById("sentimentHistogram").innerHTML = data.map(item => `
    <div class="hist-bar">
      <div class="hist-bar-value">${item.value}</div>
      <div class="hist-bar-track">
        <div class="hist-bar-fill" style="height:${Math.max(18, (item.value / maxVal) * 100)}%; background:${sentimentHistogramColor(item.tone)};"></div>
      </div>
      <div class="hist-bar-label">${item.label}</div>
    </div>
  `).join("");
}

function sentimentHistogramColor(tone) {
  if (tone === "negative") return "linear-gradient(180deg, #fb7185, rgba(239,68,68,0.62))";
  if (tone === "neutral")  return "linear-gradient(180deg, #f59e0b, rgba(217,119,6,0.62))";
  return "linear-gradient(180deg, #34d399, rgba(5,150,105,0.62))";
}

// ─── Sentiment Trend Line Chart ───────────────────────────────────────────────
function renderSentimentTrend(trend) {
  document.getElementById("sentimentTrendLegend").innerHTML = `
    <span><i class="legend-dot" style="background:#10b981;"></i>Positive</span>
    <span><i class="legend-dot" style="background:#f59e0b;"></i>Neutral</span>
    <span><i class="legend-dot" style="background:#ef4444;"></i>Negative</span>`;

  document.getElementById("sentimentTrendChart").innerHTML = renderMultiLineChart(
    trend.labels,
    [
      { data: trend.positive, color: "#10b981" },
      { data: trend.neutral,  color: "#f59e0b" },
      { data: trend.negative, color: "#ef4444" },
    ]
  );
}

function renderMultiLineChart(labels, seriesArray) {
  const width = 440, height = 120, pad = 24;
  const stepX = (width - pad * 2) / (labels.length - 1);
  const allVals = seriesArray.flatMap(s => s.data);
  const maxVal = Math.max(...allVals), minVal = Math.min(...allVals);
  const range  = maxVal - minVal || 1;

  const paths = seriesArray.map(s => {
    const points = s.data.map((v, i) => {
      const x = pad + i * stepX;
      const y = height - pad - ((v - minVal) / range) * (height - pad * 2);
      return `${x},${y}`;
    }).join(" ");
    return `<polyline fill="none" stroke="${s.color}" stroke-width="3" points="${points}"></polyline>`;
  }).join("");

  const labelsMarkup = labels.map((l, i) =>
    `<text x="${pad + i * stepX}" y="${height - 2}" text-anchor="middle" fill="currentColor" font-size="10">${l}</text>`
  ).join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <g opacity="0.18" stroke="currentColor">
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
      </g>
      ${paths}
      <g opacity="0.62">${labelsMarkup}</g>
    </svg>`;
}

// ─── Cluster Activity Bar Chart ───────────────────────────────────────────────
function renderClusterActivityBars(data) {
  const maxVal = Math.max(...data.map(i => i.value));
  document.getElementById("clusterActivityChart").innerHTML = data.map(item => `
    <div class="cluster-bar-row">
      <div class="cluster-bar-head"><span>${item.label}</span><strong>${item.value} posts</strong></div>
      <div class="cluster-bar-track">
        <div class="cluster-bar-fill" style="width:${(item.value / maxVal) * 100}%; background:${item.color};"></div>
      </div>
    </div>
  `).join("");
}

// ─── Priority Donut Chart ─────────────────────────────────────────────────────
function renderPriorityDonut(items) {
  const total = items.reduce((sum, i) => sum + i.value, 0);
  let start   = 0;
  const stops = items.map(item => {
    const end  = start + (item.value / total) * 360;
    const stop = `${item.color} ${start}deg ${end}deg`;
    start = end;
    return stop;
  }).join(", ");

  document.getElementById("priorityDistributionChart").innerHTML = `
    <div class="donut-layout">
      <div class="donut-ring" style="background: conic-gradient(${stops});">
        <div class="donut-center"><div><strong>${total}</strong>Total posts</div></div>
      </div>
      <div class="chart-legend">
        ${items.map(i => `<span><i class="legend-dot" style="background:${i.color};"></i>${i.label}: ${i.value}</span>`).join("")}
      </div>
    </div>`;
}
