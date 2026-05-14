/**
 * MANA — Utilities Module
 * Pure helper functions shared across all other modules.
 * No DOM side-effects, no API calls, no state mutations.
 */

// ─── Number & Date Formatting ─────────────────────────────────────────────────
function formatNumber(n)  { return new Intl.NumberFormat("en-US").format(n); }
function formatCompact(n) { return new Intl.NumberFormat("en-US", { notation:"compact", maximumFractionDigits:1 }).format(n); }
function formatDate(d)    { return new Date(d).toLocaleString("en-US", { month:"short", day:"numeric", hour:"numeric", minute:"2-digit" }); }
function toCount(value)   { return Number.isFinite(Number(value)) ? Number(value) : 0; }

function timeAgo(date) {
  const diff = Date.now() - new Date(date).getTime();
  const h = Math.floor(diff / 3600000);
  const d = Math.floor(diff / 86400000);
  if (h < 1) return "just now";
  if (h < 24) return `${h}h ago`;
  if (d === 1) return "1d ago";
  return `${d}d ago`;
}

function getInitials(name) {
  return name.replace(/^@/, "").split(/\s+/).slice(0, 3).map(w => w[0] || "").join("").toUpperCase().slice(0, 3);
}

function anonymizedCommentAuthor(comment = {}) {
  const seed = [
    comment.author,
    comment.id,
    comment.text,
    comment.date,
  ].filter(Boolean).join("|") || "comment";
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
  }
  const suffix = String(Math.abs(hash) % 9000 + 1000);
  return `facebookuser@${suffix}`;
}

// ─── Post Engagement ──────────────────────────────────────────────────────────
function getEngagement(post) {
  return post.source === "Facebook"
    ? (post.reactions || 0) + (post.shares || 0)  + (post.comments || 0)
    : (post.likes     || 0) + (post.reposts  || 0) + (post.comments || 0);
}

// ─── Post Filtering & Sorting ─────────────────────────────────────────────────
function matchesDateRange(postDate, range) {
  const diffDays = (Date.now() - new Date(postDate).getTime()) / (1000 * 60 * 60 * 24);
  if (range === "24h") return diffDays <= 1;
  if (range === "3d")  return diffDays <= 3;
  if (range === "7d")  return diffDays <= 7;
  if (range === "14d") return diffDays <= 14;
  if (range === "30d") return diffDays <= 30;
  return true;
}

function normalizeSearchTerm(value) {
  return String(value || "").trim().toLowerCase();
}

function getSearchTokens(value) {
  return normalizeSearchTerm(value).split(/\s+/).filter(Boolean);
}

function matchesPostSearch(post, searchTerm) {
  const tokens = getSearchTokens(searchTerm);
  if (!tokens.length) return true;

  const cluster = (state?.clusters || []).find(c => c.id === post.clusterId);
  const haystack = [
    post.caption,
    post.location,
    post.pageSource,
    post.author,
    post.source,
    post.recommendation,
    ...(Array.isArray(post.keywords) ? post.keywords : []),
    cluster?.short,
    cluster?.name,
    cluster?.description,
    ...(Array.isArray(cluster?.keywords) ? cluster.keywords : []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return tokens.every(token => haystack.includes(token));
}

function getPostStatus(post) {
  return state?.statuses?.[post.id] || post.status || "Monitoring";
}

function isResolvedStatus(status) {
  return String(status || "").toLowerCase() === "resolved";
}

function isResolvedPost(post) {
  return isResolvedStatus(getPostStatus(post));
}

function filterPosts(sourcePosts, dateRange, source, searchTerm = "", options = {}) {
  const { includeResolved = false } = options;
  return sourcePosts
    .filter(p => includeResolved || !isResolvedPost(p))
    .filter(p => matchesDateRange(p.date, dateRange))
    .filter(p => source === "All" ? true : p.source === source)
    .filter(p => matchesPostSearch(p, searchTerm));
}

function filterKeywords(sourceKeywords, searchTerm = "") {
  const tokens = getSearchTokens(searchTerm);
  if (!tokens.length) return sourceKeywords;

  return (sourceKeywords || []).filter(item => {
    const haystack = [item.keyword, item.note].filter(Boolean).join(" ").toLowerCase();
    return tokens.every(token => haystack.includes(token));
  });
}

function sortPostsByPriority(a, b) {
  if (b.severityRank !== a.severityRank) return b.severityRank - a.severityRank;
  return getEngagement(b) - getEngagement(a);
}

// ─── Sentiment ────────────────────────────────────────────────────────────────
// Matches original exactly: 80+ = Negative, 60+ = Neutral, else Positive
function getDominantSentiment(score) {
  if (score >= 80) return { label: "Negative", percent: score, tone: "negative" };
  if (score >= 60) return { label: "Neutral",  percent: score, tone: "neutral"  };
  return               { label: "Positive", percent: score, tone: "positive" };
}

// ─── CSS Class Helpers ────────────────────────────────────────────────────────
function normalizePriority(priority) {
  if (priority === "Critical")   return "High";
  if (priority === "Moderate")   return "Medium";
  if (priority === "Monitoring") return "Low";
  return priority;
}

function priorityClass(priority) {
  const p = normalizePriority(priority);
  if (p === "High")   return "priority-high";
  if (p === "Medium") return "priority-medium";
  return "priority-low";
}

function priorityPercent(post) {
  const base = { 4: 88, 3: 70, 2: 50 }[post.severityRank] || 38;
  const adj  = Math.round(((post.sentimentScore || 50) / 100) * 12) - 6;
  return Math.min(99, Math.max(20, base + adj));
}

function statusClass(status) { return `status-${status.toLowerCase()}`; }

function kpiToneClass(label) {
  if (label.includes("High Priority"))       return "kpi-gold";
  if (label.includes("Total Posts Analyzed"))return "kpi-blue";
  if (label.includes("Facebook"))            return "kpi-cyan";
  if (label.includes("X/Twitter"))           return "kpi-slate";
  return "kpi-green";
}

function applyStatusStyle(select, status) {
  // ensure the element always has status-select base class (original CSS targets .status-select.status-*)
  select.classList.add("status-select");
  select.classList.remove("status-resolved","status-ongoing","status-monitoring","status-unresolved");
  select.classList.add(statusClass(status));
}

function applySeverityStyle(select, severity) {
  select.classList.remove("severity-high","severity-medium","severity-low");
  if (severity === "High")   select.classList.add("severity-high");
  if (severity === "Medium") select.classList.add("severity-medium");
  if (severity === "Low")    select.classList.add("severity-low");
}

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(title, message) {
  const wrap  = document.getElementById("toastWrap");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<strong>${title}</strong><p>${message}</p>`;
  wrap.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById("topbarClock").textContent = "Updated 26 Apr 2026";
}
