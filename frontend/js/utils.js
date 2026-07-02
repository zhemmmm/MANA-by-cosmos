/**
 * MANA — Utilities Module
 * Pure helper functions shared across all other modules.
 * No DOM side-effects, no API calls, no state mutations.
 */

// ─── Number & Date Formatting ─────────────────────────────────────────────────
function formatNumber(n)  { return new Intl.NumberFormat("en-US").format(n); }
function formatCompact(n) { return new Intl.NumberFormat("en-US", { notation:"compact", maximumFractionDigits:1 }).format(n); }
function formatDate(d)    { return new Date(d).toLocaleString("en-US", { timeZone:"Asia/Manila", month:"short", day:"numeric", year:"numeric", hour:"numeric", minute:"2-digit" }); }
function toCount(value)   { return Number.isFinite(Number(value)) ? Number(value) : 0; }

function toTimestamp(value) {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function getPostDisplayTimestamp(post) {
  return toTimestamp(post?.date);
}

function getPostScrapeTimestamp(post) {
  return toTimestamp(post?.createdAt) || getPostDisplayTimestamp(post);
}

function getPostFilterTimestamp(post) {
  return getPostDisplayTimestamp(post);
}

function getPostChronologyTimestamp(post) {
  return getPostDisplayTimestamp(post) || getPostScrapeTimestamp(post);
}

function comparePostsByChronology(a, b, order = "newest") {
  const aTime = getPostChronologyTimestamp(a);
  const bTime = getPostChronologyTimestamp(b);
  if (aTime !== bTime) {
    return order === "oldest" ? aTime - bTime : bTime - aTime;
  }

  const aCreated = getPostScrapeTimestamp(a);
  const bCreated = getPostScrapeTimestamp(b);
  if (aCreated !== bCreated) {
    return order === "oldest" ? aCreated - bCreated : bCreated - aCreated;
  }

  return String(a?.id || "").localeCompare(String(b?.id || ""));
}

function getManilaDateKey(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Manila",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
}

function getManilaDateKeyDaysAgo(daysAgo) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Manila",
    year: "numeric",
    month: "numeric",
    day: "numeric",
  }).formatToParts(new Date());
  const year = Number(parts.find(part => part.type === "year")?.value);
  const month = Number(parts.find(part => part.type === "month")?.value);
  const day = Number(parts.find(part => part.type === "day")?.value);
  const base = new Date(Date.UTC(year, month - 1, day));
  base.setUTCDate(base.getUTCDate() - daysAgo);
  return base.toISOString().slice(0, 10);
}

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
  return comment.source === "X"
    ? `XUsers@${suffix}`
    : `facebookuser@${suffix}`;
}

// ─── Post Engagement ──────────────────────────────────────────────────────────
function getEngagement(post) {
  return post.source === "Facebook"
    ? (post.reactions || 0) + (post.shares || 0)  + (post.comments || 0)
    : (post.likes     || 0) + (post.reposts  || 0) + (post.comments || 0);
}

function normalizePostOrigin(value) {
  const origin = String(value || "").trim().toLowerCase();
  if (origin === "admin") return "Admin";
  if (origin === "people") return "People";
  return "";
}

function getPostOrigin(post) {
  const explicitOrigin = normalizePostOrigin(post?.postOrigin || post?.sourceType || post?.accountType);
  if (explicitOrigin) return explicitOrigin;
  if (post?.source !== "Facebook") return "";

  const pageSource = String(post?.pageSource || "").toLowerCase();
  const sourceUrl = String(post?.sourceUrl || post?.accountUrl || "").toLowerCase();
  if (pageSource.includes("group") || sourceUrl.includes("/groups/")) return "People";
  return "Admin";
}

function matchesPostOrigin(post, origin) {
  if (!origin || origin === "All") return true;
  return getPostOrigin(post) === origin;
}

// ─── Post Filtering & Sorting ─────────────────────────────────────────────────
function matchesDateRange(postDate, range, post = null) {
  if (range === "all") return true;
  const timestamp = post ? getPostFilterTimestamp(post) : toTimestamp(postDate);
  if (!timestamp) return false;
  if (range === "30d") {
    const postKey = getManilaDateKey(timestamp);
    if (!postKey) return false;
    return postKey >= getManilaDateKeyDaysAgo(29) && postKey <= getManilaDateKey(new Date());
  }
  const diffDays = (Date.now() - timestamp) / (1000 * 60 * 60 * 24);
  if (range === "24h") return diffDays <= 1;
  if (range === "3d")  return diffDays <= 3;
  if (range === "7d")  return diffDays <= 7;
  if (range === "14d") return diffDays <= 14;
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
  const { includeResolved = false, postOrigin = "All" } = options;
  return sourcePosts
    .filter(p => includeResolved || !isResolvedPost(p))
    .filter(p => matchesDateRange(p.date, dateRange, p))
    .filter(p => source === "All" ? true : p.source === source)
    .filter(p => matchesPostOrigin(p, postOrigin))
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
function classifyCommentImpact(comment = {}) {
  const text = String(comment.text || "").toLowerCase();
  const likes = toCount(comment.likes);
  const urgentTerms = ["urgent", "sos", "rescue", "help", "init", "tubig", "water", "hospital", "senior", "bata", "newborn"];
  const termScore = urgentTerms.reduce((score, term) => score + (text.includes(term) ? 8 : 0), 0);
  const wordScore = Math.min(text.split(/\s+/).filter(Boolean).length, 12);
  const score = termScore + (likes * 4) + wordScore;
  if (score >= 70) return "High";
  if (score >= 35) return "Medium";
  return "Low";
}

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
function formatManilaDate(now = new Date()) {
  try {
    return new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Manila",
      day: "numeric",
      month: "short",
      year: "numeric",
    }).format(now);
  } catch (_) {
    const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
    const manilaDate = new Date(utcMs + 8 * 60 * 60 * 1000);
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return `${manilaDate.getDate()} ${months[manilaDate.getMonth()]} ${manilaDate.getFullYear()}`;
  }
}

function updateClock() {
  const el = document.getElementById("topbarClock");
  if (!el) return;
  el.textContent = `Updated ${formatManilaDate(new Date())}`;
}
