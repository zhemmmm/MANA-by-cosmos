/**
 * MANA — Dashboard Module
 * Owns: KPI cards, keyword grid, trending posts, source directory, cluster nav.
 * Also owns dashboard mock data and the DataService shim for dashboard endpoints.
 *
 * API endpoints (backend must implement):
 *   GET /api/dashboard/summary   ?date_range=7d → { kpis: [{label,value,meta,bar}] }
 *   GET /api/dashboard/keywords               → { keywords: [{keyword,note,count}] }
 *   GET /api/clusters                         → Cluster[]
 */

// ─── Mock Data ────────────────────────────────────────────────────────────────
const MOCK_CLUSTERS = [
  { id:"cluster-a", short:"Cluster A", name:"Food and Non-food Items (NFIs)",                                                                  description:"Tracks posts about food packs, water, hygiene kits, blankets, and other basic relief needs.",                       keywords:["relief goods","rice","water refill","hygiene kit","blanket","food pack"],                   accent:"#f59e0b" },
  { id:"cluster-b", short:"Cluster B", name:"WASH, Medical and Public Health, Nutrition, Mental Health and Psychosocial Support (Health)",     description:"Tracks posts about health, medicine, clean water, nutrition, and mental health support.",                          keywords:["fever","insulin","washing area","dehydration","doctor","medical team"],                      accent:"#3b82f6" },
  { id:"cluster-c", short:"Cluster C", name:"Camp Coordination, Management and Protection (CCCM)",                                            description:"Tracks evacuation center crowding, camp services, registration, and protection issues.",                           keywords:["evacuation center","overcapacity","privacy","registration","safe space","toilet line"],       accent:"#8b5cf6" },
  { id:"cluster-d", short:"Cluster D", name:"Logistics",                                                                                       description:"Tracks blocked routes, delivery delays, convoy movement, and supply transport issues.",                           keywords:["blocked road","convoy","truck","warehouse","delivery","reroute"],                             accent:"#f97316" },
  { id:"cluster-e", short:"Cluster E", name:"Emergency Telecommunications (ETC)",                                                             description:"Tracks signal loss, network problems, and urgent communication needs.",                                            keywords:["signal down","no network","power bank","cell site","radio","connectivity"],                  accent:"#06b6d4" },
  { id:"cluster-f", short:"Cluster F", name:"Education",                                                                                       description:"Tracks school closures, displaced learners, and temporary learning needs.",                                        keywords:["school closure","class suspension","learning materials","temporary classroom","DepEd","students"], accent:"#10b981" },
  { id:"cluster-g", short:"Cluster G", name:"Search, Rescue and Retrieval (SRR)",                                                             description:"Tracks stranded people, rescue calls, rooftop signals, and retrieval updates.",                                    keywords:["stranded","roof","rescue boat","trapped family","SOS","retrieval"],                          accent:"#ef4444" },
  { id:"cluster-h", short:"Cluster H", name:"Management of Dead and Mission (MDM)",                                                           description:"Tracks missing persons, identification concerns, and related coordination updates.",                                keywords:["missing","identified","hospital list","family tracing","coordination desk","verification"],   accent:"#64748b" },
];

const MOCK_DASHBOARD_SUMMARY = {
  "24h": [
    { label:"High Priority Count",   value:"794",    meta:"Last 24 hours",  bar:37  },
    { label:"Total Posts Analyzed",  value:"2,148",  meta:"Last 24 hours",  bar:100 },
    { label:"Total Facebook Posts",  value:"1,245",  meta:"Last 24 hours",  bar:58  },
    { label:"Total X/Twitter Posts", value:"903",    meta:"Last 24 hours",  bar:42  },
    { label:"Active Clusters",       value:"8",      meta:"8 of 8 clusters",bar:100 },
  ],
  "7d": [
    { label:"High Priority Count",   value:"6,287",  meta:"Last 7 days",   bar:34  },
    { label:"Total Posts Analyzed",  value:"18,492", meta:"Last 7 days",   bar:100 },
    { label:"Total Facebook Posts",  value:"10,428", meta:"Last 7 days",   bar:56  },
    { label:"Total X/Twitter Posts", value:"8,064",  meta:"Last 7 days",   bar:44  },
    { label:"Active Clusters",       value:"8",      meta:"8 of 8 clusters",bar:100 },
  ],
  "14d": [
    { label:"High Priority Count",   value:"11,471", meta:"Last 14 days",  bar:36  },
    { label:"Total Posts Analyzed",  value:"31,864", meta:"Last 14 days",  bar:100 },
    { label:"Total Facebook Posts",  value:"18,065", meta:"Last 14 days",  bar:57  },
    { label:"Total X/Twitter Posts", value:"13,799", meta:"Last 14 days",  bar:43  },
    { label:"Active Clusters",       value:"8",      meta:"8 of 8 clusters",bar:100 },
  ],
  "30d": [
    { label:"High Priority Count",   value:"20,859", meta:"Last 30 days",  bar:33  },
    { label:"Total Posts Analyzed",  value:"63,208", meta:"Last 30 days",  bar:100 },
    { label:"Total Facebook Posts",  value:"35,890", meta:"Last 30 days",  bar:57  },
    { label:"Total X/Twitter Posts", value:"27,318", meta:"Last 30 days",  bar:43  },
    { label:"Active Clusters",       value:"8",      meta:"8 of 8 clusters",bar:100 },
  ],
};

// ─── API Calls ────────────────────────────────────────────────────────────────
async function apiGetClusters()            { return apiFetch("/clusters"); }
async function apiGetDashboardSummary(r)   { const d = await apiFetch(`/dashboard/summary?date_range=${r}`); return d.kpis; }
async function apiGetDashboardComments(r)  { const d = await apiFetch(`/dashboard/comments?date_range=${r}&limit=6`); return d.comments; }
async function apiUpdateEmailAlerts(en)    { return apiFetch("/settings/email-alerts", { method:"PATCH", body: JSON.stringify({ enabled: en }) }); }

// ─── DataService shim ─────────────────────────────────────────────────────────
const DashboardService = {
  async getClusters()           { return USE_MOCK ? MOCK_CLUSTERS : apiGetClusters(); },
  async getDashboardSummary(r)  { return USE_MOCK ? (MOCK_DASHBOARD_SUMMARY[r] || MOCK_DASHBOARD_SUMMARY["7d"]) : apiGetDashboardSummary(r); },
  async getDashboardComments(r) { return USE_MOCK ? null : apiGetDashboardComments(r); },
  async updateEmailAlerts(en)   { if (!USE_MOCK) return apiUpdateEmailAlerts(en); },
};

function dashboardMetaLabel(range) {
  return {
    "24h": "Last 24 hours",
    "3d": "Last 3 days",
    "7d": "Last 7 days",
    "14d": "Last 14 days",
    "30d": "Last 30 days",
  }[range] || "Recent";
}

function dashboardBarValue(value, total) {
  if (!total) return 0;
  return Math.max(12, Math.min(100, Math.round((value / total) * 100)));
}

function buildDashboardSummary(posts, range, clusters) {
  const filtered = filterPosts(posts || [], range, "All", state.globalSearch);
  const totalPosts = filtered.length;
  const highPriorityCount = filtered.filter(post => normalizePriority(post.priority) === "High").length;
  const facebookPosts = filtered.filter(post => post.source === "Facebook").length;
  const twitterPosts = filtered.filter(post => post.source === "X").length;
  const activeClusters = new Set(filtered.map(post => post.clusterId).filter(Boolean)).size;
  const clusterTotal = Math.max((clusters || []).length, 1);
  const meta = dashboardMetaLabel(range);

  return [
    {
      label: "High Priority Count",
      value: formatNumber(highPriorityCount),
      meta,
      bar: dashboardBarValue(highPriorityCount, totalPosts),
    },
    {
      label: "Total Posts Analyzed",
      value: formatNumber(totalPosts),
      meta,
      bar: dashboardBarValue(totalPosts, Math.max(totalPosts, 1)),
    },
    {
      label: "Total Facebook Posts",
      value: formatNumber(facebookPosts),
      meta,
      bar: dashboardBarValue(facebookPosts, totalPosts),
    },
    {
      label: "Total X/Twitter Posts",
      value: formatNumber(twitterPosts),
      meta,
      bar: dashboardBarValue(twitterPosts, totalPosts),
    },
    {
      label: "Active Clusters",
      value: formatNumber(activeClusters),
      meta: `${activeClusters} of ${clusterTotal} clusters`,
      bar: dashboardBarValue(activeClusters, clusterTotal),
    },
  ];
}

function buildMockDashboardComments(posts, range) {
  return filterPosts(posts || [], range, "All", state.globalSearch)
    .filter(post => Array.isArray(post.topComments))
    .flatMap(post => post.topComments.map(comment => ({
      ...comment,
      source: post.source,
      pageSource: post.pageSource,
      clusterId: post.clusterId,
      location: post.location,
      likes: toCount(comment.likes),
    })))
    .slice(0, 6);
}

function renderLoadingMessage(message) {
  return `<div class="watch-empty"><strong>${message}</strong></div>`;
}

function renderResolvedPostsPanel(postList, emptyMessage, scope = "dashboard") {
  if (!postList.length) {
    return `<div class="watch-empty"><strong>${emptyMessage}</strong></div>`;
  }

  return renderPostCards(postList, { archiveMode: true });
}

function getDashboardViewModel() {
  const filteredPosts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch);
  const sortedTrendingPosts = [...filteredPosts]
    .sort((a, b) => (b.severityRank * 1000 + getEngagement(b)) - (a.severityRank * 1000 + getEngagement(a)));
  const sourceDirectory = [...new Map(filteredPosts.map(p => [p.pageSource, p])).values()].slice(0, 8);

  return {
    filteredPosts,
    sortedTrendingPosts,
    sourceDirectory,
  };
}

function renderSourceDirectorySection(sourceDirectory) {
  document.getElementById("sourceDirectory").innerHTML = sourceDirectory.length ? sourceDirectory.map(post => `
    <div class="source-item">
      <div class="source-item-main">
        <div class="source-badge ${post.source === "Facebook" ? "facebook" : "x"}">${post.source === "Facebook" ? "F" : "X"}</div>
        <div class="source-item-meta"><strong>${post.pageSource}</strong><span>${post.source}</span></div>
      </div>
      <div class="source-count">${formatCompact(getEngagement(post))} interactions</div>
    </div>
  `).join("") : (state.loading.criticalData
    ? renderLoadingMessage("Loading source directory...")
    : `<div class="watch-empty"><strong>No sources match the current search.</strong></div>`);
}

// ─── Render: Cluster Nav ──────────────────────────────────────────────────────
function renderClusterNav() {
  document.getElementById("clusterNav").innerHTML = state.clusters.map(cluster => `
    <button class="cluster-btn ${cluster.id === state.currentCluster ? "active" : ""}"
            type="button" data-cluster-nav="${cluster.id}"
            style="--cluster-accent:${cluster.accent};">
      <span class="cluster-btn-title">${cluster.short}: ${cluster.name}</span>
    </button>
  `).join("");
}

// ─── Render: Priority Posts ───────────────────────────────────────────────────
function renderPriorityPosts(priority) {
  const filter = priority || document.getElementById("priorityFilter")?.value || "All";
  const posts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch)
    .filter(p => filter === "All" || p.priority === filter)
    .sort((a, b) => b.severityRank - a.severityRank || getEngagement(b) - getEngagement(a))
    .slice(0, 8);
  const feed = document.getElementById("priorityPostsFeed");
  if (feed) {
    feed.innerHTML = posts.length
      ? renderPostCards(posts)
      : `<div class="watch-empty"><strong>No posts match this priority.</strong></div>`;
  }
}

// ─── Render: Dashboard ────────────────────────────────────────────────────────
function renderDashboard() {
  const { filteredPosts, sortedTrendingPosts, sourceDirectory } = getDashboardViewModel();
  const summaryCards = Array.isArray(state.dashboardSummary) && state.dashboardSummary.length
    ? state.dashboardSummary
    : buildDashboardSummary(state.posts, state.dashboardRange, state.clusters);

  document.getElementById("kpiGrid").innerHTML = summaryCards.length ? summaryCards.map(card => `
    <div class="mini-card ${kpiToneClass(card.label)}">
      <div class="mini-card-label">${card.label}</div>
      <div class="mini-card-value">${card.value}</div>
      <div class="mini-card-meta">${card.meta}</div>
      <div class="mini-bar"><span style="width:${card.bar}%;"></span></div>
    </div>
  `).join("") : renderLoadingMessage("Loading dashboard summary...");

  const keywords = filterKeywords(state.keywords, state.globalSearch);
  document.getElementById("keywordGrid").innerHTML = keywords.length ? keywords.map(item => `
    <div class="keyword-chip">
      <div><strong>${item.keyword}</strong><span>${item.note}</span></div>
      <strong>${formatNumber(item.count)}</strong>
    </div>
  `).join("") : (state.loading.keywords
    ? renderLoadingMessage("Loading keyword signals...")
    : `<div class="watch-empty"><strong>No keywords match the current search.</strong></div>`);

  document.getElementById("dashboardPosts").innerHTML = state.loading.criticalData && !filteredPosts.length
    ? renderLoadingMessage("Loading trending posts...")
    : renderPostCards(sortedTrendingPosts.slice(0, 4));

  renderSourceDirectorySection(sourceDirectory);

  const commentCards = Array.isArray(state.dashboardComments) && state.dashboardComments.length
    ? state.dashboardComments
    : buildMockDashboardComments(state.posts, state.dashboardRange);

  document.getElementById("dashboardComments").innerHTML = commentCards.length ? commentCards.map(c => `
    <article class="comment-card">
      <div class="comment-tag">${c.source || "Facebook"} comment</div>
      <small>${c.author || "Facebook user"} on ${c.pageSource || "Facebook Source"} · ${(state.clusters.find(cl => cl.id === c.clusterId) || {}).short || ""}</small>
      <p>${c.text}</p>
      <small>${formatNumber(toCount(c.likes))} likes · From post in ${c.location || "Philippines"}</small>
    </article>
  `).join("") : (state.loading.dashboardComments
    ? renderLoadingMessage("Loading top comments...")
    : `<div class="watch-empty"><strong>No trending comments available.</strong>Import a Facebook comments dataset to populate this panel.</div>`);

  renderPriorityPosts();
}

function renderResolvedArchivePage() {
  const resolvedPosts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch, { includeResolved: true })
    .filter(isResolvedPost)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  document.getElementById("resolvedPostsPanel").innerHTML = renderResolvedPostsPanel(
    resolvedPosts,
    "No resolved posts yet. Resolved items will appear here after they are marked resolved.",
    "resolved"
  );
}

// ─── Render: Source Directory ─────────────────────────────────────────────────
function renderSourceDirectory() {
  const { sourceDirectory } = getDashboardViewModel();
  renderSourceDirectorySection(sourceDirectory);
}

// ─── Render: Profile Settings ─────────────────────────────────────────────────
function renderProfileSettings() {
  document.getElementById("profileUsername").value    = state.profile.name || state.profile.username;
  document.getElementById("profileRoleLabel").textContent = state.profile.role;
  document.getElementById("profileEmail").value       = state.profile.email;
  document.getElementById("topbarRole").textContent   = state.profile.role;
  document.getElementById("settingsRoleBadge").textContent = state.profile.role;
  document.getElementById("emailAlertsToggle").checked = state.emailAlerts;
}
