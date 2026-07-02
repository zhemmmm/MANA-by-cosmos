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
    "all": "All scraped data",
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

function buildDashboardSummary(posts, range, clusters, postOrigin = "All") {
  const filtered = filterPosts(posts || [], range, "All", state.globalSearch, { postOrigin });
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

function buildMockDashboardComments(posts, range, postOrigin = "All") {
  return filterPosts(posts || [], range, "All", state.globalSearch, { postOrigin })
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

function getDashboardPagination(totalPosts) {
  const perPage = Math.max(1, state.dashboardPostsPerPage || 15);
  const totalPages = Math.max(1, Math.ceil(totalPosts / perPage));
  state.dashboardPage = Math.min(Math.max(1, state.dashboardPage || 1), totalPages);
  const startIndex = (state.dashboardPage - 1) * perPage;
  const endIndex = Math.min(startIndex + perPage, totalPosts);
  return {
    perPage,
    totalPages,
    currentPage: state.dashboardPage,
    startIndex,
    endIndex,
  };
}

function buildDashboardPagination(totalPosts) {
  const { totalPages, currentPage } = getDashboardPagination(totalPosts);
  if (totalPosts <= 0 || totalPages <= 1) return "";

  const pages = [];
  const maxVisible = 5;
  let start = Math.max(1, currentPage - 2);
  let end = Math.min(totalPages, start + maxVisible - 1);
  start = Math.max(1, end - maxVisible + 1);

  for (let page = start; page <= end; page += 1) {
    pages.push(`
      <button
        class="dashboard-page-btn ${page === currentPage ? "active" : ""}"
        type="button"
        data-dashboard-page="${page}"
        ${page === currentPage ? "aria-current=\"page\"" : ""}
      >${page}</button>
    `);
  }

  return `
    <button
      class="dashboard-nav-btn"
      type="button"
      data-dashboard-page="${currentPage - 1}"
      ${currentPage === 1 ? "disabled" : ""}
    >Prev</button>
    <div class="dashboard-page-list">${pages.join("")}</div>
    <button
      class="dashboard-nav-btn dashboard-nav-btn-next"
      type="button"
      data-dashboard-page="${currentPage + 1}"
      ${currentPage === totalPages ? "disabled" : ""}
    >Next</button>
  `;
}

function getPagedArchiveView(totalPosts, currentPage, perPage) {
  const pageSize = Math.max(1, perPage || 15);
  const totalPages = Math.max(1, Math.ceil(totalPosts / pageSize));
  const safePage = Math.min(Math.max(1, currentPage || 1), totalPages);
  const startIndex = (safePage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalPosts);

  return {
    totalPages,
    currentPage: safePage,
    startIndex,
    endIndex,
  };
}

function buildArchivePagination(totalPosts, currentPage, perPage, dataAttr = "irrelevant") {
  const { totalPages, currentPage: safePage } = getPagedArchiveView(totalPosts, currentPage, perPage);
  if (totalPosts <= 0 || totalPages <= 1) return "";

  const pages = [];
  const maxVisible = 5;
  let start = Math.max(1, safePage - 2);
  let end = Math.min(totalPages, start + maxVisible - 1);
  start = Math.max(1, end - maxVisible + 1);

  for (let page = start; page <= end; page += 1) {
    pages.push(`
      <button
        class="dashboard-page-btn ${page === safePage ? "active" : ""}"
        type="button"
        data-${dataAttr}-page="${page}"
        ${page === safePage ? "aria-current=\"page\"" : ""}
      >${page}</button>
    `);
  }

  return `
    <button
      class="dashboard-nav-btn"
      type="button"
      data-${dataAttr}-page="${safePage - 1}"
      ${safePage === 1 ? "disabled" : ""}
    >Prev</button>
    <div class="dashboard-page-list">${pages.join("")}</div>
    <button
      class="dashboard-nav-btn dashboard-nav-btn-next"
      type="button"
      data-${dataAttr}-page="${safePage + 1}"
      ${safePage === totalPages ? "disabled" : ""}
    >Next</button>
  `;
}

function updateDashboardFeedMeta(startIndex, endIndex, totalCount) {
  const label = document.getElementById("dashboardFeedCount");
  if (!label) return;
  if (!totalCount) {
    label.textContent = "No matching posts";
    return;
  }
  label.textContent = `Showing ${formatNumber(startIndex + 1)}-${formatNumber(endIndex)} of ${formatNumber(totalCount)} matching posts`;
}

function updateArchiveFeedMeta(labelId, startIndex, endIndex, totalCount) {
  const label = document.getElementById(labelId);
  if (!label) return;
  if (!totalCount) {
    label.textContent = "No matching posts";
    return;
  }
  label.textContent = `Showing ${formatNumber(startIndex + 1)}-${formatNumber(endIndex)} of ${formatNumber(totalCount)} matching posts`;
}

function renderResolvedPostsPanel(postList, emptyMessage, scope = "dashboard") {
  if (!postList.length) {
    return `<div class="watch-empty"><strong>${emptyMessage}</strong></div>`;
  }

  return renderPostCards(postList, { archiveMode: true });
}

function renderIrrelevantPostsPanel(postList) {
  if ((!state.loaded.dashboardIrrelevantPosts || state.loading.dashboardIrrelevantPosts) && !postList.length) {
    return renderLoadingMessage("Loading irrelevant posts...");
  }

  if (!postList.length) {
    return `<div class="watch-empty"><strong>No irrelevant posts match the current filters.</strong>Irrelevant posts fetched from the backend will appear here for review.</div>`;
  }

  return renderPostCards(postList, { archiveMode: true });
}

function renderIrrelevantPostsSummary(postList) {
  const totalIrrelevantPosts = postList.length;
  const value = (!state.loaded.dashboardIrrelevantPosts || state.loading.dashboardIrrelevantPosts) && !totalIrrelevantPosts
    ? "..."
    : formatNumber(totalIrrelevantPosts);
  const meta = (!state.loaded.dashboardIrrelevantPosts || state.loading.dashboardIrrelevantPosts) && !totalIrrelevantPosts
    ? "Loading irrelevant records from the backend..."
    : totalIrrelevantPosts === 1
      ? "1 post marked irrelevant"
      : "Posts marked irrelevant in the current view";

  return `
    <div class="mini-card kpi-red">
      <div class="mini-card-label">Total irrelevant posts</div>
      <div class="mini-card-value">${value}</div>
      <div class="mini-card-meta">${meta}</div>
    </div>
  `;
}

function getIrrelevantPostsViewModel() {
  return filterPosts(
    state.dashboardIrrelevantPosts || [],
    state.dashboardRange,
    "All",
    state.globalSearch,
    { includeResolved: true }
  ).sort((a, b) => getPostScrapeTimestamp(b) - getPostScrapeTimestamp(a));
}

function getDashboardViewModel() {
  const dashboardOrigin = state.dashboardPostOrigin || "All";
  const filteredPosts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch, { postOrigin: dashboardOrigin });
  const dashboardSource = state.dashboardPostsSource || "All";
  const dashboardSort = state.dashboardPostsSort || "newest";
  const dashboardPanelPosts = filteredPosts
    .filter(post => dashboardSource === "All" || post.source === dashboardSource);
  const sortedTrendingPosts = [...dashboardPanelPosts].sort((a, b) => comparePostsByChronology(a, b, dashboardSort));
  const sourceDirectory = [...new Map(filteredPosts.map(p => [p.pageSource, p])).values()].slice(0, 8);
  const pagination = getDashboardPagination(sortedTrendingPosts.length);
  const visiblePosts = sortedTrendingPosts.slice(pagination.startIndex, pagination.endIndex);

  return {
    filteredPosts,
    sortedTrendingPosts,
    visiblePosts,
    sourceDirectory,
    pagination,
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
  const posts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch, { postOrigin: state.dashboardPostOrigin || "All" })
    .filter(p => filter === "All" || p.priority === filter)
    .sort((a, b) => b.severityRank - a.severityRank || getEngagement(b) - getEngagement(a))
    .slice(0, 15);
  const feed = document.getElementById("priorityPostsFeed");
  if (feed) {
    feed.innerHTML = posts.length
      ? renderPostCards(posts)
      : `<div class="watch-empty"><strong>No posts match this priority.</strong></div>`;
  }
}

// ─── Render: Dashboard ────────────────────────────────────────────────────────
function renderDashboard() {
  const { filteredPosts, visiblePosts, sourceDirectory, pagination, sortedTrendingPosts } = getDashboardViewModel();
  const summaryCards = Array.isArray(state.dashboardSummary) && state.dashboardSummary.length
    ? state.dashboardSummary
    : buildDashboardSummary(state.posts, state.dashboardRange, state.clusters, state.dashboardPostOrigin || "All");

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
    ? renderLoadingMessage("Loading dashboard posts...")
    : renderPostCards(visiblePosts);
  updateDashboardFeedMeta(pagination.startIndex, pagination.endIndex, sortedTrendingPosts.length);
  document.getElementById("dashboardPagination").innerHTML = buildDashboardPagination(sortedTrendingPosts.length);

  renderSourceDirectorySection(sourceDirectory);

  const commentCards = Array.isArray(state.dashboardComments) && state.dashboardComments.length
    ? state.dashboardComments
    : buildMockDashboardComments(state.posts, state.dashboardRange, state.dashboardPostOrigin || "All");

  document.getElementById("dashboardComments").innerHTML = commentCards.length ? commentCards.map(c => `
    <article class="comment-card">
      <div class="comment-tag">${c.source || "Facebook"} comment</div>
      <small>${anonymizedCommentAuthor(c)} on ${c.pageSource || "Facebook Source"} · ${(state.clusters.find(cl => cl.id === c.clusterId) || {}).short || ""}</small>
      <p>${c.text}</p>
      <small>${formatNumber(toCount(c.likes))} likes · From post in ${c.location || "Philippines"}</small>
    </article>
  `).join("") : (state.loading.dashboardComments
    ? renderLoadingMessage("Loading top comments...")
    : `<div class="watch-empty"><strong>No trending comments available.</strong>Import a Facebook comments dataset to populate this panel.</div>`);

  renderPriorityPosts();
}

function renderResolvedArchivePage() {
  const resolvedPosts = filterPosts(state.posts, state.dashboardRange, "All", state.globalSearch, { includeResolved: true, postOrigin: state.dashboardPostOrigin || "All" })
    .filter(isResolvedPost)
    .sort((a, b) => getPostScrapeTimestamp(b) - getPostScrapeTimestamp(a));

  document.getElementById("resolvedPostsPanel").innerHTML = renderResolvedPostsPanel(
    resolvedPosts,
    "No resolved posts yet. Resolved items will appear here after they are marked resolved.",
    "resolved"
  );
}

function renderIrrelevantArchivePage() {
  const irrelevantPosts = getIrrelevantPostsViewModel();
  const pagination = getPagedArchiveView(
    irrelevantPosts.length,
    state.irrelevantPage,
    state.irrelevantPostsPerPage
  );
  state.irrelevantPage = pagination.currentPage;
  const visiblePosts = irrelevantPosts.slice(pagination.startIndex, pagination.endIndex);

  document.getElementById("irrelevantPostsSummary").innerHTML = renderIrrelevantPostsSummary(irrelevantPosts);
  document.getElementById("irrelevantPostsPanel").innerHTML = renderIrrelevantPostsPanel(visiblePosts);
  updateArchiveFeedMeta("irrelevantFeedCount", pagination.startIndex, pagination.endIndex, irrelevantPosts.length);
  document.getElementById("irrelevantPagination").innerHTML = buildArchivePagination(
    irrelevantPosts.length,
    pagination.currentPage,
    state.irrelevantPostsPerPage
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
