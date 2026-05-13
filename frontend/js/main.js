/**
 * MANA — Main Entry Point
 * Bootstraps the app: loads data, wires events, owns app-level state and routing.
 *
 * Load order in index.html (scripts must appear in this order):
 *   config.js → utils.js → auth.js → posts.js → charts.js → dashboard.js → main.js
 */

// ─── App State ────────────────────────────────────────────────────────────────
const state = {
  // Data (populated from backend or mock on init)
  clusters:         [],
  posts:            [],
  keywords:         [],
  dashboardSummary: [],
  dashboardComments: [],
  analytics:        {},
  dashboardCommentsRange: null,
  analyticsLoadedRange:   null,

  // UI state
  currentPage:      "dashboard",
  currentCluster:   "cluster-a",
  currentTheme:     "dark",
  currentCaptcha:   "",

  // Filter state
  dashboardRange:   "30d",
  alerts:           { dateRange: "30d", source: "All" },
  clusterFilters:   { source: "All", severity: "Trending", dateRange: "30d" },
  analyticsRange:   "30d",
  globalSearch:     "",

  // User state
  pinned:        new Set(),
  statuses:      {},
  statusHistory: {},
  verifications: {},
  profile:       { username: "admin_mana", role: "LGU Analyst", email: "lgu.analyst@mana.ph" },
  emailAlerts:   true,
  loading: {
    criticalData: false,
    keywords: false,
    dashboardComments: false,
    analytics: false,
    watchlist: false,
  },
  loaded: {
    criticalData: false,
    keywords: false,
    dashboardComments: false,
    analytics: false,
    watchlist: false,
  },
};

const pageTitles = {
  dashboard:       { eyebrow:"Dashboard",        title:"MANA command overview" },
  resolved:        { eyebrow:"Resolved Posts",   title:"Resolved archive and restore queue" },
  analytics:       { eyebrow:"Analytics",        title:"Trend and sentiment analysis" },
  alerts:          { eyebrow:"Live Alerts",      title:"Priority cluster and severity watch" },
  watchlist:       { eyebrow:"Saved Intelligence",title:"Pinned incident review queue" },
  settings:        { eyebrow:"Settings",         title:"Profile, alerts, and security" },
  "cluster-detail":{ eyebrow:"Cluster Detail",   title:"Operational cluster profile" },
};

let appBootPromise = null;
let criticalDataPromise = null;
let deferredDataPromise = null;
let liveUpdatesChannel = null;
let liveRefreshTimer = null;

function resetDeferredState() {
  criticalDataPromise = null;
  deferredDataPromise = null;
  state.loaded.criticalData = false;
  state.loaded.keywords = false;
  state.loaded.dashboardComments = false;
  state.loaded.analytics = false;
  state.loaded.watchlist = false;
  state.loading.criticalData = false;
  state.loading.keywords = false;
  state.loading.dashboardComments = false;
  state.loading.analytics = false;
  state.loading.watchlist = false;
  state.dashboardCommentsRange = null;
  state.analyticsLoadedRange = null;
  state.clusters = [];
  state.posts = [];
  state.keywords = [];
  state.dashboardSummary = [];
  state.dashboardComments = [];
  state.analytics = {};
  state.statuses = {};
  state.statusHistory = {};
  state.verifications = {};
  state.pinned = new Set();
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  if (appBootPromise) return appBootPromise;

  appBootPromise = (async () => {
    hydrateLocalPreferences();
    applyTheme(state.currentTheme);
    generateCaptcha();
    bindStaticControls();
    updateClock();
    await checkRememberedSession();
  })();

  return appBootPromise;
}

async function loadCriticalAppData() {
  if (criticalDataPromise) return criticalDataPromise;

  state.loading.criticalData = true;
  criticalDataPromise = (async () => {
    try {
      const [clusters, posts] = await Promise.all([
        DashboardService.getClusters(),
        PostsService.getPosts(),
      ]);

      state.clusters = clusters;
      state.posts    = posts.map(post => ({
        ...post,
        reactions: toCount(post.reactions),
        shares: toCount(post.shares),
        likes: toCount(post.likes),
        reposts: toCount(post.reposts),
        comments: toCount(post.comments),
        views: toCount(post.views),
      }));
      state.statuses = Object.fromEntries(state.posts.map(p => [p.id, p.status]));

      const saved = localStorage.getItem("mana-statuses");
      if (saved) state.statuses = { ...state.statuses, ...JSON.parse(saved) };
      const savedHistory = localStorage.getItem("mana-status-history");
      if (savedHistory) state.statusHistory = { ...state.statusHistory, ...JSON.parse(savedHistory) };

      state.dashboardSummary = buildDashboardSummary(state.posts, state.dashboardRange, state.clusters);
      initVerifications(state.posts);
      state.loaded.criticalData = true;

      renderClusterNav();
      renderCurrentPage({ refreshDashboardSummary: false });
    } catch (err) {
      console.error("Critical data load failed:", err);
      showToast("Data load error", err.message || "Could not load data. Check backend connection.");
      throw err;
    } finally {
      state.loading.criticalData = false;
    }
  })();

  return criticalDataPromise;
}

async function loadDeferredAppData() {
  if (deferredDataPromise) return deferredDataPromise;

  deferredDataPromise = (async () => {
    await Promise.allSettled([
      loadKeywords(),
      loadDashboardComments(state.dashboardRange),
      loadWatchlist(),
      loadAnalytics(state.analyticsRange),
    ]);
  })();

  return deferredDataPromise;
}

async function loadKeywords() {
  if (state.loaded.keywords || state.loading.keywords) return state.keywords;

  state.loading.keywords = true;
  try {
    state.keywords = await PostsService.getKeywords();
    state.loaded.keywords = true;
    if (state.currentPage === "dashboard") renderDashboard();
  } catch (err) {
    console.warn("Keyword load failed:", err);
  } finally {
    state.loading.keywords = false;
  }
}

async function loadDashboardComments(range = state.dashboardRange) {
  if (state.loading.dashboardComments) return state.dashboardComments;
  if (state.loaded.dashboardComments && state.dashboardCommentsRange === range) return state.dashboardComments;

  state.loading.dashboardComments = true;
  try {
    state.dashboardComments = await DashboardService.getDashboardComments(range).catch(() => []);
    state.loaded.dashboardComments = true;
    state.dashboardCommentsRange = range;
    if (state.currentPage === "dashboard") renderDashboard();
  } finally {
    state.loading.dashboardComments = false;
  }
}

async function loadAnalytics(range = state.analyticsRange) {
  if (state.loading.analytics) return state.analytics;
  if (state.loaded.analytics && state.analyticsLoadedRange === range) return state.analytics;

  state.loading.analytics = true;
  try {
    state.analytics = await ChartsService.getAnalytics(range);
    state.loaded.analytics = true;
    state.analyticsLoadedRange = range;
    if (state.currentPage === "analytics") renderAnalytics();
  } catch (err) {
    console.warn("Analytics load failed:", err);
  } finally {
    state.loading.analytics = false;
  }
}

async function loadWatchlist() {
  if (state.loaded.watchlist || state.loading.watchlist) return state.pinned;

  state.loading.watchlist = true;
  try {
    const watchlist = await PostsService.getWatchlist();
    state.pinned = new Set(watchlist.pinned || []);
    state.loaded.watchlist = true;
    if (state.currentPage === "watchlist") renderWatchlist();
  } catch (err) {
    console.warn("Watchlist load failed:", err);
  } finally {
    state.loading.watchlist = false;
  }
}

// ─── Verification helpers ─────────────────────────────────────────────────────
function initVerifications(posts) {
  state.verifications = {};
  for (const post of posts) {
    const ref = MOCK_CROSS_REFS[post.id];
    state.verifications[post.id] = ref
      ? { status:"auto-verified",   crossRefs:ref.crossRefs, matchCount:ref.matchCount, note:"", markedBy:null }
      : { status:"auto-unverified", crossRefs:[],             matchCount:0,              note:"", markedBy:null };
  }
  const saved = localStorage.getItem("mana-verifications");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      for (const [id, v] of Object.entries(parsed)) {
        if (state.verifications[id]) state.verifications[id] = { ...state.verifications[id], ...v };
      }
    } catch (_) {}
  }
}

function persistVerifications() {
  localStorage.setItem("mana-verifications", JSON.stringify(state.verifications));
}

function refreshVerifyBox(postId) {
  const post = state.posts.find(p => p.id === postId);
  if (!post) return;

  // Capture which wrappers had an open popup before replacing
  const openWrappers = new Set();
  document.querySelectorAll(`[data-verify-for="${postId}"]`).forEach(existing => {
    if (!existing.classList.contains("hidden")) openWrappers.add(existing.closest(".verify-wrapper"));
    existing.outerHTML = renderVerifyBox(post);
  });

  // Re-open popups in the wrappers that were open
  openWrappers.forEach(wrapper => {
    if (!wrapper) return;
    const fresh = wrapper.querySelector(`[data-verify-for="${postId}"]`);
    if (fresh) fresh.classList.remove("hidden");
  });

  const v = state.verifications[postId];
  const isVerified = v.status === "auto-verified" || v.status === "manually-verified";
  document.querySelectorAll(`[data-verify-toggle="${postId}"]`).forEach(btn => {
    btn.textContent = isVerified ? "✓ Verified" : "⊕ Unverified";
    btn.className   = `verify-btn verify-${v.status}`;
  });
}

// ─── Preferences (localStorage only — no API) ─────────────────────────────────
function hydrateLocalPreferences() {
  const theme = localStorage.getItem("mana-theme");
  if (theme === "light" || theme === "dark") state.currentTheme = theme;

  if (USE_MOCK) {
    const savedPinned = localStorage.getItem("mana-pinned");
    if (savedPinned) state.pinned = new Set(JSON.parse(savedPinned));

    const savedProfile = localStorage.getItem("mana-profile");
    if (savedProfile) state.profile = { ...state.profile, ...JSON.parse(savedProfile) };
  }

  const savedAlerts = localStorage.getItem("mana-email-alerts");
  if (savedAlerts !== null) state.emailAlerts = savedAlerts === "true";
}

function persistLocalPreferences() {
  localStorage.setItem("mana-statuses", JSON.stringify(state.statuses));
  localStorage.setItem("mana-status-history", JSON.stringify(state.statusHistory));
  if (USE_MOCK) {
    localStorage.setItem("mana-pinned", JSON.stringify([...state.pinned]));
    localStorage.setItem("mana-profile", JSON.stringify(state.profile));
  }
  localStorage.setItem("mana-email-alerts", String(state.emailAlerts));
}

function getRestoreStatus(postId) {
  const remembered = state.statusHistory[postId];
  if (remembered && !isResolvedStatus(remembered)) return remembered;

  const original = state.posts.find(post => post.id === postId)?.status;
  if (original && !isResolvedStatus(original)) return original;

  return "Monitoring";
}

async function setPostStatus(postId, status, options = {}) {
  const { silent = false } = options;
  const post = state.posts.find(item => item.id === postId);
  const previousStatus = getPostStatus(post || { id: postId, status: "Monitoring" });

  if (!isResolvedStatus(previousStatus)) {
    state.statusHistory[postId] = previousStatus;
  }
  if (!isResolvedStatus(status) && status) {
    state.statusHistory[postId] = status;
  }

  state.statuses[postId] = status;
  persistLocalPreferences();
  renderCurrentPage({ refreshDashboardSummary: true });

  document.querySelectorAll(`[data-status-select="${postId}"]`).forEach(select => {
    select.value = status;
    applyStatusStyle(select, status);
  });

  try {
    const result = await PostsService.updatePostStatus(postId, status);
    if (!silent) {
      showToast(
        result?.localOnly ? "Status updated locally" : "Post status updated",
        result?.localOnly
          ? "The operational status was saved in this browser for now."
          : "The operational status has been updated."
      );
    }
  } catch (err) {
    if (!silent) {
      showToast("Status saved locally", "The card updated in the dashboard, but the backend request failed.");
    }
    console.warn("Status update failed:", err);
  }
}

function scheduleLiveRefresh() {
  clearTimeout(liveRefreshTimer);
  liveRefreshTimer = setTimeout(async () => {
    try {
      criticalDataPromise = null;
      deferredDataPromise = null;
      state.loaded.criticalData = false;
      state.loaded.keywords = false;
      state.loaded.dashboardComments = false;
      state.loaded.analytics = false;
      state.loaded.watchlist = false;

      await loadCriticalAppData();
      await loadDeferredAppData();
      renderCurrentPage({ refreshDashboardSummary: true });
      showToast("Live update received", "New Apify data was loaded automatically.");
    } catch (err) {
      console.warn("Live refresh failed:", err);
    }
  }, 2000);
}

function startLiveUpdates() {
  if (USE_MOCK || !window.supabase || liveUpdatesChannel) return;

  liveUpdatesChannel = window.supabase
    .channel("mana-live-updates")
    .on(
      "postgres_changes",
      { event: "INSERT", schema: "public", table: "posts" },
      payload => {
        console.log("Realtime post insert:", payload);
        scheduleLiveRefresh();
      }
    )
    .on(
      "postgres_changes",
      { event: "UPDATE", schema: "public", table: "posts" },
      payload => {
        console.log("Realtime post update:", payload);
        scheduleLiveRefresh();
      }
    )
    .on(
      "postgres_changes",
      { event: "INSERT", schema: "public", table: "comments" },
      payload => {
        console.log("Realtime comment insert:", payload);
        scheduleLiveRefresh();
      }
    )
    .on(
      "postgres_changes",
      { event: "UPDATE", schema: "public", table: "comments" },
      payload => {
        console.log("Realtime comment update:", payload);
        scheduleLiveRefresh();
      }
    )
    .subscribe(status => {
      console.log("Supabase Realtime status:", status);
    });
}

function stopLiveUpdates() {
  if (liveUpdatesChannel && window.supabase) {
    window.supabase.removeChannel(liveUpdatesChannel);
  }
  liveUpdatesChannel = null;
  clearTimeout(liveRefreshTimer);
}

// ─── Event Bindings ───────────────────────────────────────────────────────────
function bindStaticControls() {
  document.getElementById("loginForm").addEventListener("submit", handleLogin);

  document.getElementById("dashboardRange").addEventListener("change", async e => {
    state.dashboardRange  = e.target.value;
    state.dashboardSummary = buildDashboardSummary(state.posts, state.dashboardRange, state.clusters);
    loadDashboardComments(state.dashboardRange);
    renderDashboard();
  });

  document.getElementById("priorityFilter").addEventListener("change", e => {
    renderPriorityPosts(e.target.value);
  });

  document.getElementById("analyticsRange").addEventListener("change", async e => {
    state.analyticsRange = e.target.value;
    renderAnalytics();
    loadAnalytics(state.analyticsRange);
  });

  document.getElementById("alertsDateRange").addEventListener("change", e => { state.alerts.dateRange = e.target.value; renderAlerts(); });
  document.getElementById("alertsSource").addEventListener("change",    e => { state.alerts.source    = e.target.value; renderAlerts(); });

  document.getElementById("clusterSourceFilter").addEventListener("change",   e => { state.clusterFilters.source   = e.target.value; renderClusterDetail(); });
  document.getElementById("clusterSeverityFilter").addEventListener("change", e => { state.clusterFilters.severity = e.target.value; renderClusterDetail(); });
  document.getElementById("clusterDateFilter").addEventListener("change",     e => { state.clusterFilters.dateRange= e.target.value; renderClusterDetail(); });

  const globalSearch = document.getElementById("globalSearch");
  const applyGlobalSearch = value => {
    state.globalSearch = value;
    renderCurrentPage({ refreshDashboardSummary: true });
  };

  globalSearch.addEventListener("input", e => applyGlobalSearch(e.target.value));
  globalSearch.addEventListener("search", e => applyGlobalSearch(e.target.value));
  document.querySelector(".search-btn")?.addEventListener("click", () => applyGlobalSearch(globalSearch.value));

  document.getElementById("themeCheckbox").addEventListener("change", e => applyTheme(e.target.checked ? "dark" : "light"));

  document.getElementById("emailAlertsToggle").addEventListener("change", async e => {
    state.emailAlerts = e.target.checked;
    await DashboardService.updateEmailAlerts(state.emailAlerts).catch(() => {});
    persistLocalPreferences();
    showToast("Email alerts updated", state.emailAlerts ? "Email alerts are now on." : "Email alerts are now off.");
  });

  document.getElementById("savePasswordBtn").addEventListener("click",  savePassword);
  document.getElementById("clearPasswordBtn").addEventListener("click", clearPasswordFields);
  document.getElementById("saveProfileBtn").addEventListener("click",   saveProfile);
  document.getElementById("changeEmailBtn").addEventListener("click",   toggleEmailPopover);
  document.getElementById("verifyEmailBtn").addEventListener("click",   verifyEmailChange);
  document.getElementById("cancelEmailBtn").addEventListener("click",   closeEmailPopover);
  document.getElementById("testAlertBtn").addEventListener("click", () =>
    showToast("Test alert sent", `A sample email alert was sent to ${state.profile.email}.`));
  document.getElementById("logoutBtn").addEventListener("click", doLogout);

  document.addEventListener("click",  handleDocumentClick);
  document.addEventListener("change", handleDocumentChange);
}

function handleDocumentClick(event) {
  const nav           = event.target.closest("[data-nav]");
  const clusterNav    = event.target.closest("[data-cluster-nav]");
  const pwToggle      = event.target.closest("[data-toggle-password]");
  const action        = event.target.closest("[data-action]");
  const pin           = event.target.closest("[data-pin]");
  const commentToggle = event.target.closest("[data-toggle-comments]");
  const restorePost   = event.target.closest("[data-restore-post]");

  if (nav)        setPage(nav.dataset.nav);
  if (clusterNav) { state.currentCluster = clusterNav.dataset.clusterNav; setPage("cluster-detail"); }
  if (pwToggle)   togglePasswordField(pwToggle.dataset.togglePassword);

  if (action) {
    const a = action.dataset.action;
    if (a === "refresh-captcha") { generateCaptcha(); showToast("CAPTCHA refreshed", "A new verification code has been generated."); }
    if (a === "forgot-password") showToast("Password recovery", "Route password reset requests to the system administrator.");
    if (a === "toggle-sidebar")  toggleSidebar();
    if (a === "close-sidebar")   document.body.classList.remove("sidebar-open");
  }

  if (pin) togglePin(pin.dataset.pin);
  if (restorePost) {
    const postId = restorePost.dataset.restorePost;
    setPostStatus(postId, getRestoreStatus(postId), { silent: false });
  }

  const openPost = event.target.closest("[data-open-post]");
  if (openPost) {
    const post = state.posts.find(p => p.id === openPost.dataset.openPost);
    const sourceUrl = String(post?.sourceUrl || "").trim();

    if (sourceUrl && /^https?:\/\//i.test(sourceUrl)) {
      window.open(sourceUrl, "_blank", "noopener,noreferrer");
    } else {
      showToast("Post link unavailable", "This post does not currently have a valid source URL.");
    }
  }

  if (commentToggle) {
    const postCard = commentToggle.closest("[data-post-card]");
    const box = postCard?.querySelector(`[data-comments-box="${commentToggle.dataset.toggleComments}"]`);
    if (box) {
      box.classList.toggle("open");
      const isOpen = box.classList.contains("open");
      commentToggle.setAttribute("aria-expanded", String(isOpen));
      const label = commentToggle.querySelector(".comment-toggle-label");
      if (label) label.textContent = isOpen ? "Hide Comments" : "View Comments";
    }
  }

  const verifyToggle = event.target.closest("[data-verify-toggle]");
  if (!event.target.closest(".verify-wrapper")) {
    document.querySelectorAll(".verify-popup:not(.hidden)").forEach(p => p.classList.add("hidden"));
  }
  if (verifyToggle) {
    const popup = verifyToggle.closest(".verify-wrapper")?.querySelector(".verify-popup");
    document.querySelectorAll(".verify-popup:not(.hidden)").forEach(p => {
      if (p !== popup) p.classList.add("hidden");
    });
    if (popup) popup.classList.toggle("hidden");
  }

  const recToggle = event.target.closest("[data-rec-toggle]");
  if (!event.target.closest(".rec-wrapper")) {
    document.querySelectorAll(".rec-popup:not(.hidden)").forEach(p => p.classList.add("hidden"));
  }
  if (recToggle) {
    const popup = recToggle.closest(".rec-wrapper")?.querySelector(".rec-popup");
    document.querySelectorAll(".rec-popup:not(.hidden)").forEach(p => {
      if (p !== popup) p.classList.add("hidden");
    });
    if (popup) popup.classList.toggle("hidden");
  }

  const markUnverified = event.target.closest("[data-mark-unverified]");
  if (markUnverified) {
    const pid = markUnverified.dataset.markUnverified;
    if (state.verifications[pid]) {
      state.verifications[pid].status   = "marked-unverified";
      state.verifications[pid].markedBy = state.profile.username || state.profile.name || "Unknown";
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  const manualVerify = event.target.closest("[data-manually-verify]");
  if (manualVerify) {
    const pid = manualVerify.dataset.manuallyVerify;
    if (state.verifications[pid]) {
      state.verifications[pid].status   = "manually-verified";
      state.verifications[pid].markedBy = state.profile.username || state.profile.name || "Unknown";
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  const reverify = event.target.closest("[data-reverify]");
  if (reverify) {
    const pid = reverify.dataset.reverify;
    if (state.verifications[pid]) {
      state.verifications[pid].status   = "auto-verified";
      state.verifications[pid].markedBy = null;
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  const unverifyManual = event.target.closest("[data-unverify-manual]");
  if (unverifyManual) {
    const pid = unverifyManual.dataset.unverifyManual;
    if (state.verifications[pid]) {
      state.verifications[pid].status   = "auto-unverified";
      state.verifications[pid].markedBy = null;
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  const addNote = event.target.closest("[data-add-note]");
  if (addNote) {
    const editor = addNote.closest(".verify-wrapper")?.querySelector(`[data-note-editor="${addNote.dataset.addNote}"]`);
    if (editor) editor.classList.remove("hidden");
  }

  const cancelNote = event.target.closest("[data-cancel-note]");
  if (cancelNote) {
    const editor = cancelNote.closest(".verify-wrapper")?.querySelector(`[data-note-editor="${cancelNote.dataset.cancelNote}"]`);
    if (editor) editor.classList.add("hidden");
  }

  const saveNote = event.target.closest("[data-save-note]");
  if (saveNote) {
    const pid   = saveNote.dataset.saveNote;
    const input = saveNote.closest(".verify-wrapper")?.querySelector(`[data-note-input="${pid}"]`);
    if (input && state.verifications[pid]) {
      state.verifications[pid].note = input.value.trim();
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  const editNote = event.target.closest("[data-edit-note]");
  if (editNote && editNote.dataset.editNote) {
    const pid    = editNote.dataset.editNote;
    const wrapper = editNote.closest(".verify-wrapper");
    const editor = wrapper?.querySelector(`[data-note-editor="${pid}"]`);
    if (editor) {
      editor.classList.remove("hidden");
      const ta = wrapper?.querySelector(`[data-note-input="${pid}"]`);
      if (ta) ta.value = state.verifications[pid]?.note || "";
    }
  }

  const deleteNote = event.target.closest("[data-delete-note]");
  if (deleteNote) {
    const pid = deleteNote.dataset.deleteNote;
    if (state.verifications[pid]) {
      state.verifications[pid].note = "";
      persistVerifications();
      refreshVerifyBox(pid);
    }
  }

  if (!event.target.closest(".email-inline")) {
    document.getElementById("emailPopover").classList.add("hidden");
  }
}

async function handleDocumentChange(event) {
  if (event.target.matches("[data-status-select]")) {
    const postId = event.target.dataset.statusSelect;
    const status = event.target.value;
    applyStatusStyle(event.target, status);
    await setPostStatus(postId, status);
  }
  if (event.target.matches("#clusterSeverityFilter")) {
    applySeverityStyle(event.target, event.target.value);
  }
}

// ─── Render (orchestrates all modules) ───────────────────────────────────────
function renderAll() {
  renderProfileSettings();
  renderCurrentPage({ refreshDashboardSummary: true });
}

function renderCurrentPage({ refreshDashboardSummary = false } = {}) {
  if (refreshDashboardSummary && state.loaded.criticalData) {
    state.dashboardSummary = buildDashboardSummary(state.posts, state.dashboardRange, state.clusters);
  }

  if (state.currentPage === "dashboard") {
    renderDashboard();
    return;
  }
  if (state.currentPage === "analytics") {
    renderAnalytics();
    return;
  }
  if (state.currentPage === "resolved") {
    renderResolvedArchivePage();
    return;
  }
  if (state.currentPage === "alerts") {
    renderAlerts();
    return;
  }
  if (state.currentPage === "watchlist") {
    renderWatchlist();
    return;
  }
  if (state.currentPage === "cluster-detail") {
    renderClusterDetail();
    return;
  }
  if (state.currentPage === "settings") {
    renderProfileSettings();
  }
}

// ─── Page Routing ─────────────────────────────────────────────────────────────
function setPage(page) {
  state.currentPage = page;
  document.querySelectorAll(".page").forEach(s => s.classList.toggle("active", s.dataset.page === page));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.nav === page));
  document.querySelectorAll(".cluster-btn").forEach(b =>
    b.classList.toggle("active", page === "cluster-detail" && b.dataset.clusterNav === state.currentCluster));

  const title = pageTitles[page];
  document.getElementById("topbarEyebrow").textContent = title.eyebrow;
  document.getElementById("topbarTitle").textContent   = title.title;
  renderCurrentPage({ refreshDashboardSummary: page === "dashboard" });
  if (page === "analytics" && !state.loaded.analytics) loadAnalytics(state.analyticsRange);
  if (page === "analytics" && state.analyticsLoadedRange !== state.analyticsRange) loadAnalytics(state.analyticsRange);
  if (page === "watchlist" && !state.loaded.watchlist) loadWatchlist();
  if (page === "dashboard") {
    if (!state.loaded.keywords) loadKeywords();
    if (state.dashboardCommentsRange !== state.dashboardRange || !state.loaded.dashboardComments) {
      loadDashboardComments(state.dashboardRange);
    }
  }
  document.body.classList.remove("sidebar-open");
}

// ─── Theme & Sidebar ─────────────────────────────────────────────────────────
function applyTheme(theme) {
  state.currentTheme = theme;
  document.body.classList.toggle("light-theme", theme === "light");
  document.getElementById("themeCheckbox").checked = theme === "dark";
  document.getElementById("themeModeText").textContent = theme === "dark" ? "Dark mode active" : "Light mode active";
  localStorage.setItem("mana-theme", theme);
}

function toggleSidebar() {
  if (window.innerWidth <= 1240) { document.body.classList.toggle("sidebar-open"); return; }
  document.body.classList.toggle("sidebar-collapsed");
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
