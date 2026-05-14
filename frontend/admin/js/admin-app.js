/**
 * MANA Admin — Main Application Logic
 * Handles: routing, login, user management, dashboard, settings, modals, toasts.
 * Load order: admin-config.js → admin-data.js → admin-app.js
 */

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════
const adminState = {
  currentPage: "dashboard",
  admin: { name: "Ana Reyes", email: "ana.reyes@mana.ph", role: "Admin" },
  statsRange: "7d",
  userFilters: { search: "", role: "all", status: "all" },
  users: [],
  logs: [],
  stats: null,
  settings: null,
  editingUserId: null,
  confirmCallback: null,
  currentUserPage: 1,
  usersPerPage: 6,
};

const PAGE_TITLES = {
  dashboard:   { eyebrow: "Overview",        title: "Dashboard Monitoring"    },
  users:       { eyebrow: "Administration",   title: "User Management"         },
  logs:        { eyebrow: "Audit",            title: "Activity Logs"           },
  settings:    { eyebrow: "Configuration",    title: "System Settings"         },
};

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════
async function adminInit() {
  const token = getAdminToken();
  if (!token) {
    window.location.href = "../index.html";
    return;
  }
  try {
    const profile = await AdminData.getCurrentAdmin();
    if (profile.role === "Admin") {
      adminState.admin = {
        id: profile.username,
        name: profile.name || profile.username,
        email: profile.email,
        role: profile.role,
      };
      showAdminApp();
      await loadAllData();
      return;
    }
  } catch (err) {}
  window.location.href = "../index.html";
}

async function loadAllData() {
  renderSidebarUser();
  setAdminPage(adminState.currentPage);
  await Promise.all([
    loadDashboard(),
    loadUsers(),
    loadLogs(),
    loadSettings(),
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════════════════════════════════════
function showAdminApp() {
  document.getElementById("adminApp").classList.remove("hidden");
}

document.getElementById("adminLogoutBtn").addEventListener("click", async () => {
  await fetch(`${AUTH_API_BASE}/logout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAdminToken()}`,
    },
  }).catch(() => {});
  localStorage.removeItem("mana-token");
  clearAdminToken();
  window.location.href = "../index.html";
});

// ═══════════════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════
document.querySelectorAll("[data-admin-nav]").forEach(btn => {
  btn.addEventListener("click", () => setAdminPage(btn.dataset.adminNav));
});

document.getElementById("sidebarCollapseBtn").addEventListener("click", () => {
  document.getElementById("adminApp").classList.toggle("collapsed");
});
document.getElementById("topbarToggleBtn").addEventListener("click", () => {
  const app = document.getElementById("adminApp");
  if (window.innerWidth <= 900) {
    app.classList.toggle("mobile-open");
  } else {
    app.classList.toggle("collapsed");
  }
});

function setAdminPage(page) {
  adminState.currentPage = page;
  document.querySelectorAll(".admin-page").forEach(p => p.classList.toggle("active", p.dataset.page === page));
  document.querySelectorAll("[data-admin-nav]").forEach(b => b.classList.toggle("active", b.dataset.adminNav === page));
  const info = PAGE_TITLES[page] || PAGE_TITLES.dashboard;
  document.getElementById("topbarEyebrow").textContent = info.eyebrow;
  document.getElementById("topbarPageTitle").textContent = info.title;
  document.getElementById("adminApp").classList.remove("mobile-open");
}

function renderSidebarUser() {
  document.getElementById("sidebarUserName").textContent  = adminState.admin.name;
  document.getElementById("sidebarUserRole").textContent  = adminState.admin.role;
  document.getElementById("sidebarUserAvatar").textContent = adminState.admin.name.split(" ").map(w => w[0]).join("").slice(0,2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════
document.getElementById("statsRangeSelect").addEventListener("change", async e => {
  adminState.statsRange = e.target.value;
  await loadDashboard();
});

async function loadDashboard() {
  try {
    adminState.stats = await AdminData.getStats(adminState.statsRange);
    renderDashboard();
  } catch (err) {
    adminToast("Error", "Could not load dashboard stats.");
  }
}

function renderDashboard() {
  const s = adminState.stats;
  if (!s) return;

  // ── KPI Cards ──
  document.getElementById("kpiTotalPosts").textContent  = fmt(s.totalPosts);
  document.getElementById("kpiFbPosts").textContent     = fmt(s.fbPosts);
  document.getElementById("kpiXPosts").textContent      = fmt(s.xPosts);
  document.getElementById("kpiCritical").textContent    = fmt(s.critical);
  document.getElementById("kpiHigh").textContent        = fmt(s.high);
  document.getElementById("kpiModerate").textContent    = fmt(s.moderate);
  document.getElementById("kpiLow").textContent         = fmt(s.low);

  const total = s.critical + s.high + s.moderate + s.low;
  document.getElementById("kpiCriticalBar").style.width  = pct(s.critical,  total) + "%";
  document.getElementById("kpiHighBar").style.width      = pct(s.high,      total) + "%";
  document.getElementById("kpiModerateBar").style.width  = pct(s.moderate,  total) + "%";
  document.getElementById("kpiLowBar").style.width       = pct(s.low,       total) + "%";

  // ── Sentiment Donut ──
  renderSentimentDonut(s.sentiment);

  // ── Post Volume Sparkline ──
  renderSparkline("postVolumeSpark", s.trendFb.map((v,i) => v + s.trendX[i]), "#3b82f6");

  // ── Priority Trend Bars ──
  renderPriorityTrend(s);

  // ── Keywords ──
  document.getElementById("keywordBars").innerHTML = s.topKeywords.map(k => `
    <div class="keyword-row">
      <span class="keyword-label">${k.word}</span>
      <div class="keyword-bar-track">
        <div class="keyword-bar-fill" style="width:${k.pct}%; background: var(--accent);"></div>
      </div>
      <span class="keyword-count">${fmtCompact(k.count)}</span>
    </div>
  `).join("");
}

function renderSentimentDonut(sentiment) {
  const total = sentiment.negative + sentiment.neutral + sentiment.positive;
  const neg = (sentiment.negative / total) * 100;
  const neu = (sentiment.neutral  / total) * 100;
  const pos = (sentiment.positive / total) * 100;
  const r = 28, cx = 40, cy = 40, circumference = 2 * Math.PI * r;

  function arc(pct, offset, color) {
    const dash = (pct / 100) * circumference;
    return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="10"
      stroke-dasharray="${dash} ${circumference - dash}"
      stroke-dashoffset="${-offset * circumference / 100}"
      transform="rotate(-90 ${cx} ${cy})"/>`;
  }

  document.getElementById("sentimentDonut").innerHTML = `
    <svg viewBox="0 0 80 80" class="donut-svg">
      ${arc(neg, 0,       "#f43f5e")}
      ${arc(neu, neg,     "#f59e0b")}
      ${arc(pos, neg+neu, "#10b981")}
    </svg>`;

  document.getElementById("sentimentLegend").innerHTML = `
    <div class="donut-legend-item"><span class="legend-dot" style="background:#f43f5e;"></span><span>Negative</span><strong>${sentiment.negative}%</strong></div>
    <div class="donut-legend-item"><span class="legend-dot" style="background:#f59e0b;"></span><span>Neutral</span><strong>${sentiment.neutral}%</strong></div>
    <div class="donut-legend-item"><span class="legend-dot" style="background:#10b981;"></span><span>Positive</span><strong>${sentiment.positive}%</strong></div>`;
}

function renderSparkline(id, values, color) {
  const el = document.getElementById(id);
  if (!el) return;
  const w = 300, h = 48, max = Math.max(...values), min = Math.min(...values);
  const range = max - min || 1;
  const step = w / (values.length - 1);
  const points = values.map((v, i) => `${i * step},${h - ((v - min) / range) * (h - 8) - 4}`).join(" ");
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:100%;height:100%;">
    <polyline fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${points}"/>
  </svg>`;
}

function renderPriorityTrend(s) {
  const el = document.getElementById("priorityTrendChart");
  if (!el) return;
  const labels = s.trendLabels;
  const critical = s.priorityTrend.critical;
  const high     = s.priorityTrend.high;
  const w = 440, h = 130, pad = 20;
  const all = [...critical, ...high];
  const maxV = Math.max(...all), minV = Math.min(...all);
  const range = maxV - minV || 1;
  const stepX = (w - pad * 2) / (labels.length - 1);

  function line(data, color) {
    const points = data.map((v, i) => {
      const x = pad + i * stepX;
      const y = h - pad - ((v - minV) / range) * (h - pad * 2);
      return `${x},${y}`;
    }).join(" ");
    return `<polyline fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="${points}"/>`;
  }

  const labelMarkup = labels.map((l, i) =>
    `<text x="${pad + i * stepX}" y="${h - 3}" text-anchor="middle" fill="var(--text-faint)" font-size="9" font-family="Manrope,sans-serif">${l}</text>`
  ).join("");

  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:100%;height:100%;">
    <g opacity="0.15" stroke="var(--text-faint)">
      <line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"/>
    </g>
    ${line(critical, "#f43f5e")}
    ${line(high,     "#f59e0b")}
    <g>${labelMarkup}</g>
  </svg>`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// USER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════
async function loadUsers() {
  try {
    adminState.users = await AdminData.getUsers(adminState.userFilters);
    renderUsersTable();
  } catch (err) {
    adminToast("Error", "Could not load users.");
  }
}

function renderUsersTable() {
  const users = adminState.users;
  const perPage = adminState.usersPerPage;
  const page    = adminState.currentUserPage;
  const start   = (page - 1) * perPage;
  const slice   = users.slice(start, start + perPage);
  const total   = users.length;

  document.getElementById("userCount").textContent = `${total} user${total !== 1 ? "s" : ""}`;

  if (!slice.length) {
    document.getElementById("userTableBody").innerHTML =
      `<tr><td colspan="6" style="text-align:center;padding:32px;color:var(--text-faint);">No users match the current filters.</td></tr>`;
  } else {
    document.getElementById("userTableBody").innerHTML = slice.map(u => `
      <tr>
        <td>
          <div class="td-user">
            <div class="table-avatar" style="background:${u.color};">${u.avatar}</div>
            <div class="td-user-info">
              <strong>${u.name}</strong>
              <span>${u.email}</span>
            </div>
          </div>
        </td>
        <td><span class="badge ${roleBadgeClass(u.role)}">${u.role}</span></td>
        <td><span class="badge badge-dot ${statusBadgeClass(u.status)}">${u.status}</span></td>
        <td style="color:var(--text-soft);">${u.lastLogin}</td>
        <td style="color:var(--text-soft);">${u.loginCount}</td>
        <td>
          <div class="td-actions">
            <button class="btn-icon" title="Edit user" onclick="openEditUserModal('${u.id}')">
              <svg viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
            </button>
            <button class="btn-icon" title="Reset password" onclick="openResetPasswordModal('${u.id}')">
              <svg viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
            </button>
            <button class="btn-icon" title="${u.status === 'Active' ? 'Suspend' : 'Activate'} user"
                    style="${u.status === 'Active' ? 'color:var(--amber)' : 'color:var(--green)'}"
                    onclick="toggleUserStatus('${u.id}','${u.status}')">
              <svg viewBox="0 0 24 24">${u.status === "Active"
                ? '<circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line>'
                : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>'
              }</svg>
            </button>
            <button class="btn-icon" title="Delete user" style="color:var(--red);" onclick="confirmDeleteUser('${u.id}','${u.name}')">
              <svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6l-1 14H6L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path><path d="M9 6V4h6v2"></path></svg>
            </button>
          </div>
        </td>
      </tr>
    `).join("");
  }

  // Pagination
  const totalPages = Math.ceil(total / perPage);
  document.getElementById("paginationInfo").textContent = total
    ? `Showing ${start + 1}–${Math.min(start + perPage, total)} of ${total}`
    : "Showing 0 of 0";
  document.getElementById("paginationBtns").innerHTML = Array.from({length: totalPages}, (_, i) =>
    `<button class="page-btn ${page === i+1 ? "active" : ""}" onclick="goToUserPage(${i+1})">${i+1}</button>`
  ).join("");
}

function goToUserPage(page) {
  adminState.currentUserPage = page;
  renderUsersTable();
}

// ── Filters ──────────────────────────────────────────────────────────────────
document.getElementById("userSearchInput").addEventListener("input", async e => {
  adminState.userFilters.search = e.target.value;
  adminState.currentUserPage = 1;
  adminState.users = await AdminData.getUsers(adminState.userFilters);
  renderUsersTable();
});
document.getElementById("userRoleFilter").addEventListener("change", async e => {
  adminState.userFilters.role = e.target.value;
  adminState.currentUserPage = 1;
  adminState.users = await AdminData.getUsers(adminState.userFilters);
  renderUsersTable();
});
document.getElementById("userStatusFilter").addEventListener("change", async e => {
  adminState.userFilters.status = e.target.value;
  adminState.currentUserPage = 1;
  adminState.users = await AdminData.getUsers(adminState.userFilters);
  renderUsersTable();
});

// ── Create User Modal ─────────────────────────────────────────────────────────
document.getElementById("createUserBtn").addEventListener("click", () => {
  adminState.editingUserId = null;
  document.getElementById("userModalTitle").textContent = "Create User";
  document.getElementById("userForm").reset();
  document.getElementById("userPasswordRow").classList.remove("hidden");
  openModal("userModal");
});

function openEditUserModal(userId) {
  const user = adminState.users.find(u => u.id === userId);
  if (!user) return;
  adminState.editingUserId = userId;
  document.getElementById("userModalTitle").textContent = "Edit User";
  document.getElementById("userName").value  = user.name;
  document.getElementById("userEmail").value = user.email;
  document.getElementById("userRole").value  = user.role;
  document.getElementById("userPasswordRow").classList.add("hidden");
  openModal("userModal");
}

document.getElementById("userForm").addEventListener("submit", async e => {
  e.preventDefault();
  const data = {
    name:     document.getElementById("userName").value.trim(),
    email:    document.getElementById("userEmail").value.trim(),
    role:     document.getElementById("userRole").value,
    password: document.getElementById("userPassword").value.trim(),
  };
  if (!adminState.editingUserId) {
    if (!data.name)  { adminToast("Validation Error", "Name is required."); return; }
    if (!data.email) { adminToast("Validation Error", "Email is required."); return; }
    if (!data.password || data.password.length < 8) {
      adminToast("Validation Error", "Password must be at least 8 characters.");
      return;
    }
  }
  try {
    if (adminState.editingUserId) {
      await AdminData.updateUser(adminState.editingUserId, data);
      adminToast("User updated", `${data.name}'s profile has been updated.`);
    } else {
      await AdminData.createUser(data);
      adminToast("User created", `${data.name} has been added to the system.`);
    }
    closeModal("userModal");
    await loadUsers();
    await loadLogs();
  } catch (err) {
    adminToast("Error", err.message || "Could not save user.");
  }
});

// ── Reset Password Modal ──────────────────────────────────────────────────────
function openResetPasswordModal(userId) {
  adminState.editingUserId = userId;
  document.getElementById("resetPasswordForm").reset();
  openModal("resetPasswordModal");
}

document.getElementById("resetPasswordForm").addEventListener("submit", async e => {
  e.preventDefault();
  const newPw  = document.getElementById("newPasswordInput").value.trim();
  const confirm= document.getElementById("confirmPasswordInput").value.trim();
  if (newPw.length < 8) { adminToast("Validation Error", "Password must be at least 8 characters."); return; }
  if (newPw !== confirm) { adminToast("Mismatch", "Passwords do not match."); return; }
  try {
    await AdminData.resetPassword(adminState.editingUserId, newPw);
    closeModal("resetPasswordModal");
    adminToast("Password reset", "The user's password has been updated.");
    await loadLogs();
  } catch (err) {
    adminToast("Error", err.message);
  }
});

// ── Suspend / Activate ────────────────────────────────────────────────────────
async function toggleUserStatus(userId, currentStatus) {
  const newStatus = currentStatus === "Active" ? "Suspended" : "Active";
  const user = adminState.users.find(u => u.id === userId);
  const label = newStatus === "Suspended" ? "suspend" : "reactivate";

  adminState.confirmCallback = async () => {
    try {
      await AdminData.setUserStatus(userId, newStatus);
      adminToast("Status updated", `${user.name} has been ${newStatus === "Suspended" ? "suspended" : "reactivated"}.`);
      await loadUsers();
      await loadLogs();
    } catch (err) {
      adminToast("Error", err.message);
    }
  };

  document.getElementById("confirmText").textContent = `Are you sure you want to ${label} ${user.name}? This will ${newStatus === "Suspended" ? "prevent them from logging in" : "restore their access"}.`;
  openModal("confirmModal");
}

// ── Delete User ───────────────────────────────────────────────────────────────
function confirmDeleteUser(userId, userName) {
  adminState.confirmCallback = async () => {
    try {
      await AdminData.deleteUser(userId);
      adminToast("User deleted", `${userName} has been removed from the system.`);
      await loadUsers();
      await loadLogs();
    } catch (err) {
      adminToast("Error", err.message);
    }
  };
  document.getElementById("confirmText").textContent = `Are you sure you want to permanently delete ${userName}? This action cannot be undone.`;
  openModal("confirmModal");
}

document.getElementById("confirmActionBtn").addEventListener("click", async () => {
  closeModal("confirmModal");
  if (adminState.confirmCallback) {
    await adminState.confirmCallback();
    adminState.confirmCallback = null;
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVITY LOGS
// ═══════════════════════════════════════════════════════════════════════════════
async function loadLogs() {
  try {
    adminState.logs = await AdminData.getLogs();
    renderLogs();
  } catch {}
}

function renderLogs() {
  const TYPE_COLORS = { auth:"#3b82f6", edit:"#10b981", admin:"#8b5cf6", system:"#f59e0b" };
  document.getElementById("activityLogList").innerHTML = adminState.logs.map(log => `
    <div class="activity-item">
      <div class="activity-dot" style="background:${TYPE_COLORS[log.type] || "var(--text-faint)"};"></div>
      <div class="activity-body">
        <strong>${log.user} — ${log.action}</strong>
        <span>${log.detail}</span>
      </div>
      <div class="activity-time">${log.time}</div>
    </div>
  `).join("");
}

document.getElementById("logTypeFilter").addEventListener("change", async e => {
  adminState.logs = await AdminData.getLogs({ type: e.target.value });
  renderLogs();
});

// ═══════════════════════════════════════════════════════════════════════════════
// SETTINGS
// ═══════════════════════════════════════════════════════════════════════════════
async function loadSettings() {
  try {
    adminState.settings = await AdminData.getSettings();
    renderSettings();
  } catch {}
}

function renderSettings() {
  const s = adminState.settings;
  if (!s) return;

  // General
  document.getElementById("settingSystemName").value   = s.general.systemName;
  document.getElementById("settingSystemDesc").value   = s.general.systemDesc;
  document.getElementById("settingTimezone").value     = s.general.timezone;
  document.getElementById("settingDateFormat").value   = s.general.dateFormat;
  document.getElementById("settingDefaultRange").value = s.general.defaultRange;
  document.getElementById("settingMaintenance").checked= s.general.maintenanceMode;

  // Security
  document.getElementById("settingSessionTimeout").value   = s.security.sessionTimeout;
  document.getElementById("settingMaxAttempts").value       = s.security.maxLoginAttempts;
  document.getElementById("settingRequire2FA").checked      = s.security.require2FA;
  document.getElementById("settingMinPassword").value       = s.security.passwordMinLength;
  document.getElementById("settingLogRetention").value      = s.security.logRetentionDays;

  // Notifications
  document.getElementById("settingEmailAlerts").checked    = s.notifications.emailAlerts;
  document.getElementById("settingCriticalAlerts").checked = s.notifications.criticalAlerts;
  document.getElementById("settingDailyDigest").checked    = s.notifications.dailyDigest;
  document.getElementById("settingAlertEmail").value        = s.notifications.alertEmail;

  // System
  document.getElementById("settingScrapeInterval").value   = s.system.scrapeInterval;
  document.getElementById("settingMaxPosts").value          = s.system.maxPostsPerRun;
  document.getElementById("settingRetryOnFail").checked     = s.system.retryOnFail;
  document.getElementById("settingDebugMode").checked       = s.system.debugMode;
  document.getElementById("settingBackupEnabled").checked   = s.system.backupEnabled;
  document.getElementById("settingBackupFreq").value        = s.system.backupFreq;
}

async function saveSettingsSection(section, dataFn) {
  try {
    await AdminData.saveSettings(section, dataFn());
    adminToast("Settings saved", `${capitalize(section)} settings have been updated.`);
  } catch (err) {
    adminToast("Save failed", err.message || "Could not save settings.");
  }
}

document.getElementById("saveGeneralBtn").addEventListener("click", () =>
  saveSettingsSection("general", () => ({
    systemName:      document.getElementById("settingSystemName").value,
    systemDesc:      document.getElementById("settingSystemDesc").value,
    timezone:        document.getElementById("settingTimezone").value,
    dateFormat:      document.getElementById("settingDateFormat").value,
    defaultRange:    document.getElementById("settingDefaultRange").value,
    maintenanceMode: document.getElementById("settingMaintenance").checked,
  }))
);

document.getElementById("saveSecurityBtn").addEventListener("click", () =>
  saveSettingsSection("security", () => ({
    sessionTimeout:    +document.getElementById("settingSessionTimeout").value,
    maxLoginAttempts:  +document.getElementById("settingMaxAttempts").value,
    require2FA:        document.getElementById("settingRequire2FA").checked,
    passwordMinLength: +document.getElementById("settingMinPassword").value,
    logRetentionDays:  +document.getElementById("settingLogRetention").value,
  }))
);

document.getElementById("saveNotifBtn").addEventListener("click", () =>
  saveSettingsSection("notifications", () => ({
    emailAlerts:    document.getElementById("settingEmailAlerts").checked,
    criticalAlerts: document.getElementById("settingCriticalAlerts").checked,
    dailyDigest:    document.getElementById("settingDailyDigest").checked,
    alertEmail:     document.getElementById("settingAlertEmail").value,
  }))
);

document.getElementById("saveSystemBtn").addEventListener("click", () =>
  saveSettingsSection("system", () => ({
    scrapeInterval: +document.getElementById("settingScrapeInterval").value,
    maxPostsPerRun: +document.getElementById("settingMaxPosts").value,
    retryOnFail:    document.getElementById("settingRetryOnFail").checked,
    debugMode:      document.getElementById("settingDebugMode").checked,
    backupEnabled:  document.getElementById("settingBackupEnabled").checked,
    backupFreq:     document.getElementById("settingBackupFreq").value,
  }))
);

// ═══════════════════════════════════════════════════════════════════════════════
// MODALS
// ═══════════════════════════════════════════════════════════════════════════════
function openModal(id)  { document.getElementById(id).classList.add("open"); }
function closeModal(id) { document.getElementById(id).classList.remove("open"); }

document.querySelectorAll("[data-close-modal]").forEach(btn => {
  btn.addEventListener("click", () => closeModal(btn.dataset.closeModal));
});
document.querySelectorAll(".modal-backdrop").forEach(backdrop => {
  backdrop.addEventListener("click", e => {
    if (e.target === backdrop) backdrop.classList.remove("open");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════════════════════
function adminToast(title, message) {
  const wrap  = document.getElementById("adminToastWrap");
  const toast = document.createElement("div");
  toast.className = "admin-toast";
  toast.innerHTML = `<strong>${title}</strong><p>${message}</p>`;
  wrap.appendChild(toast);
  setTimeout(() => toast.remove(), 3400);
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
function fmt(n)         { return new Intl.NumberFormat("en-US").format(n); }
function fmtCompact(n)  { return new Intl.NumberFormat("en-US", { notation:"compact", maximumFractionDigits:1 }).format(n); }
function pct(v, total)  { return total ? Math.round((v / total) * 100) : 0; }
function capitalize(s)  { return s.charAt(0).toUpperCase() + s.slice(1); }

function roleBadgeClass(role) {
  return { Admin:"badge-purple", "LGU Analyst":"badge-blue", Viewer:"badge-slate" }[role] || "badge-slate";
}
function statusBadgeClass(status) {
  return { Active:"badge-green", Suspended:"badge-amber", Inactive:"badge-red" }[status] || "badge-slate";
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════════════════════
adminInit();
