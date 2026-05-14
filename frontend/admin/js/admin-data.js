/**
 * MANA Admin — DataService Layer backed by Flask admin endpoints.
 */

const AVATAR_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#0ea5e9", "#a78bfa"];

function profileToUser(u, idx) {
  const initials = (u.name || u.username || u.email)
    .split(" ")
    .map(word => word[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return {
    id: u.id,
    username: u.username,
    name: u.name || u.username,
    email: u.email,
    role: u.role || "LGU Analyst",
    status: u.status || "Active",
    lastLogin: u.last_login_at ? new Date(u.last_login_at).toLocaleString("en-PH", { dateStyle: "medium", timeStyle: "short" }) : "Never",
    created: (u.created_at || "").slice(0, 10),
    loginCount: u.login_count || 0,
    avatar: initials,
    color: AVATAR_COLORS[idx % AVATAR_COLORS.length],
  };
}

const AdminData = {
  async getCurrentAdmin() {
    return fetch(`${AUTH_API_BASE}/me`, {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getAdminToken()}`,
      },
    }).then(async res => {
      if (!res.ok) throw new Error("Could not load admin session.");
      return res.json();
    });
  },

  async getUsers(filters = {}) {
    const params = new URLSearchParams();
    if (filters.search) params.set("search", filters.search);
    if (filters.role) params.set("role", filters.role);
    if (filters.status) params.set("status", filters.status);
    const data = await adminFetch(`/users?${params.toString()}`);
    return (data || []).map(profileToUser);
  },

  async createUser({ name, email, role, password }) {
    const data = await adminFetch("/users", {
      method: "POST",
      body: JSON.stringify({ name, email, role, password }),
    });
    return profileToUser(data, 0);
  },

  async updateUser(id, { name, email, role }) {
    const data = await adminFetch(`/users/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ name, email, role }),
    });
    return profileToUser(data, 0);
  },

  async resetPassword(id, newPassword) {
    return adminFetch(`/users/${encodeURIComponent(id)}/reset-password`, {
      method: "POST",
      body: JSON.stringify({ new_password: newPassword }),
    });
  },

  async setUserStatus(id, status) {
    return adminFetch(`/users/${encodeURIComponent(id)}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    });
  },

  async deleteUser(id) {
    return adminFetch(`/users/${encodeURIComponent(id)}`, { method: "DELETE" });
  },

  async getLogs(filters = {}) {
    const params = new URLSearchParams();
    if (filters.type) params.set("type", filters.type);
    if (filters.user_id) params.set("user_id", filters.user_id);
    if (filters.limit) params.set("limit", filters.limit);
    const data = await adminFetch(`/logs?${params.toString()}`);
    return (data || []).map(log => ({
      id: log.id,
      userId: log.user_id,
      user: log.user_name,
      action: log.action,
      detail: log.detail || "",
      time: new Date(log.created_at).toLocaleString("en-PH", { dateStyle: "short", timeStyle: "short" }),
      type: log.type,
    }));
  },

  async getStats(dateRange = "7d") {
    return adminFetch(`/stats?date_range=${encodeURIComponent(dateRange)}`);
  },

  async getSettings() {
    return adminFetch("/settings");
  },

  async saveSettings(section, data) {
    return adminFetch(`/settings/${section}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  },
};
