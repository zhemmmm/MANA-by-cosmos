/**
 * MANA Admin — Config & API Core
 * Mirrors the main app's config.js but scoped to admin endpoints.
 *
 * USE_MOCK = true  → hardcoded demo data, no backend needed
 * USE_MOCK = false → calls Flask/Django at ADMIN_API_BASE
 *
 * Backend role guard: every admin API call requires
 *   Authorization: Bearer <token>  where the token encodes role = "Admin"
 */

const SUPABASE_URL = "https://gizuoookwwkximbqvcpx.supabase.co";
const SUPABASE_KEY = "sb_publishable_cj0YjBeAVubMaZVOyYXNyQ_D0en0BF_";

const API_ROOT = "https://mana-backend-4s1w.onrender.com/api";
// const API_ROOT = "http://localhost:5000/api";
const AUTH_API_BASE = `${API_ROOT}/auth`;
const ADMIN_API_BASE = `${API_ROOT}/admin`;
const USE_MOCK = false;

// ─── Admin JWT Helpers ────────────────────────────────────────────────────────
function getAdminToken()       { return localStorage.getItem("mana-admin-token") || localStorage.getItem("mana-token"); }
function setAdminToken(t)      { localStorage.setItem("mana-admin-token", t); }
function clearAdminToken()     { localStorage.removeItem("mana-admin-token"); }

// ─── Core Fetch ───────────────────────────────────────────────────────────────
/**
 * adminFetch(endpoint, options)
 * All admin API calls go through this.
 * Auto-attaches Bearer token, handles 401/403.
 */
async function adminFetch(endpoint, options = {}) {
  const token = getAdminToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${ADMIN_API_BASE}${endpoint}`, { ...options, headers });

  if (res.status === 401 || res.status === 403) {
    clearAdminToken();
    window.location.href = "../index.html";
    throw new Error("Session expired or insufficient permissions.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `API error ${res.status}`);
  }
  return res.json();
}
