/**
 * MANA — Configuration & API Core
 * Single source of truth for backend URL, mock toggle, JWT helpers, and apiFetch().
 *
 * Backend: Python Flask or Django (see /backend/)
 * Auth: JWT Bearer token stored in localStorage after login.
 *
 * USE_MOCK = true  → hardcoded demo data, no backend needed
 * USE_MOCK = false → all calls go to Flask/Django at API_BASE
 */

const SUPABASE_URL = "https://gilmqjaygnkcrabyacnv.supabase.co";
const SUPABASE_KEY = "sb_publishable_KAreWH4IpJSaSPTMxwT3lQ_dTianTP-";

const API_BASE = "https://mana-backend-4s1w.onrender.com/api";
// const API_BASE = "http://localhost:5000/api";
// try lang
const USE_MOCK = false;

// ─── JWT Helpers ──────────────────────────────────────────────────────────────
function getToken() { return localStorage.getItem("mana-token"); }
function setToken(token) { localStorage.setItem("mana-token", token); }
function clearToken() { localStorage.removeItem("mana-token"); }

// ─── Core Fetch Wrapper ───────────────────────────────────────────────────────
/**
 * apiFetch(endpoint, options)
 * Attaches auth headers, handles 401 auto-logout, and throws readable errors.
 *
 * @param {string}      endpoint  e.g. "/posts?date_range=7d"
 * @param {RequestInit} options   standard fetch options
 * @returns {Promise<any>}        parsed JSON response
 */
async function apiFetch(endpoint, options = {}) {
  const { skipAuthRedirect = false, ...fetchOptions } = options;
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(fetchOptions.headers || {}),
  };

  let res;
  try {
    res = await fetch(`${API_BASE}${endpoint}`, { ...fetchOptions, headers });
  } catch (err) {
    throw new Error(err?.message || "Failed to fetch");
  }

  if (res.status === 401) {
    const shouldRedirect = !skipAuthRedirect && endpoint === "/auth/me";
    if (shouldRedirect) {
      clearToken();
      showAuthView();
      throw new Error("Session expired. Please sign in again.");
    }
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message || `API error ${res.status}`);
  }
  return res.json();
}
