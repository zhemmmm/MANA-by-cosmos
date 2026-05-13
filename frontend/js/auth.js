/**
 * MANA — Auth Module
 * Handles login, logout, session memory, captcha, and password-toggle UI.
 *
 * API endpoints (backend must implement):
 *   POST /api/auth/login           { username, password, remember } → { token, user }
 *   POST /api/auth/logout          → { success }
 *   GET  /api/auth/me              → { username, role, email }
 *   PATCH /api/auth/me             { username?, role? } → updated user
 *   POST /api/auth/change-password { current_password, new_password } → { success }
 *   POST /api/auth/request-email-change  { new_email } → { success }
 *   POST /api/auth/verify-email-change   { new_email, code } → { success, email }
 */

// ─── API Calls ────────────────────────────────────────────────────────────────
async function apiLogin(username, password, remember) {
  const data = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password, remember }),
  });
  if (data.token) setToken(data.token);
  return data;
}

async function apiLogout() {
  await apiFetch("/auth/logout", { method: "POST" }).catch(() => {});
  clearToken();
}

async function apiGetProfile()      { return apiFetch("/auth/me", { skipAuthRedirect: true }); }
async function apiUpdateProfile(u)  { return apiFetch("/auth/me", { method: "PATCH", body: JSON.stringify(u) }); }
async function apiChangePassword(current, next) {
  return apiFetch("/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: current, new_password: next, confirm_password: next }),
  });
}
async function apiRequestEmailChange(newEmail) {
  return apiFetch("/auth/request-email-change", { method: "POST", body: JSON.stringify({ new_email: newEmail }) });
}
async function apiVerifyEmailChange(newEmail, code) {
  return apiFetch("/auth/verify-email-change", { method: "POST", body: JSON.stringify({ new_email: newEmail, code }) });
}

// ─── DataService shim (used by main.js) ──────────────────────────────────────
const AuthService = {
  async login(identity, password, remember) {
    const data = await apiLogin(identity, password, remember);
    if (data.token) {
      setToken(data.token);
      if (data.user?.role === "Admin") localStorage.setItem("mana-admin-token", data.token);
    }
    return { _uid: data.user.username, user: data.user };
  },

  async logout() {
    await apiLogout();
    clearToken();
    localStorage.removeItem("mana-admin-token");
  },

  async getProfile() {
    if (USE_MOCK) return state.profile;
    return apiGetProfile();
  },

  async updateProfile(updates) {
    if (USE_MOCK) return { ...state.profile, ...updates };
    return apiUpdateProfile(updates);
  },

  async changePassword(current, next) {
    if (USE_MOCK) return { success: true };
    return apiChangePassword(current, next);
  },

  async requestEmailChange(newEmail) {
    if (USE_MOCK) return { success: true };
    return apiRequestEmailChange(newEmail);
  },

  async verifyEmailChange(newEmail, code) {
    if (USE_MOCK) {
      if (code !== "246810") throw new Error("Wrong code");
      return { success: true, email: newEmail };
    }
    return apiVerifyEmailChange(newEmail, code);
  },
};

// ─── Login Handler ────────────────────────────────────────────────────────────
async function handleLogin(event) {
  event.preventDefault();
  const identity = document.getElementById("loginIdentity").value.trim();
  const password = document.getElementById("loginPassword").value.trim();
  const captchaInput = document.getElementById("captchaInput").value.trim().toUpperCase();
  const remember = document.getElementById("rememberSession").checked;
  const submitBtn = document.querySelector("#loginForm button[type='submit']");
  const originalLabel = submitBtn?.textContent;

  if (!identity || !password) {
    showToast("Sign-in incomplete", "Enter both username/email and password to continue.");
    return;
  }
  if (captchaInput !== state.currentCaptcha) {
    showToast("CAPTCHA mismatch", "The verification code does not match. Please try again.");
    generateCaptcha();
    document.getElementById("captchaInput").value = "";
    return;
  }

  try {
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Signing in...";
    }

    const result = await AuthService.login(identity, password, remember);
    state.profile = result.user || state.profile;

    if (state.profile.role === "Admin") {
      window.location.href = "admin/index.html";
      return;
    }

    sessionStorage.setItem("mana-active-session", "true");
    if (remember) {
      localStorage.setItem("mana-session", JSON.stringify({ expiresAt: Date.now() + 30 * 24 * 60 * 60 * 1000 }));
    } else {
      localStorage.removeItem("mana-session");
    }

    renderProfileSettings();
    showApp();
    renderAll();
    loadCriticalAppData()
      .then(() => {
        startLiveUpdates();
        return loadDeferredAppData();
      })
      .catch(() => {});
    showToast("Authenticated", "MANA is now ready for dashboard monitoring.");
  } catch (err) {
    showToast("Sign-in failed", err.message || "Invalid credentials. Please try again.");
    console.error("Login error:", err);
    generateCaptcha();
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = originalLabel || "Secure Sign In";
    }
  }
}

async function doLogout() {
  await AuthService.logout().catch(() => {});
  localStorage.removeItem("mana-session");
  sessionStorage.removeItem("mana-active-session");
  stopLiveUpdates();
  resetDeferredState();
  showAuthView();
  document.body.classList.remove("sidebar-open", "sidebar-collapsed");
  showToast("Logged out", "You have been signed out of MANA.");
}

async function checkRememberedSession() {
  const savedSession = localStorage.getItem("mana-session");
  const hasActiveTabSession = sessionStorage.getItem("mana-active-session") === "true";

  if (!savedSession && !hasActiveTabSession) {
    showAuthView();
    return;
  }

  if (savedSession) {
    try {
      const parsedSession = JSON.parse(savedSession);
      if (!parsedSession?.expiresAt || Number(parsedSession.expiresAt) <= Date.now()) {
        localStorage.removeItem("mana-session");
        if (!hasActiveTabSession) {
          clearToken();
          localStorage.removeItem("mana-admin-token");
          showAuthView();
          return;
        }
      }
    } catch (_) {
      localStorage.removeItem("mana-session");
      if (!hasActiveTabSession) {
        clearToken();
        localStorage.removeItem("mana-admin-token");
        showAuthView();
        return;
      }
    }
  }

  const sessionToken = getToken();
  if (!sessionToken) {
    localStorage.removeItem("mana-session");
    localStorage.removeItem("mana-admin-token");
    sessionStorage.removeItem("mana-active-session");
    showAuthView();
    return;
  }
  try {
    const profile = await AuthService.getProfile();
    if (getToken() !== sessionToken) return;
    state.profile = {
      username: profile.username,
      name: profile.name || profile.username,
      role: profile.role,
      email: profile.email,
    };
    if (profile.role === "Admin") {
      localStorage.setItem("mana-admin-token", getToken());
      window.location.href = "admin/index.html";
      return;
    }
    renderProfileSettings();
    showApp();
    renderAll();
    loadCriticalAppData()
      .then(() => {
        startLiveUpdates();
        return loadDeferredAppData();
      })
      .catch(() => {});
  } catch (err) {
    if (getToken() !== sessionToken) return;
    localStorage.removeItem("mana-session");
    sessionStorage.removeItem("mana-active-session");
    clearToken();
    localStorage.removeItem("mana-admin-token");
    showAuthView();
  }
}

// ─── Profile & Password UI ────────────────────────────────────────────────────
async function saveProfile() {
  const username = document.getElementById("profileUsername").value.trim() || state.profile.username;
  try {
    const profile = await AuthService.updateProfile({ name: username });
    state.profile = { ...state.profile, ...profile };
    persistLocalPreferences();
    renderProfileSettings();
    showToast("Profile saved", "Profile details were updated.");
  } catch (err) {
    showToast("Save failed", err.message || "Could not save profile.");
  }
}

async function savePassword() {
  const current = document.getElementById("currentPassword").value.trim();
  const next    = document.getElementById("newPassword").value.trim();
  const confirm = document.getElementById("confirmPassword").value.trim();
  if (!current || !next || !confirm) { showToast("Incomplete", "Fill in all password fields before saving."); return; }
  if (next !== confirm) { showToast("Password mismatch", "New password and confirmation do not match."); return; }
  if (next.length < 8) { showToast("Password too short", "New password must be at least 8 characters."); return; }
  try {
    await apiFetch("/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ current_password: current, new_password: next, confirm_password: confirm }),
    });
    showToast("Password saved", "Your password has been updated.");
    clearPasswordFields();
  } catch (err) {
    showToast("Password update failed", err.message || "Incorrect current password or server error.");
  }
}

function toggleEmailPopover() {
  const popover = document.getElementById("emailPopover");
  popover.classList.toggle("hidden");
  if (!popover.classList.contains("hidden")) {
    AuthService.requestEmailChange(document.getElementById("newEmailInput").value).catch(() => {});
    if (USE_MOCK) showToast("Verification code sent", "Use code 246810 to verify the new email in this prototype.");
  }
}

function closeEmailPopover() { document.getElementById("emailPopover").classList.add("hidden"); }

async function verifyEmailChange() {
  const newEmail = document.getElementById("newEmailInput").value.trim();
  const code     = document.getElementById("emailCodeInput").value.trim();
  if (!newEmail) { showToast("New email required", "Enter the new email address first."); return; }
  try {
    await AuthService.verifyEmailChange(newEmail, code);
    state.profile.email = newEmail;
    document.getElementById("profileEmail").value = newEmail;
    document.getElementById("emailPopover").classList.add("hidden");
    document.getElementById("newEmailInput").value = "";
    document.getElementById("emailCodeInput").value = "";
    persistLocalPreferences();
    showToast("Email verified", "The new email address was saved.");
  } catch (err) {
    showToast("Wrong code", err.message || "The verification code is not correct.");
  }
}

// ─── Captcha ──────────────────────────────────────────────────────────────────
function generateCaptcha() {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < 5; i++) code += chars[Math.floor(Math.random() * chars.length)];
  state.currentCaptcha = code;
  drawCaptcha(code);
}

function drawCaptcha(code) {
  const canvas = document.getElementById("captchaCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;

  // Background
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, "#0d1526");
  bg.addColorStop(1, "#121d33");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Noise dots
  for (let i = 0; i < 140; i++) {
    ctx.fillStyle = `rgba(148,163,184,${0.1 + Math.random() * 0.25})`;
    ctx.fillRect(Math.random() * W, Math.random() * H, Math.random() * 2.5, Math.random() * 2.5);
  }

  // Interference bezier curves
  const curveColors = [
    "rgba(56,189,248,0.15)", "rgba(129,140,248,0.13)",
    "rgba(52,211,153,0.12)", "rgba(251,191,36,0.10)"
  ];
  for (let i = 0; i < 4; i++) {
    ctx.strokeStyle = curveColors[i];
    ctx.lineWidth = 1 + Math.random();
    ctx.beginPath();
    ctx.moveTo(0, Math.random() * H);
    ctx.bezierCurveTo(W * 0.25, Math.random() * H, W * 0.75, Math.random() * H, W, Math.random() * H);
    ctx.stroke();
  }

  // Letters — each randomly rotated, offset, and colored
  const palette = ["#38bdf8", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#fbbf24"];
  const slotW = W / code.length;
  for (let i = 0; i < code.length; i++) {
    ctx.save();
    ctx.translate(slotW * i + slotW / 2, H / 2 + (Math.random() * 10 - 5));
    ctx.rotate((Math.random() - 0.5) * 0.6);
    ctx.font = `bold ${20 + Math.random() * 7}px 'Courier New', monospace`;
    ctx.fillStyle = palette[Math.floor(Math.random() * palette.length)];
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 5;
    ctx.fillText(code[i], 0, 0);
    ctx.restore();
  }
}

// ─── Misc UI ──────────────────────────────────────────────────────────────────
function showApp()      { document.getElementById("authView").classList.add("hidden"); document.getElementById("appView").classList.remove("hidden"); }
function showAuthView() { document.getElementById("appView").classList.add("hidden"); document.getElementById("authView").classList.remove("hidden"); }

function togglePasswordField(inputId) {
  const field = document.getElementById(inputId);
  if (field) field.type = field.type === "password" ? "text" : "password";
}

function clearPasswordFields() {
  ["currentPassword", "newPassword", "confirmPassword"].forEach(id => {
    const f = document.getElementById(id);
    f.value = ""; f.type = "password";
  });
}
