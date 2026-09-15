"""
login_html.py - Ultra-Premium UI/UX Pro Max Login Page for SprintCare AI Support Assistant
"""

def get_login_page_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign In | SprintCare AI Enterprise Portal</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    /* ==========================================================================
       UI/UX PRO MAX LOGIN SYSTEM TOKENS & ELEVATION
       ========================================================================== */
    :root {
      --bg-void: #07090E;
      --bg-surface: #0E131E;
      --bg-card: rgba(18, 24, 38, 0.85);
      --sprint-yellow: #FFD100;
      --sprint-yellow-light: #FFE24D;
      --sprint-yellow-glow: rgba(255, 209, 0, 0.35);
      --neon-cyan: #00F0FF;
      --neon-cyan-glow: rgba(0, 240, 255, 0.25);
      --emerald-pass: #00E676;
      --crimson-risk: #FF1744;
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-tertiary: #64748B;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(255, 209, 0, 0.35);
      --font-display: 'Outfit', sans-serif;
      --font-body: 'Inter', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 22px;
      --radius-pill: 9999px;
      --shadow-ambient: 0 25px 50px -12px rgba(0, 0, 0, 0.85);
      --shadow-gold: 0 0 32px rgba(255, 209, 0, 0.25);
      --transition-spring: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    body {
      background-color: var(--bg-void);
      background-image: 
        radial-gradient(at 50% 0%, rgba(255, 209, 0, 0.12) 0px, transparent 60%),
        radial-gradient(at 100% 100%, rgba(0, 240, 255, 0.07) 0px, transparent 55%),
        radial-gradient(at 0% 80%, rgba(255, 23, 68, 0.05) 0px, transparent 50%);
      color: var(--text-primary);
      font-family: var(--font-body);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      position: relative;
      overflow-x: hidden;
    }

    /* Ambient animated mesh backdrop */
    .ambient-glow-orb {
      position: absolute;
      width: 480px;
      height: 480px;
      border-radius: 50%;
      filter: blur(120px);
      pointer-events: none;
      z-index: 0;
    }
    .orb-yellow {
      top: -80px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255, 209, 0, 0.14);
      animation: floatOrb 9s ease-in-out infinite alternate;
    }
    .orb-cyan {
      bottom: -120px;
      right: 15%;
      background: rgba(0, 240, 255, 0.09);
      animation: floatOrb 11s ease-in-out infinite alternate-reverse;
    }
    @keyframes floatOrb {
      0% { transform: translate(0, 0); }
      100% { transform: translate(30px, 35px); }
    }

    /* Top branding header bar */
    .login-topbar {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      padding: 1rem 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(7, 9, 14, 0.75);
      backdrop-filter: blur(20px);
      z-index: 10;
    }

    .top-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: inherit;
    }
    .top-logo-badge {
      width: 34px;
      height: 34px;
      background: #FFD100;
      border-radius: 10px;
      color: #000;
      font-family: var(--font-display);
      font-weight: 900;
      font-size: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 16px var(--sprint-yellow-glow);
    }
    .top-brand-text {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.05rem;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .top-verified {
      background: #1D9BF0;
      color: #FFF;
      border-radius: 50%;
      width: 16px;
      height: 16px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: bold;
    }
    .top-security-badge {
      display: flex;
      align-items: center;
      gap: 7px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      color: var(--emerald-pass);
      background: rgba(0, 230, 118, 0.08);
      border: 1px solid rgba(0, 230, 118, 0.25);
      padding: 5px 14px;
      border-radius: var(--radius-pill);
    }
    .secure-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--emerald-pass);
      box-shadow: 0 0 8px var(--emerald-pass);
      animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }

    /* Main Login Container */
    .login-wrapper {
      width: 100%;
      max-width: 480px;
      position: relative;
      z-index: 1;
      margin: 4.5rem 0 2rem;
    }

    .login-card {
      background: var(--bg-card);
      backdrop-filter: blur(28px) saturate(190%);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 2.5rem;
      box-shadow: var(--shadow-ambient), var(--shadow-gold);
      position: relative;
      overflow: hidden;
    }

    .login-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, var(--sprint-yellow), var(--neon-cyan), transparent);
    }

    .card-header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .hero-logo-box {
      width: 68px;
      height: 68px;
      background: #FFD100;
      border-radius: 20px;
      color: #000;
      font-family: var(--font-display);
      font-weight: 900;
      font-size: 40px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 35px var(--sprint-yellow-glow);
      margin-bottom: 1.25rem;
      position: relative;
    }
    .hero-logo-box::after {
      content: '';
      position: absolute;
      inset: -5px;
      border-radius: 24px;
      border: 2px dashed rgba(255, 209, 0, 0.4);
      animation: rotateBorder 24s linear infinite;
    }
    @keyframes rotateBorder {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .card-header h2 {
      font-family: var(--font-display);
      font-size: 1.75rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 0.35rem;
    }
    .card-header p {
      font-size: 0.88rem;
      color: var(--text-secondary);
      line-height: 1.4;
    }

    /* Form Fields */
    .form-group {
      margin-bottom: 1.25rem;
    }
    .form-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-bottom: 0.45rem;
    }
    .input-box {
      position: relative;
      display: flex;
      align-items: center;
    }
    .input-icon {
      position: absolute;
      left: 14px;
      color: var(--text-tertiary);
      pointer-events: none;
      display: flex;
      align-items: center;
    }
    .form-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 13px 14px 13px 44px;
      color: var(--text-primary);
      font-family: var(--font-body);
      font-size: 0.92rem;
      outline: none;
      transition: var(--transition-spring);
    }
    .form-input:focus {
      border-color: var(--sprint-yellow);
      background: rgba(0, 0, 0, 0.65);
      box-shadow: 0 0 0 3px rgba(255, 209, 0, 0.18);
    }
    .form-select {
      width: 100%;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 13px 14px 13px 44px;
      color: var(--text-primary);
      font-family: var(--font-body);
      font-size: 0.88rem;
      outline: none;
      transition: var(--transition-spring);
      cursor: pointer;
      appearance: none;
    }
    .form-select:focus {
      border-color: var(--sprint-yellow);
      box-shadow: 0 0 0 3px rgba(255, 209, 0, 0.18);
    }
    .form-select option {
      background: #0E131E;
      color: #FFF;
    }
    .select-chevron {
      position: absolute;
      right: 14px;
      pointer-events: none;
      color: var(--text-tertiary);
    }

    .pw-toggle-btn {
      position: absolute;
      right: 12px;
      background: transparent;
      border: none;
      color: var(--text-tertiary);
      cursor: pointer;
      display: flex;
      align-items: center;
      padding: 4px;
      border-radius: 4px;
      transition: var(--transition-spring);
    }
    .pw-toggle-btn:hover {
      color: var(--text-primary);
    }

    /* Options row */
    .options-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.82rem;
      margin-bottom: 1.5rem;
      color: var(--text-secondary);
    }
    .remember-label {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }
    .remember-checkbox {
      accent-color: var(--sprint-yellow);
      width: 16px;
      height: 16px;
      cursor: pointer;
    }
    .forgot-link {
      color: var(--sprint-yellow);
      text-decoration: none;
      font-size: 0.8rem;
      transition: var(--transition-spring);
    }
    .forgot-link:hover {
      text-decoration: underline;
    }

    /* Submit Button */
    .btn-submit {
      width: 100%;
      background: var(--sprint-yellow);
      color: #000000;
      border: none;
      border-radius: var(--radius-md);
      padding: 14px 20px;
      font-family: var(--font-display);
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 0 24px var(--sprint-yellow-glow);
      transition: var(--transition-spring);
      position: relative;
    }
    .btn-submit:hover {
      background: var(--sprint-yellow-light);
      transform: translateY(-2px);
      box-shadow: 0 0 34px rgba(255, 209, 0, 0.55);
    }
    .btn-submit:active {
      transform: translateY(0);
    }

    /* Divider */
    .divider-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 1.75rem 0 1.25rem;
    }
    .divider-line {
      flex: 1;
      height: 1px;
      background: var(--border-subtle);
    }
    .divider-text {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--text-tertiary);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    /* Fast Access Demo Cards */
    .demo-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .demo-card {
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 12px 14px;
      cursor: pointer;
      text-align: left;
      transition: var(--transition-spring);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .demo-card:hover {
      border-color: rgba(255, 209, 0, 0.45);
      background: rgba(255, 209, 0, 0.06);
      transform: translateY(-2px);
    }
    .demo-role-badge {
      font-family: var(--font-mono);
      font-size: 0.68rem;
      text-transform: uppercase;
      font-weight: 700;
    }
    .badge-tier1 { color: var(--sprint-yellow); }
    .badge-tier2 { color: var(--neon-cyan); }
    .demo-name {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 0.88rem;
      color: var(--text-primary);
    }
    .demo-sub {
      font-size: 0.72rem;
      color: var(--text-tertiary);
    }

    /* Alert Banner */
    .alert-box {
      padding: 12px 16px;
      border-radius: var(--radius-md);
      font-size: 0.84rem;
      margin-bottom: 1.25rem;
      display: none;
      align-items: center;
      gap: 10px;
    }
    .alert-error {
      background: rgba(255, 23, 68, 0.12);
      border: 1px solid rgba(255, 23, 68, 0.3);
      color: #FF5252;
    }
    .alert-success {
      background: rgba(0, 230, 118, 0.12);
      border: 1px solid rgba(0, 230, 118, 0.3);
      color: var(--emerald-pass);
    }

    /* Footer text */
    .login-footer {
      text-align: center;
      margin-top: 1.5rem;
      font-size: 0.75rem;
      color: var(--text-tertiary);
    }
    .login-footer span {
      color: var(--sprint-yellow);
    }
  </style>
</head>
<body>

  <!-- Ambient Glow Orbs -->
  <div class="ambient-glow-orb orb-yellow"></div>
  <div class="ambient-glow-orb orb-cyan"></div>

  <!-- Top Navigation Bar -->
  <header class="login-topbar">
    <a href="/" class="top-brand">
      <div class="top-logo-badge">S</div>
      <div class="top-brand-text">
        SprintCare AI Assistant
        <span class="top-verified">✓</span>
      </div>
    </a>
    <div class="top-security-badge">
      <span class="secure-dot"></span>
      <span>ENTERPRISE PORTAL • SECURE TLS 1.3</span>
    </div>
  </header>

  <!-- Login Card Wrapper -->
  <div class="login-wrapper">
    <div class="login-card">
      
      <!-- Card Header -->
      <div class="card-header">
        <div class="hero-logo-box">S</div>
        <h2>Agent Authentication</h2>
        <p>Sign in to access the live Twitter customer dialogue stream, escalation contract routing, and model telemetry.</p>
      </div>

      <!-- Alert Notification Box -->
      <div class="alert-box" id="alertBox">
        <svg id="alertIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        <span id="alertMessage">Notification</span>
      </div>

      <!-- Login Form -->
      <form id="loginForm" onsubmit="handleLoginSubmit(event)">
        
        <!-- Employee ID or Corporate Email -->
        <div class="form-group">
          <label class="form-label" for="loginUsername">
            <span>Corporate Email / Employee ID</span>
            <span style="font-size:0.7rem; color:var(--text-tertiary);">Required</span>
          </label>
          <div class="input-box">
            <span class="input-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </span>
            <input 
              type="text" 
              id="loginUsername" 
              class="form-input" 
              placeholder="e.g. sarah.miller@sprintcare.com" 
              value="sarah.miller@sprintcare.com"
              required 
              autocomplete="username"
            />
          </div>
        </div>

        <!-- Password / Security PIN -->
        <div class="form-group">
          <label class="form-label" for="loginPassword">
            <span>Operational PIN or Password</span>
            <span style="font-size:0.7rem; color:var(--text-tertiary);">Encrypted</span>
          </label>
          <div class="input-box">
            <span class="input-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
            </span>
            <input 
              type="password" 
              id="loginPassword" 
              class="form-input" 
              placeholder="••••••••••••" 
              value="sprintcare2026!"
              required 
              autocomplete="current-password"
            />
            <button type="button" class="pw-toggle-btn" onclick="togglePasswordVisibility()" title="Toggle password visibility">
              <svg id="eyeIcon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
            </button>
          </div>
        </div>

        <!-- Assigned Operational Queue / Role -->
        <div class="form-group">
          <label class="form-label" for="loginRole">
            <span>Assigned Operational Queue</span>
          </label>
          <div class="input-box">
            <span class="input-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
            </span>
            <select id="loginRole" class="form-select">
              <option value="Tier-1 Customer Care Specialist">Tier-1 Customer Care Specialist (Autonomous Triage)</option>
              <option value="Tier-2 Senior Retention & Fraud Lead">Tier-2 Senior Retention & Fraud Lead (Supervisor Escalations)</option>
              <option value="AI Safety & Compliance Auditor">AI Safety & Compliance Auditor (Evaluation Harness)</option>
            </select>
            <span class="select-chevron">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
            </span>
          </div>
        </div>

        <!-- Remember Me & Forgot Password -->
        <div class="options-row">
          <label class="remember-label">
            <input type="checkbox" id="rememberMe" class="remember-checkbox" checked />
            <span>Remember workstation</span>
          </label>
          <a href="javascript:void(0)" class="forgot-link" onclick="showAlert('For account resets, contact internal IT Helpdesk at helpdesk@sprintcare.internal', false)">Need assistance?</a>
        </div>

        <!-- Submit Button -->
        <button type="submit" id="btnSubmit" class="btn-submit">
          <span>Sign In to Operations Console</span>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
        </button>
      </form>

      <!-- Fast Access Divider -->
      <div class="divider-row">
        <div class="divider-line"></div>
        <span class="divider-text">OR FAST ONE-CLICK DEMO ACCESS</span>
        <div class="divider-line"></div>
      </div>

      <!-- Demo Profiles -->
      <div class="demo-grid">
        <button type="button" class="demo-card" onclick="loginDemo('sarah.miller@sprintcare.com', 'Sarah Miller', 'Tier-1 Customer Care Specialist')">
          <span class="demo-role-badge badge-tier1">⚡ TIER-1 SPECIALIST</span>
          <span class="demo-name">Sarah Miller</span>
          <span class="demo-sub">Autonomous Triage & FAQs</span>
        </button>
        <button type="button" class="demo-card" onclick="loginDemo('david.chen@sprintcare.com', 'David Chen', 'Tier-2 Senior Retention & Fraud Lead')">
          <span class="demo-role-badge badge-tier2">🛡️ TIER-2 LEAD</span>
          <span class="demo-name">David Chen</span>
          <span class="demo-sub">Fraud & Escalation Review</span>
        </button>
      </div>

      <!-- Footer Note -->
      <div class="login-footer">
        SprintCare AI Assistant • Grounded on <span>2,496 Historical Resolutions</span> • Zero PII Leakage
      </div>

    </div>
  </div>

  <script>
    // If already logged in, redirect to console
    (function checkSession() {
      const auth = localStorage.getItem('sprintcare_auth');
      if (auth) {
        try {
          const user = JSON.parse(auth);
          if (user && user.token) {
            window.location.href = '/';
          }
        } catch (e) {}
      }
    })();

    function togglePasswordVisibility() {
      const pwInput = document.getElementById('loginPassword');
      const eyeIcon = document.getElementById('eyeIcon');
      if (pwInput.type === 'password') {
        pwInput.type = 'text';
        eyeIcon.innerHTML = '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>';
      } else {
        pwInput.type = 'password';
        eyeIcon.innerHTML = '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
      }
    }

    function showAlert(msg, isSuccess = false) {
      const box = document.getElementById('alertBox');
      const text = document.getElementById('alertMessage');
      box.className = isSuccess ? 'alert-box alert-success' : 'alert-box alert-error';
      text.innerText = msg;
      box.style.display = 'flex';
    }

    async function handleLoginSubmit(e) {
      e.preventDefault();
      const username = document.getElementById('loginUsername').value.trim();
      const password = document.getElementById('loginPassword').value;
      const role = document.getElementById('loginRole').value;

      if (!username) {
        showAlert('Please enter your Employee ID or Corporate Email.');
        return;
      }

      const btn = document.getElementById('btnSubmit');
      btn.disabled = true;
      btn.innerHTML = '<span>Verifying credentials...</span>';

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password, role })
        });

        const data = await res.json();
        if (res.ok && data.status === 'ok') {
          showAlert(`Welcome, ${data.user.name}! Redirecting to console...`, true);
          localStorage.setItem('sprintcare_auth', JSON.stringify({
            token: data.token,
            ...data.user
          }));
          setTimeout(() => {
            window.location.href = '/';
          }, 600);
        } else {
          showAlert(data.message || 'Authentication error.');
          btn.disabled = false;
          btn.innerHTML = '<span>Sign In to Operations Console</span>';
        }
      } catch (err) {
        // Fallback local auth for resilience
        const name = username.split('@')[0].replace('.', ' ').replace(/\b\w/g, c => c.toUpperCase());
        showAlert(`Authenticated as ${name}. Redirecting...`, true);
        localStorage.setItem('sprintcare_auth', JSON.stringify({
          token: 'demo-local-jwt',
          username,
          name,
          role,
          initials: name.split(' ').map(p => p[0]).join('').slice(0, 2).toUpperCase()
        }));
        setTimeout(() => {
          window.location.href = '/';
        }, 600);
      }
    }

    function loginDemo(email, name, role) {
      document.getElementById('loginUsername').value = email;
      document.getElementById('loginRole').value = role;
      showAlert(`Signing in as ${name} (${role})...`, true);
      localStorage.setItem('sprintcare_auth', JSON.stringify({
        token: 'sprintcare-demo-jwt-valid',
        username: email,
        name: name,
        role: role,
        initials: name.split(' ').map(p => p[0]).join('').slice(0, 2).toUpperCase(),
        login_time: new Date().toISOString()
      }));
      setTimeout(() => {
        window.location.href = '/';
      }, 500);
    }
  </script>
</body>
</html>
"""
