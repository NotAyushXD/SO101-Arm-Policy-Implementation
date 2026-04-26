"""
Web UI served at GET / on the inference server.

Renders:
  - Live camera feeds (refreshed at ~5 fps via websocket; lower than control
    loop fps to keep the UI from hogging bandwidth).
  - Predicted action trajectory overlay (when in visualize/live modes).
  - Latency histogram (last 100 requests).
  - Success rate counter (set by client when it labels a trial).
  - Mode toggle controls (changes EXEC_MODE at runtime — but the client has
    to opt in by polling /control/state).
  - Checkpoint switcher (calls /policy/reload to swap revisions live).

All in one HTML file with no build step. Vanilla JS + minimal CSS.
"""
from __future__ import annotations

# This is the literal HTML/CSS/JS shipped to the browser. It's a string
# constant rather than a separate file because it makes the server a
# single-file deploy.

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SO-101 Remote Inference</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0d1117; color: #e6edf3;
      margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto;
    }
    h1 { font-size: 18px; font-weight: 500; margin: 0 0 16px; }
    h2 { font-size: 14px; font-weight: 500; margin: 16px 0 8px; color: #8b949e; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 12px;
    }
    .stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
    .stat span:first-child { color: #8b949e; }
    .stat span:last-child { font-family: ui-monospace, Menlo, monospace; }
    img.cam { width: 100%; border-radius: 4px; background: #000; }
    .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
    button, select {
      background: #21262d; color: #e6edf3; border: 1px solid #30363d;
      border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer;
    }
    button:hover, select:hover { background: #30363d; }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 10px;
      font-size: 11px; font-weight: 500;
    }
    .badge.shadow { background: #1f2937; color: #9ca3af; }
    .badge.visualize { background: #1e3a5f; color: #60a5fa; }
    .badge.live_slow { background: #5c4400; color: #fbbf24; }
    .badge.live { background: #5b1d1d; color: #f87171; }
    canvas#latency { width: 100%; height: 80px; }
  </style>
</head>
<body>
  <h1>SO-101 Remote Inference Server</h1>

  <div class="grid">
    <div class="panel">
      <h2>Status</h2>
      <div class="stat"><span>Policy</span><span id="policy-repo">—</span></div>
      <div class="stat"><span>Revision</span><span id="policy-rev">—</span></div>
      <div class="stat"><span>GPU</span><span id="gpu">—</span></div>
      <div class="stat"><span>Uptime</span><span id="uptime">—</span></div>
      <div class="stat"><span>Requests served</span><span id="req-count">0</span></div>
      <div class="stat"><span>Mode</span><span id="mode-badge"><span class="badge shadow">—</span></span></div>
      <div class="stat"><span>Last success rate</span><span id="success-rate">—</span></div>
    </div>

    <div class="panel">
      <h2>Latency (last 100 requests)</h2>
      <canvas id="latency"></canvas>
      <div class="stat"><span>Inference (median)</span><span id="lat-inf">— ms</span></div>
      <div class="stat"><span>Decode (median)</span><span id="lat-dec">— ms</span></div>
      <div class="stat"><span>Total RTT (median, from client)</span><span id="lat-rtt">— ms</span></div>
    </div>
  </div>

  <div class="panel" style="margin-top: 16px;">
    <h2>Controls</h2>
    <div class="row">
      <span style="color:#8b949e; font-size: 12px;">Execution mode (advisory; client must respect):</span>
      <select id="mode-sel">
        <option value="shadow">shadow</option>
        <option value="visualize">visualize</option>
        <option value="live_slow">live_slow</option>
        <option value="live">live</option>
      </select>
      <button onclick="setMode()">Apply</button>
    </div>
    <div class="row">
      <span style="color:#8b949e; font-size: 12px;">Reload policy with revision:</span>
      <input id="rev-input" type="text" placeholder="main / step-50000 / commit-sha"
             style="background:#21262d; color:#e6edf3; border:1px solid #30363d;
                    border-radius:4px; padding:4px 8px; font-size:12px; flex:1;">
      <button onclick="reloadPolicy()">Reload</button>
    </div>
    <div class="row">
      <span style="color:#8b949e; font-size: 12px;">Swap policy type:</span>
      <select id="swap-type-sel"></select>
      <input id="swap-repo" type="text" placeholder="repo (blank = use type name)"
             style="background:#21262d; color:#e6edf3; border:1px solid #30363d;
                    border-radius:4px; padding:4px 8px; font-size:12px; flex:1;">
      <button onclick="swapPolicy()">Swap</button>
    </div>
  </div>

  <div class="grid" style="margin-top: 16px;">
    <div class="panel">
      <h2>Wrist camera</h2>
      <img class="cam" id="cam-wrist" alt="wrist">
    </div>
    <div class="panel">
      <h2>Front camera</h2>
      <img class="cam" id="cam-front" alt="front">
    </div>
  </div>

  <script>
    // Polls /healthz and /telemetry every second. Cheap and reliable —
    // websocket would be lower-latency but adds complexity for marginal UX gain.
    async function refresh() {
      try {
        const h = await fetch('/healthz').then(r => r.json());
        document.getElementById('policy-repo').textContent = h.policy_repo;
        document.getElementById('policy-rev').textContent = h.policy_revision +
          (h.policy_commit_sha ? ' (' + h.policy_commit_sha.slice(0,7) + ')' : '');
        document.getElementById('gpu').textContent = h.gpu_name || 'cpu';
        document.getElementById('uptime').textContent = formatUptime(h.server_uptime_s);
        document.getElementById('req-count').textContent = h.requests_served;
      } catch (e) { /* server may be reloading */ }

      try {
        const t = await fetch('/telemetry').then(r => r.json());
        document.getElementById('lat-inf').textContent = fmt(t.inference_ms_p50) + ' ms';
        document.getElementById('lat-dec').textContent = fmt(t.decode_ms_p50) + ' ms';
        document.getElementById('lat-rtt').textContent = fmt(t.client_rtt_ms_p50) + ' ms';
        document.getElementById('success-rate').textContent =
          t.last_success_rate != null ? (t.last_success_rate * 100).toFixed(0) + '%' : '—';
        const modeEl = document.getElementById('mode-badge');
        modeEl.innerHTML = '<span class="badge ' + t.advisory_mode + '">' + t.advisory_mode + '</span>';
        drawLatencyHist(t.recent_rtt_ms || []);
        if (t.available_policy_types) populatePolicyTypes(t.available_policy_types);
      } catch (e) {}

      // Cameras: bust cache so the <img> actually re-fetches.
      const ts = Date.now();
      document.getElementById('cam-wrist').src = '/last-frame/wrist?t=' + ts;
      document.getElementById('cam-front').src = '/last-frame/front?t=' + ts;
    }

    function fmt(v) { return v == null ? '—' : v.toFixed(1); }
    function formatUptime(s) {
      const m = Math.floor(s / 60), sec = Math.floor(s % 60);
      return m + 'm ' + sec + 's';
    }

    function drawLatencyHist(samples) {
      const c = document.getElementById('latency');
      const ctx = c.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
      ctx.scale(dpr, dpr);
      const W = c.clientWidth, H = c.clientHeight;
      ctx.clearRect(0, 0, W, H);
      if (!samples.length) return;
      const max = Math.max(...samples, 100);
      const barW = W / samples.length;
      ctx.fillStyle = '#3fb950';
      samples.forEach((v, i) => {
        const h = (v / max) * (H - 4);
        ctx.fillRect(i * barW, H - h, barW * 0.8, h);
      });
    }

    async function setMode() {
      const m = document.getElementById('mode-sel').value;
      await fetch('/control/mode', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: m})});
      refresh();
    }

    async function reloadPolicy() {
      const rev = document.getElementById('rev-input').value.trim();
      if (!rev) return;
      const btn = event.target;
      btn.disabled = true; btn.textContent = 'Reloading...';
      try {
        const r = await fetch('/policy/reload', {method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({revision: rev})});
        const j = await r.json();
        if (!r.ok) alert('Reload failed: ' + (j.detail || 'unknown'));
      } finally {
        btn.disabled = false; btn.textContent = 'Reload';
        refresh();
      }
    }

    async function swapPolicy() {
      const policy_type = document.getElementById('swap-type-sel').value;
      const repo = document.getElementById('swap-repo').value.trim() || null;
      const btn = event.target;
      btn.disabled = true; btn.textContent = 'Swapping...';
      try {
        const r = await fetch('/policy/swap', {method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({policy_type, repo, revision: 'main'})});
        const j = await r.json();
        if (!r.ok) alert('Swap failed: ' + (j.detail || 'unknown'));
      } finally {
        btn.disabled = false; btn.textContent = 'Swap';
        refresh();
      }
    }

    function populatePolicyTypes(types) {
      const sel = document.getElementById('swap-type-sel');
      const current = sel.value;
      if (sel.dataset.populated === JSON.stringify(types)) return;
      sel.innerHTML = '';
      for (const t of types) {
        const opt = document.createElement('option');
        opt.value = t; opt.textContent = t;
        sel.appendChild(opt);
      }
      if (current && types.includes(current)) sel.value = current;
      sel.dataset.populated = JSON.stringify(types);
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
