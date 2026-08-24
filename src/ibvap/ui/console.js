/* IBVAP operator console.
 *
 * Deliberately dependency-free: no build step, no CDN, no framework. A BOP node
 * is often air-gapped or reachable only over a metered satellite link, and an
 * operator console that cannot load because a CDN is unreachable is worse than
 * no console at all. Everything here is served from the node itself.
 */
'use strict';

const API = '/api/v1';

const state = {
  token: null,
  refreshToken: null,
  user: null,
  role: null,
  cameras: [],
  alerts: [],
  selectedAlert: null,
  socket: null,
  reconnectDelay: 1000,
  timers: [],
};

/* --------------------------------------------------------------- helpers -- */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/** Escape untrusted text used in an attribute or template context. */
function safe(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function clockTime(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** Call the API, transparently refreshing an expired access token once. */
async function api(path, options = {}, retry = true) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json';

  const response = await fetch(API + path, { ...options, headers });

  if (response.status === 401 && retry && state.refreshToken) {
    if (await refreshSession()) return api(path, options, false);
    signOut();
    throw new Error('session expired');
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

async function refreshSession() {
  try {
    const response = await fetch(`${API}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: state.refreshToken }),
    });
    if (!response.ok) return false;
    const data = await response.json();
    state.token = data.access_token;
    sessionStorage.setItem('ibvap.token', data.access_token);
    return true;
  } catch { return false; }
}

/* ---------------------------------------------------------------- session -- */

$('login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const errorNode = $('login-error');
  errorNode.textContent = '';
  try {
    const response = await fetch(`${API}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: $('username').value, password: $('password').value }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || 'sign-in failed');
    const data = await response.json();
    startSession(data);
    if (data.must_change_password) {
      errorNode.textContent = 'This account must change its password soon.';
    }
  } catch (error) {
    errorNode.textContent = error.message;
  }
});

function startSession(data) {
  state.token = data.access_token;
  state.refreshToken = data.refresh_token;
  state.user = data.username;
  state.role = data.role;
  // sessionStorage, not localStorage: a shared console in a control room must
  // not leave a usable token behind for the next shift to find.
  sessionStorage.setItem('ibvap.token', data.access_token);
  sessionStorage.setItem('ibvap.refresh', data.refresh_token);
  sessionStorage.setItem('ibvap.user', data.username);
  sessionStorage.setItem('ibvap.role', data.role);

  $('login').hidden = true;
  $('app').hidden = false;
  $('user-name').textContent = `${data.username} (${data.role})`;
  boot();
}

function signOut() {
  state.timers.forEach(clearInterval);
  state.timers = [];
  if (state.socket) { state.socket.onclose = null; state.socket.close(); state.socket = null; }
  sessionStorage.clear();
  Object.assign(state, { token: null, refreshToken: null, user: null, role: null, alerts: [] });
  $('app').hidden = true;
  $('login').hidden = false;
}

$('logout').addEventListener('click', async () => {
  try { await api('/auth/logout', { method: 'POST' }); } catch { /* revoke is best effort */ }
  signOut();
});

/* ------------------------------------------------------------ navigation -- */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
    tab.classList.add('active');
    $(`view-${tab.dataset.view}`).classList.add('active');
    if (tab.dataset.view === 'watchlists') loadWatchlists();
    if (tab.dataset.view === 'system') loadSystem();
    if (tab.dataset.view === 'alerts') loadAlerts();
  });
});

/* ------------------------------------------------------------------ boot -- */

async function boot() {
  await Promise.all([loadCameras(), loadAlerts(), loadHealth()]);
  openAlertStream();
  // Health and camera status are polled; alerts arrive by WebSocket. Polling
  // status on a slow link is cheap, whereas polling for alerts would add
  // latency to exactly the thing that must be immediate.
  state.timers.push(setInterval(loadHealth, 10000));
  state.timers.push(setInterval(loadCameras, 15000));
  state.timers.push(setInterval(renderAlerts, 30000));
}

/* ----------------------------------------------------------- video wall -- */

async function loadCameras() {
  try {
    state.cameras = await api('/cameras');
  } catch { return; }
  renderWall();
  renderCameraTable();
  const online = state.cameras.filter((c) => c.stream.connected).length;
  $('wall-summary').textContent = `${online} of ${state.cameras.length} cameras online`;

  const filter = $('filter-camera');
  if (filter.options.length !== state.cameras.length + 1) {
    const current = filter.value;
    filter.innerHTML = '<option value="">all</option>';
    state.cameras.forEach((c) => {
      const option = el('option', null, c.camera.name || c.camera.id);
      option.value = c.camera.id;
      filter.appendChild(option);
    });
    filter.value = current;
  }
}

function renderWall() {
  const grid = $('camera-grid');
  const annotate = $('show-overlays').checked ? 1 : 0;

  state.cameras.forEach((cam) => {
    const id = cam.camera.id;
    let tile = grid.querySelector(`[data-camera="${CSS.escape(id)}"]`);
    if (!tile) {
      tile = el('div', 'camera-tile');
      tile.dataset.camera = id;
      tile.innerHTML = `
        <figure>
          <img alt="Live view from ${safe(cam.camera.name || id)}">
          <div class="offline" hidden>camera offline</div>
        </figure>
        <figcaption>
          <span class="cam-name"></span>
          <span class="pill pill-muted state"></span>
          <span class="cam-meta"></span>
        </figcaption>`;
      grid.appendChild(tile);
    }

    const image = tile.querySelector('img');
    const offline = tile.querySelector('.offline');
    const connected = cam.stream.connected;

    // Rebuild the MJPEG URL only when the feed actually changes state:
    // reassigning src restarts the stream, and doing that on every poll would
    // make every tile stutter every few seconds.
    const wanted = `${API}/cameras/${encodeURIComponent(id)}/stream.mjpeg?fps=4&annotate=${annotate}&token=${encodeURIComponent(state.token)}`;
    if (connected && image.dataset.src !== wanted) {
      image.dataset.src = wanted;
      image.src = wanted;
    }
    if (!connected) { image.removeAttribute('src'); delete image.dataset.src; }
    image.hidden = !connected;
    offline.hidden = connected;

    tile.querySelector('.cam-name').textContent = cam.camera.name || id;
    const stateNode = tile.querySelector('.state');
    stateNode.textContent = connected ? 'live' : 'offline';
    stateNode.className = `pill state ${connected ? 'pill-ok' : 'pill-bad'}`;
    tile.querySelector('.cam-meta').textContent =
      `${cam.stream.fps.toFixed(1)} fps · ${cam.stats.avg_latency_ms} ms`;
  });

  const known = new Set(state.cameras.map((c) => c.camera.id));
  grid.querySelectorAll('[data-camera]').forEach((tile) => {
    if (!known.has(tile.dataset.camera)) tile.remove();
  });
}

$('grid-size').addEventListener('change', (event) => {
  $('camera-grid').className = `camera-grid cols-${event.target.value}`;
});
$('show-overlays').addEventListener('change', () => {
  $('camera-grid').innerHTML = '';  // force the MJPEG streams to restart
  renderWall();
});

/* ---------------------------------------------------------------- alerts -- */

async function loadAlerts() {
  const params = new URLSearchParams({ limit: '150' });
  const severity = $('filter-severity').value;
  const camera = $('filter-camera').value;
  const search = $('filter-search').value.trim();
  if (severity) params.set('min_severity', severity);
  if (camera) params.append('camera_id', camera);
  if (search) params.set('search', search);
  if ($('filter-unack').checked) params.set('acknowledged', 'false');

  try {
    const page = await api(`/events?${params}`);
    state.alerts = page.events;
    renderAlerts();
  } catch (error) {
    console.error('alert load failed', error);
  }
}

['filter-severity', 'filter-camera', 'filter-unack'].forEach((id) =>
  $(id).addEventListener('change', loadAlerts));
$('refresh-alerts').addEventListener('click', loadAlerts);

let searchTimer;
$('filter-search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadAlerts, 350);
});

function renderAlerts() {
  const list = $('alert-list');
  list.innerHTML = '';
  if (!state.alerts.length) {
    list.appendChild(el('li', 'muted', 'No alerts match the current filter.'));
    return;
  }
  state.alerts.forEach((alert) => list.appendChild(alertRow(alert)));
}

function alertRow(alert, isNew = false) {
  const item = el('li', `alert-item${alert.acknowledged ? ' acked' : ''}${isNew ? ' new' : ''}`);
  item.dataset.id = alert.event_id;
  if (state.selectedAlert === alert.event_id) item.classList.add('selected');

  item.appendChild(el('div', `bar sev-${alert.severity}`));

  const body = el('div');
  const headline = el('div', 'headline');
  headline.appendChild(el('span', `sev sev-${alert.severity}`, alert.severity));
  headline.appendChild(el('span', 'type', alert.event_type));
  headline.appendChild(el('span', 'type', alert.camera_id));
  if (alert.acknowledged) headline.appendChild(el('span', 'type', '✓ acknowledged'));
  body.appendChild(headline);
  body.appendChild(el('span', 'msg', alert.message));
  item.appendChild(body);

  item.appendChild(el('span', 'when', clockTime(alert.timestamp)));
  item.addEventListener('click', () => selectAlert(alert.event_id));
  return item;
}

async function selectAlert(eventId) {
  state.selectedAlert = eventId;
  document.querySelectorAll('.alert-item').forEach((node) =>
    node.classList.toggle('selected', node.dataset.id === eventId));

  const panel = $('alert-detail');
  panel.innerHTML = '<p class="muted">Loading…</p>';
  let alert;
  try {
    alert = await api(`/events/${encodeURIComponent(eventId)}`);
  } catch (error) {
    panel.innerHTML = `<p class="error">${safe(error.message)}</p>`;
    return;
  }

  panel.innerHTML = '';
  panel.appendChild(el('h2', null, alert.message));

  if (alert.has_snapshot) {
    const image = el('img');
    image.alt = `Evidence snapshot for ${alert.event_type} on ${alert.camera_id}`;
    // The <img> cannot send an Authorization header, so the token travels as a
    // query parameter on this same-origin request.
    image.src = `${API}/events/${encodeURIComponent(eventId)}/snapshot?token=${encodeURIComponent(state.token)}`;
    panel.appendChild(image);
  }

  const list = el('dl');
  const rows = [
    ['Severity', alert.severity], ['Type', alert.event_type], ['Camera', alert.camera_id],
    ['Time', new Date(alert.timestamp * 1000).toLocaleString()],
    ['Confidence', alert.confidence.toFixed(2)], ['Rule', alert.rule_id || '—'],
    ['Zone', alert.zone_id || '—'], ['Tracks', alert.track_ids.join(', ') || '—'],
  ];
  Object.entries(alert.attributes || {}).forEach(([k, v]) =>
    rows.push([k.replace(/_/g, ' '), typeof v === 'object' ? JSON.stringify(v) : String(v)]));
  if (alert.acknowledged) {
    rows.push(['Acknowledged by', alert.acknowledged_by || '—']);
    rows.push(['Disposition', alert.disposition || '—']);
  }
  rows.forEach(([key, value]) => {
    list.appendChild(el('dt', null, key));
    list.appendChild(el('dd', null, value));
  });
  panel.appendChild(list);

  if (alert.has_snapshot) {
    const verify = el('button', 'ghost', 'Verify evidence integrity');
    verify.addEventListener('click', async () => {
      verify.disabled = true;
      try {
        const result = await api(`/events/${encodeURIComponent(eventId)}/verify`);
        verify.textContent = result.valid
          ? '✓ Evidence verified — unaltered'
          : `✗ ${result.issues.join('; ')}`;
        verify.className = result.valid ? 'ghost verify-ok' : 'ghost verify-bad';
      } catch (error) {
        verify.textContent = error.message;
        verify.className = 'ghost verify-bad';
      }
    });
    panel.appendChild(verify);
  }

  if (!alert.acknowledged && ['operator', 'supervisor', 'admin'].includes(state.role)) {
    const note = el('textarea');
    note.placeholder = 'Disposition — what action was taken?';
    note.setAttribute('aria-label', 'Disposition');
    const button = el('button', 'primary', 'Acknowledge');
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        await api(`/events/${encodeURIComponent(eventId)}/acknowledge`, {
          method: 'POST', body: JSON.stringify({ disposition: note.value }),
        });
        await loadAlerts();
        selectAlert(eventId);
      } catch (error) {
        button.disabled = false;
        panel.appendChild(el('p', 'error', error.message));
      }
    });
    panel.appendChild(note);
    panel.appendChild(button);
  }
}

/* --------------------------------------------------------- live alert feed -- */

function openAlertStream() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${scheme}://${location.host}${API}/events/stream?token=${encodeURIComponent(state.token)}`;
  const socket = new WebSocket(url);
  state.socket = socket;

  socket.onopen = () => {
    state.reconnectDelay = 1000;
    setLinkState('live', 'pill-ok');
  };
  socket.onmessage = (message) => {
    let frame;
    try { frame = JSON.parse(message.data); } catch { return; }
    if (frame.type === 'event') onLiveEvent(frame.event);
  };
  socket.onclose = () => {
    setLinkState('reconnecting', 'pill-warn');
    // Capped exponential backoff: a control room console left open overnight
    // must not hammer a node that is down for maintenance.
    setTimeout(() => { if (state.token) openAlertStream(); }, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 30000);
  };
  socket.onerror = () => setLinkState('link error', 'pill-bad');
}

function setLinkState(text, cls) {
  const node = $('link-state');
  node.textContent = text;
  node.className = `pill ${cls}`;
}

function onLiveEvent(payload) {
  const alert = {
    event_id: payload.event_id,
    camera_id: payload.camera.id,
    event_type: payload.event_type,
    severity: payload.severity,
    timestamp: payload.timestamp,
    message: payload.message,
    acknowledged: false,
  };
  state.alerts.unshift(alert);
  state.alerts = state.alerts.slice(0, 200);

  const list = $('alert-list');
  const first = list.firstElementChild;
  if (first && first.classList.contains('muted')) list.innerHTML = '';
  list.insertBefore(alertRow(alert, true), list.firstChild);

  const tile = $('camera-grid').querySelector(`[data-camera="${CSS.escape(alert.camera_id)}"]`);
  if (tile && ['high', 'critical'].includes(alert.severity)) {
    tile.classList.add('alerting');
    setTimeout(() => tile.classList.remove('alerting'), 5000);
  }
}

/* ------------------------------------------------------------ watchlists -- */

async function loadWatchlists() {
  try {
    const [plates, faces] = await Promise.all([
      api('/watchlists/plates'), api('/watchlists/faces'),
    ]);
    renderPlates(plates);
    renderFaces(faces);
  } catch (error) {
    $('plate-error').textContent = error.message;
  }
}

function renderPlates(plates) {
  const body = $('plate-rows');
  body.innerHTML = '';
  plates.forEach((plate) => {
    const row = el('tr');
    const code = el('td'); code.appendChild(el('code', null, plate.plate));
    row.appendChild(code);
    row.appendChild(el('td', null, plate.category));
    row.appendChild(el('td', null, plate.reference || '—'));
    row.appendChild(el('td', null, plate.created_by || '—'));
    const actions = el('td');
    const remove = el('button', 'danger', 'Remove');
    remove.addEventListener('click', async () => {
      if (!confirm(`Remove ${plate.plate} from the watchlist?`)) return;
      try {
        await api(`/watchlists/plates/${encodeURIComponent(plate.plate)}`, { method: 'DELETE' });
        loadWatchlists();
      } catch (error) { $('plate-error').textContent = error.message; }
    });
    actions.appendChild(remove);
    row.appendChild(actions);
    body.appendChild(row);
  });
}

function renderFaces(faces) {
  const body = $('face-rows');
  body.innerHTML = '';
  faces.forEach((face) => {
    const row = el('tr');
    const id = el('td'); id.appendChild(el('code', null, face.person_id));
    row.appendChild(id);
    row.appendChild(el('td', null, face.name || '—'));
    row.appendChild(el('td', null, face.category));
    row.appendChild(el('td', null, String(face.embedding_count)));
    row.appendChild(el('td', null, face.created_by || '—'));
    const actions = el('td');
    if (['supervisor', 'admin'].includes(state.role)) {
      const remove = el('button', 'danger', 'Remove');
      remove.addEventListener('click', async () => {
        if (!confirm(`Remove ${face.person_id} from the watchlist?`)) return;
        try {
          await api(`/watchlists/faces/${encodeURIComponent(face.person_id)}`, { method: 'DELETE' });
          loadWatchlists();
        } catch (error) { alert(error.message); }
      });
      actions.appendChild(remove);
    }
    row.appendChild(actions);
    body.appendChild(row);
  });
}

$('plate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('plate-error').textContent = '';
  try {
    await api('/watchlists/plates', {
      method: 'POST',
      body: JSON.stringify({
        plate: $('plate-value').value,
        category: $('plate-category').value,
        reference: $('plate-reference').value,
      }),
    });
    $('plate-value').value = '';
    $('plate-reference').value = '';
    loadWatchlists();
  } catch (error) {
    $('plate-error').textContent = error.message;
  }
});

/* ---------------------------------------------------------------- system -- */

/** Call a root-level endpoint (health, metrics) that sits outside /api/v1. */
async function apiRoot(path) {
  const headers = state.token ? { Authorization: `Bearer ${state.token}` } : {};
  const response = await fetch(path, { headers });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function loadHealth() {
  try {
    const health = await apiRoot('/health');
    const node = $('health-state');
    node.textContent = health.status;
    node.className = `pill ${
      { healthy: 'pill-ok', degraded: 'pill-warn', partial: 'pill-warn' }[health.status] || 'pill-bad'
    }`;
    $('site-name').textContent = `${health.site_name} · ${health.site_id}`;
    return health;
  } catch { return null; }
}

async function loadSystem() {
  const [health, models, stats] = await Promise.all([
    loadHealth(),
    apiRoot('/api/v1/system/models').catch(() => null),
    api('/events/statistics?hours=24').catch(() => null),
  ]);
  if (health) renderStats(health, stats);
  if (models) renderModels(models);
  if (stats) renderEventSummary(stats);
  renderCameraTable();
}

function renderStats(health, stats) {
  const row = $('stat-row');
  row.innerHTML = '';
  const online = health.cameras_online;
  const total = health.cameras_total;
  const cards = [
    ['Cameras online', `${online}/${total}`, online === total ? 'ok' : online ? 'warn' : 'bad'],
    ['Node status', health.status, health.status === 'healthy' ? 'ok' : 'warn'],
    ['Events (24 h)', stats ? stats.total : '—', ''],
    ['Unacknowledged', stats ? stats.unacknowledged : '—', stats && stats.unacknowledged ? 'warn' : 'ok'],
    ['Uptime', `${Math.floor(health.uptime_seconds / 3600)}h`, ''],
    ['Evidence', `${health.storage?.evidence?.megabytes ?? 0} MB`, ''],
  ];
  cards.forEach(([label, value, tone]) => {
    const card = el('div', `stat ${tone}`);
    card.appendChild(el('div', 'value', String(value)));
    card.appendChild(el('div', 'label', label));
    row.appendChild(card);
  });
}

function renderModels(models) {
  const node = $('model-info');
  node.innerHTML = '';
  const active = models.active || {};
  const list = el('dl', 'kv');
  const rows = [
    ['Detector', active.detector_mode || 'unavailable'],
    ['Neural', active.detector_is_neural ? 'yes' : 'no (degraded fallback)'],
    ['ANPR', active.anpr_available ? 'available' : 'unavailable'],
    ['Face recognition', active.face_available ? 'available' : 'unavailable'],
    ['Providers', (models.providers || []).join(', ')],
    ['Registry entries', String((models.entries || []).length)],
  ];
  rows.forEach(([k, v]) => {
    list.appendChild(el('dt', null, k));
    list.appendChild(el('dd', null, v));
  });
  node.appendChild(list);

  if (active.degraded) {
    node.appendChild(el('p', 'error',
      'This node is running classical fallback detection. Accuracy is materially ' +
      'reduced until model artefacts are installed.'));
  }
}

function renderEventSummary(stats) {
  const node = $('event-summary');
  node.innerHTML = '';
  const entries = Object.entries(stats.by_type || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { node.appendChild(el('p', 'muted', 'No events in the last 24 hours.')); return; }

  const max = Math.max(...entries.map(([, count]) => count));
  const chart = el('div', 'bar-chart');
  entries.forEach(([type, count]) => {
    const row = el('div', 'row');
    row.appendChild(el('span', null, type.replace(/_/g, ' ')));
    const track = el('div', 'track');
    const fill = el('div', 'fill');
    fill.style.width = `${(count / max) * 100}%`;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el('span', 'count', String(count)));
    chart.appendChild(row);
  });
  node.appendChild(chart);
}

function renderCameraTable() {
  const body = $('camera-rows');
  if (!body) return;
  body.innerHTML = '';
  state.cameras.forEach((cam) => {
    const row = el('tr');
    row.appendChild(el('td', null, cam.camera.name || cam.camera.id));
    const status = el('td');
    const pill = el('span', `pill ${cam.stream.connected ? 'pill-ok' : 'pill-bad'}`,
      cam.stream.connected ? 'online' : 'offline');
    status.appendChild(pill);
    row.appendChild(status);
    row.appendChild(el('td', null, cam.stream.fps.toFixed(1)));
    row.appendChild(el('td', null, `${cam.stats.avg_latency_ms} ms`));
    row.appendChild(el('td', null, cam.detector_mode));
    row.appendChild(el('td', null, String(cam.stats.frames_processed)));
    body.appendChild(row);
  });
}

/* ------------------------------------------------------- session restore -- */

(function restore() {
  const token = sessionStorage.getItem('ibvap.token');
  if (!token) return;
  state.token = token;
  state.refreshToken = sessionStorage.getItem('ibvap.refresh');
  state.user = sessionStorage.getItem('ibvap.user');
  state.role = sessionStorage.getItem('ibvap.role');
  $('login').hidden = true;
  $('app').hidden = false;
  $('user-name').textContent = `${state.user} (${state.role})`;
  boot();
})();
