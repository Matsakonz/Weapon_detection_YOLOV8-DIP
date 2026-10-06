// Dashboard logic: Multi-camera grid, single camera switching, real-time polling, and alerts

let activeCameraId = "cam_1"; // Default working camera index 1
let isAlertActive = false;
let isGridView = false;
let cameraList = [];

document.addEventListener('DOMContentLoaded', () => {
  loadCameras();
  startStatusPolling();
});

// 1. Fetch available cameras and populate selector tabs
async function loadCameras() {
  try {
    const res = await fetch('/api/cameras');
    cameraList = await res.json();
    const container = document.getElementById('cameraSelector');
    container.innerHTML = '';

    cameraList.forEach((cam) => {
      const btn = document.createElement('button');
      btn.className = `cam-btn ${(!isGridView && cam.id === activeCameraId) ? 'active' : ''}`;
      btn.setAttribute('data-cam-id', cam.id);
      btn.innerText = cam.name;
      btn.onclick = () => switchCamera(cam.id, btn);
      container.appendChild(btn);
    });

    if (isGridView) {
      renderCameraGrid();
    } else if (cameraList.length > 0 && !activeCameraId) {
      switchCamera(cameraList[0].id, container.querySelectorAll('.cam-btn')[0]);
    }
  } catch (err) {
    console.error("Failed to load cameras:", err);
  }
}

// 2. "Watch All Cameras" Grid View
function watchAllCameras() {
  isGridView = true;
  document.getElementById('singleViewWrapper').style.display = 'none';
  const gridWrapper = document.getElementById('gridViewWrapper');
  gridWrapper.style.display = 'block';

  // Highlight "Watch All" button and un-highlight individual buttons
  document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
  const btnWatchAll = document.getElementById('btnWatchAll');
  if (btnWatchAll) btnWatchAll.classList.add('active');

  const hudCam = document.getElementById('hudCamera');
  if (hudCam) hudCam.innerText = `FEED: ALL CAMERAS (${cameraList.length})`;

  renderCameraGrid();
}

// 3. Render Multi-Camera Grid
function renderCameraGrid() {
  const container = document.getElementById('camerasGrid');
  if (!container) return;

  if (cameraList.length === 0) {
    container.innerHTML = '<div class="empty-state" style="grid-column: 1 / -1;">No cameras configured. <a href="/settings" style="color:#ffffff;">Add a camera</a></div>';
    return;
  }

  container.innerHTML = cameraList.map(cam => `
    <div class="grid-cam-card" id="card_${cam.id}">
      <div class="grid-cam-header">
        <div class="grid-cam-title">
          <span class="live-dot"></span>
          <span>${cam.name}</span>
          <span class="grid-type-pill">${cam.type === 'ip' ? 'IP' : 'USB'}</span>
        </div>
        <button class="grid-focus-btn" onclick="focusCamera('${cam.id}')">↗ Focus</button>
      </div>
      <div class="grid-cam-body">
        <img class="grid-stream-img" src="/video_feed/${cam.id}" alt="${cam.name}" loading="lazy" />
      </div>
    </div>
  `).join('');
}

// 4. Focus a single camera from Grid View
function focusCamera(camId) {
  const btn = document.querySelector(`.cam-btn[data-cam-id="${camId}"]`);
  switchCamera(camId, btn);
}

// 5. Switch Single Camera Feed
function switchCamera(camId, btnElement) {
  isGridView = false;
  document.getElementById('gridViewWrapper').style.display = 'none';
  document.getElementById('singleViewWrapper').style.display = 'flex';

  activeCameraId = camId;
  const streamImg = document.getElementById('liveStream');
  streamImg.src = `/video_feed/${camId}`;

  // Update tabs
  document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
  if (btnElement) {
    btnElement.classList.add('active');
  } else {
    const matching = document.querySelector(`.cam-btn[data-cam-id="${camId}"]`);
    if (matching) matching.classList.add('active');
  }

  const hudCam = document.getElementById('hudCamera');
  const camObj = cameraList.find(c => c.id === camId);
  const name = camObj ? camObj.name : camId.toUpperCase();
  if (hudCam) hudCam.innerText = `FEED: ${name}`;
}

// 6. Real-time Status Polling
function startStatusPolling() {
  setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      // Update HUD elements
      const fpsEl = document.getElementById('hudFps');
      if (fpsEl) fpsEl.innerText = `FPS: ${data.fps || '--'}`;

      const modeEl = document.getElementById('hudMode');
      if (modeEl) {
        if (data.view_mode === 'detect') {
          modeEl.innerText = 'MODE: DETECT IMAGE (FORCED)';
          modeEl.style.color = '#ffffff';
          modeEl.style.borderColor = '#ffffff';
        } else if (data.view_mode === 'original') {
          modeEl.innerText = 'MODE: ORIGINAL CAM (FORCED)';
          modeEl.style.color = 'var(--text-secondary)';
          modeEl.style.borderColor = 'var(--border-subtle)';
        } else if (data.has_threat) {
          modeEl.innerText = 'MODE: DETECT IMAGE (THREAT)';
          modeEl.style.color = '#ff6666';
          modeEl.style.borderColor = 'rgba(255, 51, 51, 0.4)';
        } else {
          modeEl.innerText = 'MODE: ORIGINAL CAM (SMOOTH FPS)';
          modeEl.style.color = 'var(--text-secondary)';
          modeEl.style.borderColor = 'var(--border-subtle)';
        }
      }

      if (data.view_mode) {
        currentViewMode = data.view_mode;
        const btn = document.getElementById('btnViewModeSwitch');
        if (btn) btn.innerText = `VIEW: ${data.view_mode.toUpperCase()}`;
      }

      const timeEl = document.getElementById('hudTimestamp');
      if (timeEl) {
        timeEl.innerText = data.timestamp ? `TIME: ${data.timestamp.split(' ')[1]}` : 'TIME: --:--:--';
      }

      const threatEl = document.getElementById('statThreats');
      if (threatEl) threatEl.innerText = data.threat_count || 0;

      const personEl = document.getElementById('statPersons');
      if (personEl) personEl.innerText = data.person_count || 0;

      // Handle threat alert banner & chime
      const banner = document.getElementById('alarmBanner');
      const sound = document.getElementById('alertSound');

      if (data.is_confirmed) {
        if (banner) banner.classList.add('active');
        const alarmText = document.getElementById('alarmText');
        if (alarmText) alarmText.innerText = `WARNING: ${data.threat_label || 'THREAT'} CONFIRMED`;

        // Play audio alert once per confirmation cycle
        if (!isAlertActive) {
          isAlertActive = true;
          if (sound) sound.play().catch(() => {});
        }
      } else {
        if (banner) banner.classList.remove('active');
        isAlertActive = false;
      }

      // Update recent events list
      renderEvents(data.recent_events || []);
    } catch (e) {
      // Ignore polling hiccups
    }
  }, 600);
}

// 7. Render Recent Events in Sidebar
function renderEvents(events) {
  const container = document.getElementById('eventList');
  const countBadge = document.getElementById('eventCount');
  if (countBadge) countBadge.innerText = `${events.length} event${events.length === 1 ? '' : 's'}`;

  if (!container) return;

  if (events.length === 0) {
    container.innerHTML = '<div class="empty-state">No threat incidents recorded yet.</div>';
    return;
  }

  container.innerHTML = events.map(evt => `
    <div class="event-item">
      <div>
        <span class="event-badge danger">${evt.label || 'Threat'}</span>
        <span style="margin-left: 8px; font-weight: 500; color: #fff;">${evt.camera || 'Camera'}</span>
        <span style="margin-left: 6px; color: var(--text-muted); font-size: 0.72rem;">${Math.round((evt.score || 0) * 100)}%</span>
      </div>
      <div class="event-time">${evt.time || ''}</div>
    </div>
  `).join('');
}

// 8. Toggle Features
async function toggleFeature(feature) {
  try {
    const res = await fetch(`/api/toggle/${feature}`, { method: 'POST' });
    const data = await res.json();

    if (feature === 'zoom') {
      document.getElementById('btnToggleZoom').innerText = `Zoom: ${data.state ? 'ON' : 'OFF'}`;
    } else if (feature === 'sahi') {
      const btn = document.getElementById('btnToggleSahi');
      if (btn) btn.innerText = `SAHI: ${data.state ? 'ON' : 'OFF'}`;
    } else if (feature === 'dip') {
      document.getElementById('btnToggleDip').innerText = `DIP: ${data.state ? 'ON' : 'OFF'}`;
    } else if (feature === 'wavelet') {
      document.getElementById('btnToggleWavelet').innerText = `Wavelet: ${data.state ? 'ON' : 'OFF'}`;
    }
  } catch (err) {
    console.error(err);
  }
}

// 9. Snapshot Trigger
async function takeSnapshot() {
  try {
    const res = await fetch('/api/snapshot', { method: 'POST' });
    const data = await res.json();
    alert(`Snapshot saved successfully to:\n${data.path}`);
  } catch (err) {
    alert("Failed to capture snapshot.");
  }
}

// 10. Cycle View Mode (Auto / Detect Image / Original Cam)
let currentViewMode = "auto";
async function cycleViewMode() {
  const modes = ['auto', 'detect', 'original'];
  const nextIdx = (modes.indexOf(currentViewMode) + 1) % modes.length;
  currentViewMode = modes[nextIdx];

  try {
    const res = await fetch(`/api/view_mode/${currentViewMode}`, { method: 'POST' });
    const data = await res.json();
    const btn = document.getElementById('btnViewModeSwitch');
    if (btn) btn.innerText = `VIEW: ${currentViewMode.toUpperCase()}`;
  } catch (err) {
    console.error("Failed to switch view mode:", err);
  }
}

