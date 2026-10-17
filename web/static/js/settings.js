// Settings and Camera Management Logic

document.addEventListener('DOMContentLoaded', () => {
  loadCamerasTable();
  loadSettings();
});

// 1. Load and Render Camera Table
async function loadCamerasTable() {
  try {
    const res = await fetch('/api/cameras');
    const cameras = await res.json();
    const tbody = document.getElementById('cameraTableBody');

    if (cameras.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-sub); padding: 24px;">No cameras configured. Click Add New Camera above.</td></tr>';
      return;
    }

    tbody.innerHTML = cameras.map(cam => `
      <tr>
        <td style="font-weight: 600;">${cam.name}</td>
        <td>
          <span class="type-pill ${cam.type === 'ip' ? 'ip' : ''}">
            ${cam.type === 'ip' ? 'IP STREAM' : 'LOCAL USB'}
          </span>
        </td>
        <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-secondary); word-break: break-all;">
          ${cam.source}
        </td>
        <td>
          <span style="color: #ffffff; font-size: 0.72rem; font-family: var(--font-mono);">● ONLINE</span>
        </td>
        <td style="text-align: right; display: flex; gap: 6px; justify-content: flex-end;">
          <button class="btn-secondary" onclick="openEditCameraModal('${cam.id}', '${cam.name.replace(/'/g, "\\'").replace(/"/g, "&quot;")}', '${cam.type}', '${cam.source.replace(/'/g, "\\'").replace(/"/g, "&quot;")}')" style="padding: 5px 12px; font-size: 0.75rem;">✏ Edit</button>
          <button class="btn-danger" onclick="deleteCamera('${cam.id}', '${cam.name}')">Delete</button>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    console.error("Error loading cameras:", err);
  }
}

// 2. Add Camera Modal Control
function openAddCameraModal() {
  document.getElementById('addCameraModal').classList.add('active');
}

function closeAddCameraModal() {
  document.getElementById('addCameraModal').classList.remove('active');
  document.getElementById('addCameraForm').reset();
}

function onCameraTypeChange(type) {
  const label = document.getElementById('camSourceLabel');
  const input = document.getElementById('camSource');
  const help = document.getElementById('camSourceHelp');

  if (type === 'local') {
    label.innerText = 'Device Index';
    input.placeholder = 'e.g. 0, 1, or 2';
    help.innerText = 'Local camera device index (0 for built-in, 1 for external USB / iPhone).';
  } else {
    label.innerText = 'RTSP / HTTP Stream URL';
    input.placeholder = 'rtsp://admin:password@192.168.1.100:554/stream';
    help.innerText = 'Supports RTSP, HTTP MJPEG, or HLS stream URLs from your IP camera.';
  }
}

// 3. Submit New Camera
async function submitAddCamera(event) {
  event.preventDefault();
  const name = document.getElementById('camName').value.trim();
  const type = document.getElementById('camType').value;
  const source = document.getElementById('camSource').value.trim();

  if (!name || !source) return;

  try {
    const res = await fetch('/api/cameras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, type, source })
    });
    const result = await res.json();

    if (result.success) {
      closeAddCameraModal();
      loadCamerasTable();
      alert(`Camera "${name}" added successfully!`);
    } else {
      alert(`Error: ${result.message}`);
    }
  } catch (err) {
    alert("Failed to add camera.");
  }
}

// 4. Delete Camera
async function deleteCamera(camId, name) {
  if (!confirm(`Are you sure you want to remove "${name}"?`)) return;

  try {
    const res = await fetch(`/api/cameras/${camId}`, { method: 'DELETE' });
    const result = await res.json();
    if (result.success) {
      loadCamerasTable();
    } else {
      alert(`Error: ${result.message}`);
    }
  } catch (err) {
    alert("Failed to delete camera.");
  }
}

// 5. Edit Camera Modal
function openEditCameraModal(camId, name, type, source) {
  document.getElementById('editCamId').value = camId;
  document.getElementById('editCamName').value = name;
  document.getElementById('editCamType').value = type || 'ip';
  document.getElementById('editCamSource').value = source;
  onEditCameraTypeChange(type || 'ip');
  document.getElementById('editCameraModal').classList.add('active');
}

function closeEditCameraModal() {
  document.getElementById('editCameraModal').classList.remove('active');
  document.getElementById('editCameraForm').reset();
}

function onEditCameraTypeChange(type) {
  const label = document.getElementById('editCamSourceLabel');
  const input = document.getElementById('editCamSource');
  const help = document.getElementById('editCamSourceHelp');
  if (type === 'local') {
    label.innerText = 'Device Index';
    input.placeholder = 'e.g. 0, 1, or 2';
    help.innerText = 'Local camera device index (0 for built-in, 1 for external USB / iPhone).';
  } else {
    label.innerText = 'RTSP / HTTP Stream URL';
    input.placeholder = 'rtsp://admin:password@192.168.1.100:554/stream';
    help.innerText = 'Supports RTSP, HTTP MJPEG, or HLS stream URLs from your IP camera.';
  }
}

async function submitEditCamera(event) {
  event.preventDefault();
  const camId = document.getElementById('editCamId').value;
  const name = document.getElementById('editCamName').value.trim();
  const type = document.getElementById('editCamType').value;
  const source = document.getElementById('editCamSource').value.trim();

  if (!name || !source) return;

  try {
    const res = await fetch(`/api/cameras/${camId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, type, source })
    });
    const result = await res.json();
    if (result.success) {
      closeEditCameraModal();
      loadCamerasTable();
    } else {
      alert(`Error: ${result.message}`);
    }
  } catch (err) {
    alert('Failed to update camera.');
  }
}

// 6. Load & Save Settings & Model Selection

let availableModels = [];
let currentActiveModel = "";

function updateVal(elementId, val, isPercent = false) {
  const display = isPercent ? `${Math.round(parseFloat(val) * 100)}%` : val;
  const el = document.getElementById(elementId);
  if (el) el.innerText = display;
}

// Model Manager Functions
async function loadModelsList(selectedToHighlight = null) {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    availableModels = data.models || [];
    currentActiveModel = data.active_model || (availableModels[0] ? availableModels[0].name : "");

    const selectEl = document.getElementById('modelSelect');
    const activeBadge = document.getElementById('activeModelBadge');
    const infoText = document.getElementById('modelInfoText');

    if (infoText) {
      infoText.innerText = `${availableModels.length} model(s) available in models/`;
    }

    if (activeBadge) {
      activeBadge.innerText = `Active: ${currentActiveModel}`;
      activeBadge.style.background = 'rgba(76, 175, 80, 0.15)';
      activeBadge.style.borderColor = 'rgba(76, 175, 80, 0.4)';
      activeBadge.style.color = '#81c784';
    }

    if (selectEl) {
      const targetModel = selectedToHighlight || currentActiveModel;
      selectEl.innerHTML = availableModels.map(m => {
        const isSel = (m.name === targetModel);
        return `<option value="${m.name}" ${isSel ? 'selected' : ''}>${m.name} (${m.size_mb} MB — ${m.classes.length} classes)</option>`;
      }).join('');

      onModelSelectionChange(targetModel);
    }
  } catch (err) {
    console.error("Error loading models:", err);
  }
}

function onModelSelectionChange(modelName) {
  const model = availableModels.find(m => m.name === modelName);
  const sizeBadge = document.getElementById('modelSizeBadge');
  const previewContainer = document.getElementById('modelClassesPreview');

  if (!model) {
    if (sizeBadge) sizeBadge.innerText = '';
    if (previewContainer) previewContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.75rem;">No model details available.</span>';
    return;
  }

  if (sizeBadge) {
    const isActive = (model.name === currentActiveModel);
    sizeBadge.innerHTML = `<span style="color: ${isActive ? '#81c784' : 'var(--text-secondary)'}; font-weight: 600;">${isActive ? '● CURRENTLY LOADED' : '○ READY TO LOAD'}</span> &nbsp;|&nbsp; ${model.size_mb} MB &nbsp;|&nbsp; ${model.classes.length} detected class(es)`;
  }

  if (previewContainer) {
    if (!model.classes || model.classes.length === 0) {
      previewContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.75rem;">No class metadata found in model.</span>';
    } else {
      const threatKeywords = ['pistol', 'gun', 'rifle', 'weapon', 'handgun', 'firearm', 'armed'];
      previewContainer.innerHTML = model.classes.map(cls => {
        const isThreat = threatKeywords.some(k => cls.toLowerCase().includes(k));
        return `
          <span style="font-family: var(--font-mono); font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; background: ${isThreat ? 'rgba(255, 82, 82, 0.15)' : 'rgba(255, 255, 255, 0.05)'}; color: ${isThreat ? '#ff5252' : '#cccccc'}; border: 1px solid ${isThreat ? 'rgba(255, 82, 82, 0.3)' : 'rgba(255, 255, 255, 0.1)'};">
            ${cls}
          </span>
        `;
      }).join('');
    }
  }
}

async function applyModelSelection() {
  const selectEl = document.getElementById('modelSelect');
  if (!selectEl) return;
  const selectedModel = selectEl.value;

  const btn = document.getElementById('btnApplyModel');
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Loading Model...';
  }

  try {
    const res = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel })
    });
    const result = await res.json();

    if (result.success) {
      currentActiveModel = result.active_model;
      const activeBadge = document.getElementById('activeModelBadge');
      if (activeBadge) {
        activeBadge.innerText = `Active: ${currentActiveModel}`;
      }
      onModelSelectionChange(currentActiveModel);

      // Refresh target threat class filter checkboxes for new model
      renderTargetClasses(result.all_classes, result.target_classes);

      alert(`Model "${currentActiveModel}" successfully loaded and active in the live detection pipeline!`);
    } else {
      alert(`Error loading model: ${result.message}`);
    }
  } catch (err) {
    console.error("Error switching model:", err);
    alert("Failed to switch model.");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '⚡ Switch Model';
    }
  }
}

function renderTargetClasses(all_classes, target_classes) {
  const container = document.getElementById('classesCheckboxContainer');
  if (!container || !all_classes) return;

  const targetList = (target_classes || []).map(c => c.toLowerCase());

  container.innerHTML = all_classes.map(clsName => {
    const isChecked = targetList.includes(clsName.toLowerCase());
    const textColor = isChecked ? '#ff5252' : '#cccccc';
    const borderColor = isChecked ? 'rgba(255, 82, 82, 0.5)' : 'rgba(255,255,255,0.08)';
    const bgColor = isChecked ? 'rgba(255, 82, 82, 0.08)' : 'rgba(0,0,0,0.3)';

    return `
      <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 6px 10px; background: ${bgColor}; border-radius: 6px; border: 1px solid ${borderColor}; font-size: 0.8rem; transition: all 0.2s;">
        <input type="checkbox" name="targetClass" value="${clsName}" ${isChecked ? 'checked' : ''} style="cursor: pointer;" onchange="this.parentElement.style.borderColor = this.checked ? 'rgba(255,82,82,0.5)' : 'rgba(255,255,255,0.08)'; this.parentElement.style.background = this.checked ? 'rgba(255,82,82,0.08)' : 'rgba(0,0,0,0.3)'; this.nextElementSibling.style.color = this.checked ? '#ff5252' : '#cccccc';">
        <span style="font-weight: 500; color: ${textColor};">${clsName}</span>
      </label>
    `;
  }).join('');
}

async function loadSettings() {
  try {
    await loadModelsList();

    const res = await fetch('/api/settings');
    const cfg = await res.json();

    if (cfg.model_weapon && document.getElementById('modelSelect')) {
      document.getElementById('modelSelect').value = cfg.model_weapon;
      onModelSelectionChange(cfg.model_weapon);
    }
    if (cfg.CONF_MIN !== undefined && document.getElementById('confMin')) {
      document.getElementById('confMin').value = cfg.CONF_MIN;
      updateVal('valConfMin', cfg.CONF_MIN, true);
    }
    if (cfg.FRAME_ACCUMULATION_MIN !== undefined && document.getElementById('frameAccum')) {
      document.getElementById('frameAccum').value = cfg.FRAME_ACCUMULATION_MIN;
      updateVal('valFrameAccum', cfg.FRAME_ACCUMULATION_MIN + ' Frames');
    }
    if (cfg.USE_SAHI !== undefined) {
      const sahiBox = document.getElementById('useSahi');
      if (sahiBox) sahiBox.checked = Boolean(cfg.USE_SAHI);
    }
    if (cfg.SAVE_SNAPSHOTS !== undefined || cfg.save_snapshots !== undefined) {
      const snapBox = document.getElementById('saveSnapshots');
      const val = (cfg.SAVE_SNAPSHOTS !== undefined) ? cfg.SAVE_SNAPSHOTS : cfg.save_snapshots;
      if (snapBox) snapBox.checked = Boolean(val);
    }
    if (cfg.DASHBOARD_ACTIVE !== undefined) {
      const dashBox = document.getElementById('dashboardActive');
      if (dashBox) {
        dashBox.checked = Boolean(cfg.DASHBOARD_ACTIVE);
        onAlarmActiveToggle(dashBox.checked);
      }
    }

    // Render Target Threat Classes checkboxes
    if (cfg.all_classes) {
      renderTargetClasses(cfg.all_classes, cfg.target_classes);
    }
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

function onAlarmActiveToggle(checked) {
  const pill = document.getElementById('alarmArmedPill');
  if (pill) {
    if (checked) {
      pill.innerText = 'ARMED';
      pill.style.background = 'rgba(76, 175, 80, 0.2)';
      pill.style.color = '#81c784';
      pill.style.borderColor = 'rgba(76, 175, 80, 0.4)';
    } else {
      pill.innerText = 'DISARMED (MUTED)';
      pill.style.background = 'rgba(255, 82, 82, 0.15)';
      pill.style.color = '#ff6666';
      pill.style.borderColor = 'rgba(255, 82, 82, 0.3)';
    }
  }
}

function playSynthesizedChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime); // A5 note
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3); // Drop to A4

    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch (e) {
    console.warn("Web Audio chime failed:", e);
  }
}

async function triggerTestAlarm() {
  const sound = document.getElementById('alertSound');
  if (sound) {
    sound.currentTime = 0;
    sound.play().catch(() => playSynthesizedChime());
  } else {
    playSynthesizedChime();
  }

  try {
    const res = await fetch('/api/test_alarm', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      alert(`🔔 Test Alarm Triggered!\n\n• macOS System Buzzer: Activated (Ping.aiff)\n• Test Snapshot Saved: ${data.snapshot}\n• System Armed: ${data.dashboard_active ? 'YES' : 'NO (Muted)'}`);
    } else {
      alert("Failed to trigger test alarm.");
    }
  } catch (err) {
    console.error("Test alarm error:", err);
    alert("Test alarm request failed.");
  }
}

async function saveSettings(event) {
  event.preventDefault();

  const selectedClasses = Array.from(document.querySelectorAll('input[name="targetClass"]:checked')).map(cb => cb.value);
  const selectedModel = document.getElementById('modelSelect') ? document.getElementById('modelSelect').value : null;

  const confMinEl = document.getElementById('confMin');
  const frameAccumEl = document.getElementById('frameAccum');

  const newSettings = {
    model_weapon: selectedModel,
    CONF_MIN: confMinEl ? parseFloat(confMinEl.value) : 0.60,
    FRAME_ACCUMULATION_MIN: frameAccumEl ? parseInt(frameAccumEl.value) : 4,
    USE_SAHI: document.getElementById('useSahi') ? document.getElementById('useSahi').checked : true,
    SAVE_SNAPSHOTS: document.getElementById('saveSnapshots') ? document.getElementById('saveSnapshots').checked : false,
    DASHBOARD_ACTIVE: document.getElementById('dashboardActive') ? document.getElementById('dashboardActive').checked : true,
    target_classes: selectedClasses
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newSettings)
    });
    const result = await res.json();
    if (result.success) {
      if (result.model_weapon) {
        currentActiveModel = result.model_weapon;
        const activeBadge = document.getElementById('activeModelBadge');
        if (activeBadge) activeBadge.innerText = `Active: ${currentActiveModel}`;
        onModelSelectionChange(currentActiveModel);
      }
      if (result.all_classes && result.target_classes) {
        renderTargetClasses(result.all_classes, result.target_classes);
      }
      alert("Settings saved successfully and applied to active detection pipeline!");
    }
  } catch (err) {
    alert("Failed to save settings.");
  }
}

