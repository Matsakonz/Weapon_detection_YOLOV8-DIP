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
        <td style="text-align: right;">
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

// 5. Load & Save Settings
function updateVal(elementId, val, isPercent = false) {
  const display = isPercent ? `${Math.round(parseFloat(val) * 100)}%` : val;
  document.getElementById(elementId).innerText = display;
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    const cfg = await res.json();

    if (cfg.CONF_MIN !== undefined) {
      document.getElementById('confMin').value = cfg.CONF_MIN;
      updateVal('valConfMin', cfg.CONF_MIN, true);
    }
    if (cfg.CONF_HIGH !== undefined) {
      document.getElementById('confHigh').value = cfg.CONF_HIGH;
      updateVal('valConfHigh', cfg.CONF_HIGH, true);
    }
    if (cfg.WEIGHT_AI !== undefined) {
      document.getElementById('weightAi').value = cfg.WEIGHT_AI;
      updateVal('valWeightAi', cfg.WEIGHT_AI);
    }
    if (cfg.WEIGHT_SHAPE !== undefined) {
      document.getElementById('weightShape').value = cfg.WEIGHT_SHAPE;
      updateVal('valWeightShape', cfg.WEIGHT_SHAPE);
    }
    if (cfg.WEIGHTED_SCORE_MIN !== undefined) {
      document.getElementById('scoreMin').value = cfg.WEIGHTED_SCORE_MIN;
      updateVal('valScoreMin', cfg.WEIGHTED_SCORE_MIN, true);
    }
    if (cfg.FRAME_ACCUMULATION_MIN !== undefined) {
      document.getElementById('frameAccum').value = cfg.FRAME_ACCUMULATION_MIN;
      updateVal('valFrameAccum', cfg.FRAME_ACCUMULATION_MIN + ' Frames');
    }
    if (cfg.USE_SAHI !== undefined) {
      const sahiBox = document.getElementById('useSahi');
      if (sahiBox) sahiBox.checked = Boolean(cfg.USE_SAHI);
    }

    // Render Target Threat Classes checkboxes
    if (cfg.all_classes && document.getElementById('classesCheckboxContainer')) {
      const container = document.getElementById('classesCheckboxContainer');
      const targetList = (cfg.target_classes || []).map(c => c.toLowerCase());
      container.innerHTML = cfg.all_classes.map(clsName => {
        const isChecked = targetList.includes(clsName.toLowerCase());
        const isDefaultWeapon = ['pistol', 'knife'].includes(clsName.toLowerCase());
        return `
          <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 6px 10px; background: rgba(0,0,0,0.3); border-radius: 6px; border: 1px solid ${isChecked ? 'rgba(255, 82, 82, 0.4)' : 'rgba(255,255,255,0.06)'}; font-size: 0.8rem;">
            <input type="checkbox" name="targetClass" value="${clsName}" ${isChecked ? 'checked' : ''} style="cursor: pointer;">
            <span style="font-weight: 500; color: ${isDefaultWeapon ? '#ff5252' : '#ffffff'};">${clsName}</span>
          </label>
        `;
      }).join('');
    }
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

async function saveSettings(event) {
  event.preventDefault();

  const selectedClasses = Array.from(document.querySelectorAll('input[name="targetClass"]:checked')).map(cb => cb.value);

  const newSettings = {
    CONF_MIN: parseFloat(document.getElementById('confMin').value),
    CONF_HIGH: parseFloat(document.getElementById('confHigh').value),
    WEIGHT_AI: parseFloat(document.getElementById('weightAi').value),
    WEIGHT_SHAPE: parseFloat(document.getElementById('weightShape').value),
    WEIGHTED_SCORE_MIN: parseFloat(document.getElementById('scoreMin').value),
    FRAME_ACCUMULATION_MIN: parseInt(document.getElementById('frameAccum').value),
    USE_SAHI: document.getElementById('useSahi') ? document.getElementById('useSahi').checked : true,
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
      alert("Settings saved successfully and applied to active detection pipeline!");
    }
  } catch (err) {
    alert("Failed to save settings.");
  }
}
