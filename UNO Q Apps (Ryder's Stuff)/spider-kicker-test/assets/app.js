const form = document.querySelector('#move-form');
const statusEl = document.querySelector('#status');
const positionReadout = document.querySelector('#position-readout');
const errorContainer = document.querySelector('#error-container');
const jogButtons = document.querySelectorAll('.jog-button');
const nudgeButtons = document.querySelectorAll('.nudge-button');
const kickButton = document.querySelector('#kick-button');
const homeButton = document.querySelector('#home-button');
const resetHomeButton = document.querySelector('#reset-home-button');

const debugToggleLink = document.querySelector('#debug-toggle-link');
const debugPanel = document.querySelector('#debug-panel');
const autonomousCheckbox = document.querySelector('#autonomous-checkbox');
const autonomousStateEl = document.querySelector('#autonomous-state');
const calibratedEl = document.querySelector('#calibrated');
const platformDetectedEl = document.querySelector('#platform-detected');
const ballDetectedEl = document.querySelector('#ball-detected');
const cyclePhaseEl = document.querySelector('#cycle-phase');
const ballSamplesEl = document.querySelector('#ball-samples');
const cycleTargetEl = document.querySelector('#cycle-target');
const verifyErrorEl = document.querySelector('#verify-error');
const verifyRetriesEl = document.querySelector('#verify-retries');
const lastKickEl = document.querySelector('#last-kick');
const exposureManualCheckbox = document.querySelector('#exposure-manual-checkbox');
const exposureSlider = document.querySelector('#exposure-slider');
const exposureValueEl = document.querySelector('#exposure-value');

const ui = new WebUI();
ui.on_connect(onUIConnected);
ui.on_disconnect(onUIDisconnected);
ui.on_message('state_update', onStateUpdate);
ui.on_message('move_done', onMoveDone);
ui.on_message('move_error', onMoveError);
ui.on_message('go_home_done', onGoHomeDone);
ui.on_message('reset_home_done', onResetHomeDone);
ui.on_message('nudge_done', onNudgeDone);
ui.on_message('nudge_error', onNudgeError);
ui.on_message('jog_done', onJogDone);
ui.on_message('jog_error', onJogError);
ui.on_message('kick_done', onKickDone);
ui.on_message('kick_error', onKickError);
ui.on_message('tracking_state', onTrackingState);
ui.on_message('autonomous_state', onAutonomousState);
ui.on_message('exposure_state', onExposureState);
ui.on_message('exposure_error', onExposureError);

function onUIConnected() {
  errorContainer.style.display = 'none';
  errorContainer.textContent = '';
  ui.send_message('get_state');
  ui.send_message('get_tracking_state');
  ui.send_message('get_exposure_state');
}

function updatePosition(data) {
  document.querySelector('#x').value = data.x;
  document.querySelector('#y').value = data.y;
  positionReadout.textContent = `x: ${data.x}   y: ${data.y}   z: ${data.z}`;
}

function onStateUpdate(data) {
  updatePosition(data);
}

function onUIDisconnected() {
  errorContainer.style.display = 'block';
  errorContainer.textContent = 'Connection to the board lost. Please check the connection.';
}

function onMoveDone(data) {
  updatePosition(data);
  statusEl.textContent = `Moved to (${data.x}, ${data.y}, ${data.z}).`;
}

function onMoveError(data) {
  statusEl.textContent = `Error: ${data.error}`;
}

function onGoHomeDone(data) {
  updatePosition(data);
  statusEl.textContent = `Returned home: (${data.x}, ${data.y}, ${data.z}).`;
}

function onResetHomeDone(data) {
  updatePosition(data);
  statusEl.textContent = `Home reset to current position: (${data.x}, ${data.y}, ${data.z}).`;
}

function onNudgeDone(data) {
  updatePosition(data);
  statusEl.textContent = `Moved to (${data.x}, ${data.y}, ${data.z}).`;
}

function onNudgeError(data) {
  statusEl.textContent = `Error: ${data.error}`;
}

function onJogDone(data) {
  statusEl.textContent = `Motor ${data.motor + 1} jogged ${data.reel_in ? 'in' : 'out'}.`;
}

function onJogError(data) {
  statusEl.textContent = `Error: ${data.error}`;
}

function onKickDone() {
  statusEl.textContent = 'Kicked.';
}

function onKickError(data) {
  statusEl.textContent = `Error: ${data.error}`;
}

function formatTarget(targetMm) {
  if (!targetMm) return '--';
  const [x, y] = targetMm;
  return `(${x.toFixed(0)}, ${y.toFixed(0)})`;
}

function applyTrackingState(data) {
  calibratedEl.textContent = data.calibrated ? 'yes' : 'no';
  platformDetectedEl.textContent = data.platform_detected ? 'yes' : 'no';
  ballDetectedEl.textContent = data.ball_detected ? 'yes' : 'no';
  cyclePhaseEl.textContent = data.cycle_phase ?? '--';
  ballSamplesEl.textContent = `${data.samples_collected ?? 0}/${data.sample_count_target ?? '--'}`;
  cycleTargetEl.textContent = formatTarget(data.last_target_mm);
  verifyErrorEl.textContent = data.last_verify_error_mm == null ? '--' : data.last_verify_error_mm.toFixed(1);
  verifyRetriesEl.textContent = data.verify_retry_count ?? '--';
  lastKickEl.textContent = data.last_kick_ago_s == null ? 'never' : `${data.last_kick_ago_s}s ago`;
  if (typeof data.autonomous === 'boolean') {
    autonomousCheckbox.checked = data.autonomous;
    autonomousStateEl.textContent = data.autonomous ? 'ON' : 'OFF';
  }
}

function onTrackingState(data) {
  applyTrackingState(data);
}

function onAutonomousState(data) {
  autonomousCheckbox.checked = data.autonomous;
  autonomousStateEl.textContent = data.autonomous ? 'ON' : 'OFF';
}

function onExposureState(data) {
  exposureManualCheckbox.checked = data.enabled;
  exposureSlider.value = data.value;
  exposureValueEl.textContent = data.value;
}

function onExposureError(data) {
  statusEl.textContent = `Error: ${data.error}`;
}

// Poll tracking status regularly so the default view stays live even though
// the video feed itself is a plain MJPEG <img> with no per-frame message.
setInterval(() => ui.send_message('get_tracking_state'), 1000);

autonomousCheckbox.addEventListener('change', () => {
  ui.send_message('set_autonomous', { enabled: autonomousCheckbox.checked });
});

function sendExposure() {
  ui.send_message('set_exposure', {
    enabled: exposureManualCheckbox.checked,
    value: Number(exposureSlider.value),
  });
}

exposureManualCheckbox.addEventListener('change', sendExposure);
exposureSlider.addEventListener('input', () => {
  exposureValueEl.textContent = exposureSlider.value;
});
exposureSlider.addEventListener('change', sendExposure); // fires on release, not per drag-tick

debugToggleLink.addEventListener('click', (event) => {
  event.preventDefault();
  const showing = debugPanel.style.display !== 'none';
  debugPanel.style.display = showing ? 'none' : 'block';
  debugToggleLink.textContent = showing ? 'Show manual/debug controls' : 'Hide manual/debug controls';
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  ui.send_message('move_to', {
    x: document.querySelector('#x').value,
    y: document.querySelector('#y').value,
  });
  statusEl.textContent = 'Moving...';
});

jogButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const motor = Number(button.dataset.motor);
    const reelIn = button.dataset.reelIn === 'true';
    ui.send_message('jog', { motor, reel_in: reelIn });
    statusEl.textContent = 'Jogging...';
  });
});

kickButton.addEventListener('click', () => {
  ui.send_message('kick');
  statusEl.textContent = 'Kicking...';
});

homeButton.addEventListener('click', () => {
  ui.send_message('go_home');
  statusEl.textContent = 'Returning home...';
});

resetHomeButton.addEventListener('click', () => {
  if (!confirm('Set the platform\'s CURRENT position as the new home? Only do this if it is actually where you want home to be.')) {
    return;
  }
  ui.send_message('reset_home');
  statusEl.textContent = 'Resetting home...';
});

nudgeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const axis = button.dataset.axis;
    const sign = Number(button.dataset.sign);
    ui.send_message('nudge', { axis, sign });
    statusEl.textContent = 'Moving...';
  });
});
