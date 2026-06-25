import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

let API_BASE = "/api/replay";

const COLORS = {
  own: 0x4d8dff,
  target: 0xff5757,
  ownTrail: 0x003b8e,
  targetTrail: 0x9f0d19,
  ownWez: 0x4d8dff,
  targetWez: 0xff6b6b,
  sea: 0x0969a8,
  sky: 0x7fb7d7,
};

const AIRCRAFT_MODEL_YAW_OFFSET_DEG = 180;
const CAMERA_MODES = new Set(["blue", "red", "midpoint"]);

const State = {
  logs: [],
  replay: null,
  mesh: null,
  playing: true,
  speed: 5,
  simTime: 0,
  lastNow: 0,
  framesRendered: 0,
  showHud: true,
  showSea: true,
  showTrails: true,
  showWez: true,
  cameraMode: "midpoint",
};

const Scene = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  root: null,
  aircraftGeometry: null,
  ownship: null,
  target: null,
  sea: null,
  ownTrail: null,
  targetTrail: null,
  ownWez: null,
  targetWez: null,
};

const $ = id => document.getElementById(id);

window.DogFightViewerDebug = {
  webglOk: false,
  framesRendered: 0,
  activeLogPair: null,
  samples: 0,
};

export async function initReplayViewer(options = {}) {
  API_BASE = options.apiBase || API_BASE;
  bindEvents();
  initScene();
  await Promise.all([loadMesh(), refreshLogs()]);
  animate(0);
}

function bindEvents() {
  $("reload-button").addEventListener("click", () => refreshLogs());
  $("play-button").addEventListener("click", () => {
    State.playing = !State.playing;
    $("play-button").textContent = State.playing ? "Pause" : "Play";
  });
  $("log-select").addEventListener("change", event => {
    const index = Number(event.target.value);
    if (Number.isInteger(index) && State.logs[index]) {
      loadReplay(State.logs[index]);
    }
  });
  $("speed-slider").addEventListener("input", event => {
    State.speed = Number(event.target.value);
    $("speed-value").textContent = `${State.speed.toFixed(1)}x`;
  });
  $("timeline").addEventListener("input", event => {
    if (!State.replay) {
      return;
    }
    const ratio = Number(event.target.value);
    State.simTime = lerp(State.replay.startTime, State.replay.endTime, ratio);
    updateFrame();
  });

  bindToggle("toggle-hud", "showHud", updateVisibility);
  bindToggle("toggle-sea", "showSea", updateVisibility);
  bindToggle("toggle-trails", "showTrails", updateVisibility);
  bindToggle("toggle-wez", "showWez", updateVisibility);
  document.querySelectorAll("[data-camera-mode]").forEach(button => {
    button.addEventListener("click", () => setCameraMode(button.dataset.cameraMode));
  });
  updateCameraModeButtons();
  window.addEventListener("resize", resizeRenderer);
}

function bindToggle(id, key, callback) {
  $(id).addEventListener("change", event => {
    State[key] = event.target.checked;
    callback();
  });
}

function setCameraMode(mode) {
  if (!CAMERA_MODES.has(mode)) {
    return;
  }
  State.cameraMode = mode;
  updateCameraModeButtons();
  updateFrame();
}

function updateCameraModeButtons() {
  document.querySelectorAll("[data-camera-mode]").forEach(button => {
    const active = button.dataset.cameraMode === State.cameraMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function initScene() {
  Scene.root = $("scene-root");
  Scene.scene = new THREE.Scene();
  Scene.scene.background = new THREE.Color(COLORS.sky);
  Scene.scene.fog = new THREE.Fog(COLORS.sky, 6000, 28000);

  Scene.camera = new THREE.PerspectiveCamera(50, 1, 1, 60000);
  Scene.camera.up.set(0, 0, 1);

  Scene.renderer = new THREE.WebGLRenderer({ antialias: true });
  Scene.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  Scene.renderer.outputColorSpace = THREE.SRGBColorSpace;
  Scene.root.appendChild(Scene.renderer.domElement);
  window.DogFightViewerDebug.webglOk = Boolean(Scene.renderer.getContext());

  Scene.controls = new OrbitControls(Scene.camera, Scene.renderer.domElement);
  Scene.controls.enableDamping = true;
  Scene.controls.dampingFactor = 0.08;
  Scene.controls.screenSpacePanning = false;

  Scene.scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const sun = new THREE.DirectionalLight(0xffffff, 1.25);
  sun.position.set(0.4, -0.6, 1.0).normalize();
  Scene.scene.add(sun);

  resizeRenderer();
}

async function refreshLogs() {
  setStatus("Loading log list...");
  const data = await apiFetch("/api/logs");
  State.logs = data.logs || [];
  renderLogOptions();
  if (State.logs.length === 0) {
    setStatus("No Blue/Red CSV log pairs found.");
    return;
  }
  await loadReplay(State.logs[0]);
}

function renderLogOptions() {
  const select = $("log-select");
  select.innerHTML = "";
  if (State.logs.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No logs";
    select.appendChild(option);
    return;
  }
  State.logs.forEach((log, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = log.label || log.ownshipName || `Replay ${index + 1}`;
    select.appendChild(option);
  });
  select.value = "0";
}

async function loadMesh() {
  State.mesh = await apiFetch("/api/mesh/f16");
  Scene.aircraftGeometry = buildAircraftGeometry(State.mesh);
}

async function loadReplay(logPair) {
  setStatus("Loading replay...");
  const ownship = encodeURIComponent(logPair.ownship);
  const target = encodeURIComponent(logPair.target);
  State.replay = await apiFetch(`/api/data?ownship=${ownship}&target=${target}`);
  State.simTime = State.replay.startTime;
  State.playing = true;
  $("play-button").textContent = "Pause";
  $("timeline").value = "0";
  window.DogFightViewerDebug.activeLogPair = {
    ownship: State.replay.logs.ownship,
    target: State.replay.logs.target,
  };
  window.DogFightViewerDebug.samples = State.replay.ownship.time.length;
  $("debug-samples").textContent = String(State.replay.ownship.time.length);
  setupReplayScene();
  updateFrame();
  setStatus(`Loaded ${State.replay.logs.ownship}`);
}

function setupReplayScene() {
  clearReplayObjects();
  const replay = State.replay;
  const size = replay.seaSizeM;

  const seaGeometry = new THREE.PlaneGeometry(size, size, 80, 80);
  const positions = seaGeometry.attributes.position;
  for (let i = 0; i < positions.count; i += 1) {
    const x = positions.getX(i);
    const y = positions.getY(i);
    const z = 4 * Math.sin(x / 850) + 2.5 * Math.cos(y / 650);
    positions.setZ(i, THREE.MathUtils.clamp(z, -10, 10));
  }
  positions.needsUpdate = true;
  seaGeometry.computeVertexNormals();
  Scene.sea = new THREE.Mesh(
    seaGeometry,
    new THREE.MeshPhongMaterial({
      color: COLORS.sea,
      transparent: true,
      opacity: 0.86,
      shininess: 22,
      side: THREE.DoubleSide,
    })
  );
  Scene.scene.add(Scene.sea);

  const ownMaterial = new THREE.MeshPhongMaterial({ color: COLORS.own, shininess: 50 });
  const targetMaterial = new THREE.MeshPhongMaterial({
    color: COLORS.target,
    shininess: 50,
  });
  Scene.ownship = new THREE.Mesh(Scene.aircraftGeometry, ownMaterial);
  Scene.target = new THREE.Mesh(Scene.aircraftGeometry, targetMaterial);
  Scene.scene.add(Scene.ownship);
  Scene.scene.add(Scene.target);

  Scene.ownTrail = makeLine(COLORS.ownTrail, 4);
  Scene.targetTrail = makeLine(COLORS.targetTrail, 4);
  Scene.scene.add(Scene.ownTrail);
  Scene.scene.add(Scene.targetTrail);

  Scene.ownWez = makeWezMesh(COLORS.ownWez, 0.14);
  Scene.targetWez = makeWezMesh(COLORS.targetWez, 0.12);
  Scene.scene.add(Scene.ownWez);
  Scene.scene.add(Scene.targetWez);

  setInitialCamera();
  updateVisibility();
}

function clearReplayObjects() {
  for (const key of [
    "ownship",
    "target",
    "sea",
    "ownTrail",
    "targetTrail",
    "ownWez",
    "targetWez",
  ]) {
    const object = Scene[key];
    if (object) {
      Scene.scene.remove(object);
      disposeObject(object);
      Scene[key] = null;
    }
  }
}

function buildAircraftGeometry(mesh) {
  const vertices = [];
  for (const vertex of mesh.vertices) {
    vertices.push(vertex[0], vertex[1], vertex[2]);
  }
  const indices = [];
  for (const triangle of mesh.triangles) {
    indices.push(triangle[0], triangle[1], triangle[2]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeLine(color, width) {
  return new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color, linewidth: width })
  );
}

function makeWezMesh(color, opacity) {
  return new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      depthWrite: false,
      wireframe: false,
    })
  );
}

function updateVisibility() {
  $("hud-left").hidden = !State.showHud;
  $("hud-right").hidden = !State.showHud;
  $("hud-bottom").hidden = !State.showHud;
  if (Scene.sea) {
    Scene.sea.visible = State.showSea;
  }
  if (Scene.ownTrail) {
    Scene.ownTrail.visible = State.showTrails;
  }
  if (Scene.targetTrail) {
    Scene.targetTrail.visible = State.showTrails;
  }
  if (Scene.ownWez) {
    Scene.ownWez.visible = State.showWez;
  }
  if (Scene.targetWez) {
    Scene.targetWez.visible = State.showWez;
  }
}

function setInitialCamera() {
  const replay = State.replay;
  const own = replay.ownship.position[0];
  const target = replay.target.position[0];
  const focal = [
    (own[0] + target[0]) / 2,
    (own[1] + target[1]) / 2,
    (own[2] + target[2]) / 2,
  ];
  const scale = Math.max(
    replay.sceneExtentM,
    replay.defaults.wezRangeM * 2.5,
    1500
  );
  Scene.camera.position.set(
    focal[0] + 0.36 * scale,
    focal[1] - 0.5 * scale,
    focal[2] + 0.28 * scale
  );
  Scene.camera.near = 1;
  Scene.camera.far = Math.max(scale * 8, 10000);
  Scene.camera.updateProjectionMatrix();
  Scene.controls.target.set(focal[0], focal[1], focal[2]);
  Scene.controls.update();
}

function animate(now) {
  requestAnimationFrame(animate);
  const dt = State.lastNow ? (now - State.lastNow) / 1000 : 0;
  State.lastNow = now;

  if (State.replay && State.playing) {
    State.simTime += dt * State.speed;
    if (State.simTime > State.replay.endTime) {
      State.simTime = State.replay.startTime;
    }
    updateFrame();
  }
  Scene.controls?.update();
  Scene.renderer?.render(Scene.scene, Scene.camera);
  State.framesRendered += 1;
  window.DogFightViewerDebug.framesRendered = State.framesRendered;
  $("debug-frames").textContent = String(State.framesRendered);
}

function updateFrame() {
  const replay = State.replay;
  if (!replay || !Scene.ownship || !Scene.target) {
    return;
  }
  const ownIndex = nearestIndex(replay.ownship.time, State.simTime);
  const targetIndex = nearestIndex(replay.target.time, State.simTime);
  updateAircraft(Scene.ownship, replay.ownship, ownIndex);
  updateAircraft(Scene.target, replay.target, targetIndex);
  updateTrail(Scene.ownTrail, replay.ownship, State.simTime);
  updateTrail(Scene.targetTrail, replay.target, State.simTime);
  updateWez(Scene.ownWez, replay.ownship, ownIndex);
  updateWez(Scene.targetWez, replay.target, targetIndex);
  updateHud(ownIndex, targetIndex);
  updateFollowCamera(ownIndex, targetIndex);
  updateTimeline();
}

function updateAircraft(object, track, index) {
  const pos = track.position[index];
  const matrix = aircraftVisualMatrix(
    track.rollDeg[index],
    track.pitchDeg[index],
    track.yawDeg[index]
  );
  const rot = new THREE.Matrix4().set(
    matrix[0][0], matrix[0][1], matrix[0][2], 0,
    matrix[1][0], matrix[1][1], matrix[1][2], 0,
    matrix[2][0], matrix[2][1], matrix[2][2], 0,
    0, 0, 0, 1
  );
  object.position.set(pos[0], pos[1], pos[2]);
  object.quaternion.setFromRotationMatrix(rot);
  object.scale.setScalar(State.replay.aircraftDisplayLengthM);
}

function aircraftVisualMatrix(rollDeg, pitchDeg, yawDeg) {
  const bodyMatrix = attitudeMatrix(rollDeg, pitchDeg, yawDeg);
  if (AIRCRAFT_MODEL_YAW_OFFSET_DEG === 0) {
    return bodyMatrix;
  }
  return matmul3(
    bodyMatrix,
    zRotationMatrix(AIRCRAFT_MODEL_YAW_OFFSET_DEG)
  );
}

function updateFollowCamera(ownIndex, targetIndex) {
  if (!Scene.camera || !Scene.controls) {
    return;
  }
  const focal = cameraFocalPoint(ownIndex, targetIndex);
  const delta = focal.clone().sub(Scene.controls.target);
  if (delta.lengthSq() <= 1e-8) {
    return;
  }
  Scene.camera.position.add(delta);
  Scene.controls.target.copy(focal);
  Scene.controls.update();
}

function cameraFocalPoint(ownIndex, targetIndex) {
  const replay = State.replay;
  const own = replay.ownship.position[ownIndex];
  const target = replay.target.position[targetIndex];
  if (State.cameraMode === "blue") {
    return new THREE.Vector3(own[0], own[1], own[2]);
  }
  if (State.cameraMode === "red") {
    return new THREE.Vector3(target[0], target[1], target[2]);
  }
  return new THREE.Vector3(
    (own[0] + target[0]) / 2,
    (own[1] + target[1]) / 2,
    (own[2] + target[2]) / 2
  );
}

function updateTrail(line, track, simTime) {
  if (!line) {
    return;
  }
  const startTime = simTime - State.replay.defaults.trailSeconds;
  const start = lowerBound(track.time, startTime);
  const end = upperBound(track.time, simTime);
  const points = [];
  for (let i = start; i < end; i += 1) {
    const p = track.position[i];
    points.push(new THREE.Vector3(p[0], p[1], p[2]));
  }
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
}

function updateWez(mesh, track, index) {
  if (!mesh) {
    return;
  }
  const pos = track.position[index];
  const forward = forwardVector(track.yawDeg[index], track.pitchDeg[index]);
  mesh.geometry.dispose();
  mesh.geometry = buildWezGeometry(
    new THREE.Vector3(pos[0], pos[1], pos[2]),
    new THREE.Vector3(forward[0], forward[1], forward[2]),
    State.replay.defaults.wezMinRangeM,
    State.replay.defaults.wezRangeM,
    State.replay.defaults.wezAngleDeg
  );
}

function buildWezGeometry(nose, direction, minRange, maxRange, angleDeg) {
  const dir = direction.lengthSq() > 0 ? direction.clone().normalize() : new THREE.Vector3(1, 0, 0);
  const nearRange = Math.max(0, minRange);
  const farRange = Math.max(nearRange, maxRange);
  const halfAngle = THREE.MathUtils.degToRad(angleDeg / 2);
  const nearRadius = nearRange * Math.tan(halfAngle);
  const farRadius = farRange * Math.tan(halfAngle);
  let ref = new THREE.Vector3(0, 0, 1);
  if (Math.abs(dir.dot(ref)) > 0.98) {
    ref = new THREE.Vector3(0, 1, 0);
  }
  const side = new THREE.Vector3().crossVectors(dir, ref).normalize();
  const up = new THREE.Vector3().crossVectors(side, dir).normalize();
  const resolution = 48;
  const nearCenter = nose.clone().addScaledVector(dir, nearRange);
  const farCenter = nose.clone().addScaledVector(dir, farRange);
  const vertices = [];
  const indices = [];

  for (let step = 0; step < resolution; step += 1) {
    const theta = (2 * Math.PI * step) / resolution;
    const radial = side.clone().multiplyScalar(Math.cos(theta))
      .add(up.clone().multiplyScalar(Math.sin(theta)));
    const nearPoint = nearCenter.clone().addScaledVector(radial, nearRadius);
    const farPoint = farCenter.clone().addScaledVector(radial, farRadius);
    vertices.push(nearPoint.x, nearPoint.y, nearPoint.z);
    vertices.push(farPoint.x, farPoint.y, farPoint.z);
  }

  for (let step = 0; step < resolution; step += 1) {
    const next = (step + 1) % resolution;
    const nearA = step * 2;
    const farA = nearA + 1;
    const nearB = next * 2;
    const farB = nearB + 1;
    indices.push(nearA, nearB, farB, nearA, farB, farA);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function updateHud(ownIndex, targetIndex) {
  const replay = State.replay;
  const own = replay.ownship;
  const target = replay.target;
  const ownPos = own.position[ownIndex];
  const targetPos = target.position[targetIndex];
  const relative = sub(targetPos, ownPos);
  const distance = norm(relative);
  const ownForward = forwardVector(own.yawDeg[ownIndex], own.pitchDeg[ownIndex]);
  const targetForward = forwardVector(target.yawDeg[targetIndex], target.pitchDeg[targetIndex]);
  const ownAta = angleBetweenDeg(ownForward, relative);
  const targetAta = angleBetweenDeg(targetForward, scale(relative, -1));
  const ownAa = angleBetweenDeg(targetForward, scale(relative, -1));
  const ownSpeed = speedAt(own, ownIndex);
  const targetSpeed = speedAt(target, targetIndex);
  const closure = distance > 0
    ? -dot(relative, sub(velocityAt(target, targetIndex), velocityAt(own, ownIndex))) / distance
    : 0;
  const ownWez = inWez(distance, ownAta);
  const targetWez = inWez(distance, targetAta);
  const state = State.playing ? "PLAY" : "PAUSE";

  $("hud-left").textContent =
    `${state}  t=${State.simTime.toFixed(2)}s  x${State.speed.toFixed(1)}\n` +
    `Own  alt=${fmt0(ownPos[2])}m  v=${fmt1(ownSpeed)}m/s  hp=${fmtHealth(own.health[ownIndex])}\n` +
    `Tgt  alt=${fmt0(targetPos[2])}m  v=${fmt1(targetSpeed)}m/s  hp=${fmtHealth(target.health[targetIndex])}`;
  $("hud-right").textContent =
    `Range      ${fmt0(distance)} m\n` +
    `Closure    ${fmt1(closure)} m/s\n` +
    `Rel Alt    ${fmt0(targetPos[2] - ownPos[2])} m\n` +
    `Own ATA    ${fmt1(ownAta)} deg\n` +
    `Target AA  ${fmt1(ownAa)} deg\n` +
    `Own WEZ    ${ownWez ? "IN" : "out"}\n` +
    `Threat     ${targetWez ? "IN" : "out"}`;
  $("hud-right").style.color = targetWez ? "#ff6b6b" : ownWez ? "#f3b34c" : "#eef3f7";
  $("hud-bottom").textContent =
    `${replay.logs.ownship}\n${replay.logs.target}\nEnd: ${replay.endCondition}`;

  $("time-readout").textContent = `${State.simTime.toFixed(2)}s`;
  $("range-readout").textContent = `${fmt0(distance)} m`;
  $("closure-readout").textContent = `${fmt1(closure)} m/s`;
  $("relalt-readout").textContent = `${fmt0(targetPos[2] - ownPos[2])} m`;
  $("log-info").textContent = JSON.stringify({
    ownship: replay.logs.ownship,
    target: replay.logs.target,
    metadata: replay.logs.metadata,
    end: replay.endCondition,
  }, null, 2);
}

function updateTimeline() {
  const replay = State.replay;
  const denom = replay.endTime - replay.startTime;
  const ratio = denom > 0 ? (State.simTime - replay.startTime) / denom : 0;
  $("timeline").value = String(THREE.MathUtils.clamp(ratio, 0, 1));
}

function resizeRenderer() {
  if (!Scene.renderer || !Scene.camera || !Scene.root) {
    return;
  }
  const rect = Scene.root.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  Scene.renderer.setSize(width, height, false);
  Scene.camera.aspect = width / height;
  Scene.camera.updateProjectionMatrix();
  $("debug-webgl").textContent = window.DogFightViewerDebug.webglOk ? "ok" : "fail";
}

async function apiFetch(path) {
  const response = await fetch(path.replace(/^\/api/, API_BASE), { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function setStatus(message) {
  $("replay-status-text").textContent = message;
}

function disposeObject(object) {
  object.traverse?.(child => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach(material => material.dispose?.());
    } else {
      child.material?.dispose?.();
    }
  });
}

function nearestIndex(times, value) {
  return Math.max(0, Math.min(upperBound(times, value) - 1, times.length - 1));
}

function lowerBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function upperBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] <= target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

function attitudeMatrix(rollDeg, pitchDeg, yawDeg) {
  const roll = THREE.MathUtils.degToRad(rollDeg);
  const pitch = THREE.MathUtils.degToRad(pitchDeg);
  const yaw = THREE.MathUtils.degToRad(90 - yawDeg);
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const rotX = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];
  const rotY = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]];
  const rotZ = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]];
  return matmul3(matmul3(rotZ, rotY), rotX);
}

function zRotationMatrix(deg) {
  const rad = THREE.MathUtils.degToRad(deg);
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}

function matmul3(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      out[row][col] = a[row][0] * b[0][col] +
        a[row][1] * b[1][col] +
        a[row][2] * b[2][col];
    }
  }
  return out;
}

function forwardVector(yawDeg, pitchDeg) {
  const matrix = attitudeMatrix(0, pitchDeg, yawDeg);
  const direction = [matrix[0][0], matrix[1][0], matrix[2][0]];
  const length = norm(direction);
  return length > 0 ? scale(direction, 1 / length) : [1, 0, 0];
}

function speedAt(track, index) {
  if (track.time.length < 2) {
    return 0;
  }
  const prev = Math.max(0, index - 1);
  const next = Math.min(track.time.length - 1, index + 1);
  const dt = track.time[next] - track.time[prev];
  return dt > 0 ? norm(sub(track.position[next], track.position[prev])) / dt : 0;
}

function velocityAt(track, index) {
  if (track.time.length < 2) {
    return [0, 0, 0];
  }
  const prev = Math.max(0, index - 1);
  const next = Math.min(track.time.length - 1, index + 1);
  const dt = track.time[next] - track.time[prev];
  return dt > 0 ? scale(sub(track.position[next], track.position[prev]), 1 / dt) : [0, 0, 0];
}

function angleBetweenDeg(first, second) {
  const firstNorm = norm(first);
  const secondNorm = norm(second);
  if (firstNorm <= 0 || secondNorm <= 0) {
    return 0;
  }
  const cosine = THREE.MathUtils.clamp(dot(first, second) / (firstNorm * secondNorm), -1, 1);
  return THREE.MathUtils.radToDeg(Math.acos(cosine));
}

function inWez(rangeM, ataDeg) {
  const defs = State.replay.defaults;
  return rangeM >= defs.wezMinRangeM &&
    rangeM <= defs.wezRangeM &&
    ataDeg <= Math.max(0, defs.wezAngleDeg / 2);
}

function sub(first, second) {
  return [first[0] - second[0], first[1] - second[1], first[2] - second[2]];
}

function scale(vector, scalar) {
  return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar];
}

function dot(first, second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

function lerp(start, end, ratio) {
  return start + (end - start) * ratio;
}

function fmt0(value) {
  return Number.isFinite(value) ? value.toFixed(0).padStart(6, " ") : "n/a";
}

function fmt1(value) {
  return Number.isFinite(value) ? value.toFixed(1).padStart(6, " ") : "n/a";
}

function fmtHealth(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(3);
}
