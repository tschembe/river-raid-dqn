// RIVER RAID - Clean Engineering & RL Simulation View
(() => {
  const canvas = document.getElementById('gameCanvas');
  const ctx = canvas.getContext('2d');
  const muteStatusEl = document.getElementById('muteStatus');

  // RL UI Elements
  const rlModeBadge = document.getElementById('rlModeBadge');
  const btnToggleAI = document.getElementById('btnToggleAI');
  const btnWarp = document.getElementById('btnWarp');
  const btnSensors = document.getElementById('btnSensors');
  const btnLearn = document.getElementById('btnLearn');
  const btnSaveBrain = document.getElementById('btnSaveBrain');
  const btnResetBrain = document.getElementById('btnResetBrain');
  const elEpisode = document.getElementById('rlEpisode');
  const elEpsilon = document.getElementById('rlEpsilon');
  const elReward = document.getElementById('rlReward');
  const elSimScore = document.getElementById('simScore');
  const elBestScore = document.getElementById('rlBestScore');
  const elSimSector = document.getElementById('simSector');
  const elMemory = document.getElementById('rlMemory');
  const elAction = document.getElementById('rlAction');
  const elStateDump = document.getElementById('stateVectorDump');

  // Physics constants mirror river_raid_env.py so Python-trained weights see the same game.
  const W = canvas.width;
  const PLAY_H = canvas.height;
  const SLICE_H = 4;
  const START_Y = PLAY_H - 72;
  const STEER_STEP = 4.2;
  const RAY_STEP = 10;
  const RAY_MAX = 260;

  // Blue River & Green Land Color Scheme
  const COLORS = {
    water: '#2563eb',
    gridLine: 'rgba(255, 255, 255, 0.12)',
    land: '#2e8b3c',
    shoreline: '#1f6329',
    road: '#6e7781',
    jet: '#ffffff',
    enemyJet: '#dc2626'
  };

  // --- Optional Minimal Audio (Muted by default for engineering UI) ---
  class SoundFX {
    constructor() {
      this.muted = true;
    }
    init() {}
    toggleMute() {
      this.muted = !this.muted;
      muteStatusEl.textContent = this.muted ? 'OFF' : 'ON';
    }
    updateEngine() {}
    playExplosion() {}
  }

  const sfx = new SoundFX();

  // --- RL Agent & Configuration ---
  const agent = new window.RiverRaidDQNAgent();
  const rlConfig = {
    enabled: true, // Start directly in RL Autopilot mode
    warpOptions: [1, 4, 16, 40],
    warpIdx: 0,
    warpFactor: 1,
    showSensors: true,
    learning: true,
    lastSensorRays: [],
    lastBestTargetX: W / 2,
    lastNearestJet: null,
    lastObs: new Float32Array(13)
  };

  function syncModeUI() {
    rlModeBadge.textContent = rlConfig.enabled ? 'RL_AUTOPILOT' : 'MANUAL_CTRL';
    rlModeBadge.classList.toggle('ai-on', rlConfig.enabled);
    btnToggleAI.classList.toggle('active', rlConfig.enabled);
    btnToggleAI.textContent = rlConfig.enabled ? 'Disable RL Agent [T]' : 'Enable RL Agent [T]';
  }

  function toggleAI() {
    rlConfig.enabled = !rlConfig.enabled;
    syncModeUI();
    if (rlConfig.enabled && state.mode !== 'PLAYING') {
      startNewGame();
    }
  }

  btnToggleAI.addEventListener('click', toggleAI);

  btnWarp.addEventListener('click', () => {
    rlConfig.warpIdx = (rlConfig.warpIdx + 1) % rlConfig.warpOptions.length;
    rlConfig.warpFactor = rlConfig.warpOptions[rlConfig.warpIdx];
    btnWarp.textContent = `Sim Speed: ${rlConfig.warpFactor}x`;
    btnWarp.classList.toggle('active', rlConfig.warpFactor > 1);
  });

  btnSensors.addEventListener('click', () => {
    rlConfig.showSensors = !rlConfig.showSensors;
    btnSensors.textContent = `Lidar Vectors: ${rlConfig.showSensors ? 'ON' : 'OFF'}`;
    btnSensors.classList.toggle('active', rlConfig.showSensors);
  });

  function syncLearnUI() {
    btnLearn.textContent = `Online Learning: ${rlConfig.learning ? 'ON' : 'OFF'}`;
    btnLearn.classList.toggle('active', rlConfig.learning);
  }

  btnLearn.addEventListener('click', () => {
    rlConfig.learning = !rlConfig.learning;
    syncLearnUI();
  });

  // A Python-trained brain is already converged: watch it instead of fine-tuning with browser SGD.
  agent.onPythonBrainLoaded = () => {
    rlConfig.learning = false;
    syncLearnUI();
  };

  // When served by river_raid.py, verify the browser physics still match the Python env.
  fetch('/api/config')
    .then((res) => (res.ok ? res.json() : null))
    .then((cfg) => {
      if (!cfg) return;
      const local = {
        W, PLAY_H, SLICE_H, playerY: START_Y, playerW: player.w, playerH: player.h,
        minSpeed: player.minSpeed, maxSpeed: player.maxSpeed, cruiseSpeed: player.cruiseSpeed,
        steerStep: STEER_STEP, rayStep: RAY_STEP, rayMax: RAY_MAX
      };
      const drift = Object.keys(local).filter((k) => cfg[k] !== undefined && cfg[k] !== local[k]);
      if (drift.length) {
        console.warn('Browser physics differ from Python env:', drift.map((k) => `${k}: js=${local[k]} py=${cfg[k]}`));
      }
    })
    .catch(() => {});

  btnSaveBrain.addEventListener('click', () => {
    agent.saveToStorage();
    btnSaveBrain.textContent = 'Saved';
    setTimeout(() => { btnSaveBrain.textContent = 'Save Weights'; }, 900);
  });

  btnResetBrain.addEventListener('click', () => {
    agent.resetBrain();
    updateRLDashboard();
  });

  // --- Input Handling ---
  const keys = {};
  window.addEventListener('keydown', (e) => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
      e.preventDefault();
    }

    if (!keys[e.code]) {
      if (e.code === 'KeyP') {
        state.paused = !state.paused;
      }
      if (e.code === 'KeyM') {
        sfx.toggleMute();
      }
      if (e.code === 'KeyT') {
        toggleAI();
      }
      if (e.code === 'Space' || e.code === 'Enter') {
        startNewGame();
      }
    }
    keys[e.code] = true;
  });

  window.addEventListener('keyup', (e) => {
    keys[e.code] = false;
  });

  // --- Simulation State ---
  const state = {
    mode: 'PLAYING',
    paused: false,
    score: 0,
    highScore: Number(localStorage.getItem('riverRaidHighScore') || 0),
    sector: 1,
    checkpointSector: 1,
    distanceInSector: 0,
    sectorLength: 2600,
    scrollPos: 0
  };

  const player = {
    x: W / 2,
    y: START_Y,
    w: 28,
    h: 28,
    speed: 3.2,
    minSpeed: 1.8,
    maxSpeed: 5.8,
    cruiseSpeed: 3.2,
    bank: 0
  };

  let slices = [];
  let entities = [];

  const riverGen = {
    center: W / 2,
    targetCenter: W / 2,
    width: 280,
    targetWidth: 280,
    islandWidth: 0,
    targetIslandWidth: 0,
    islandTimer: 0,
    bridgePending: false,
    bridgeCountdown: 0,
    spawnCooldown: 0
  };

  function resetRiverGenerator() {
    riverGen.center = W / 2;
    riverGen.targetCenter = W / 2;
    riverGen.width = 290;
    riverGen.targetWidth = 290;
    riverGen.islandWidth = 0;
    riverGen.targetIslandWidth = 0;
    riverGen.islandTimer = 140;
    riverGen.bridgePending = false;
    riverGen.bridgeCountdown = 0;
    riverGen.spawnCooldown = 35;

    slices = [];
    entities = [];

    const numSlices = Math.ceil(PLAY_H / SLICE_H) + 4;
    for (let i = 0; i < numSlices; i++) {
      const y = PLAY_H - i * SLICE_H;
      slices.push({
        y,
        center: W / 2,
        width: 290,
        islandWidth: 0,
        isRoad: false
      });
    }
  }

  function generateTopSlice() {
    state.distanceInSector += SLICE_H;

    if (state.distanceInSector >= state.sectorLength && !riverGen.bridgePending) {
      riverGen.bridgePending = true;
      riverGen.bridgeCountdown = 45;
      riverGen.targetCenter = W / 2;
      riverGen.targetWidth = 116;
      riverGen.targetIslandWidth = 0;
    }

    if (riverGen.bridgePending) {
      riverGen.bridgeCountdown--;
      riverGen.center += (W / 2 - riverGen.center) * 0.12;
      riverGen.width += (116 - riverGen.width) * 0.14;
      riverGen.islandWidth = Math.max(0, riverGen.islandWidth - 8);
    } else {
      if (Math.random() < 0.04) {
        const margin = 130;
        riverGen.targetCenter = margin + Math.random() * (W - margin * 2);
      }
      if (Math.random() < 0.045) {
        const minW = Math.max(125, 210 - state.sector * 10);
        const maxW = 420;
        riverGen.targetWidth = minW + Math.random() * (maxW - minW);
      }

      riverGen.islandTimer--;
      if (riverGen.islandTimer <= 0) {
        if (riverGen.targetIslandWidth === 0 && riverGen.width > 300 && state.sector >= 1 && Math.random() < 0.45) {
          riverGen.targetIslandWidth = 65 + Math.random() * 65;
          riverGen.islandTimer = 90 + Math.floor(Math.random() * 90);
        } else {
          riverGen.targetIslandWidth = 0;
          riverGen.islandTimer = 110 + Math.floor(Math.random() * 120);
        }
      }

      if (riverGen.width < 270) {
        riverGen.targetIslandWidth = 0;
      }

      riverGen.center += (riverGen.targetCenter - riverGen.center) * 0.035;
      riverGen.width += (riverGen.targetWidth - riverGen.width) * 0.04;
      riverGen.islandWidth += (riverGen.targetIslandWidth - riverGen.islandWidth) * 0.06;
      if (riverGen.islandWidth < 6 && riverGen.targetIslandWidth === 0) {
        riverGen.islandWidth = 0;
      }
    }

    const halfW = riverGen.width / 2;
    if (riverGen.center - halfW < 36) riverGen.center = 36 + halfW;
    if (riverGen.center + halfW > W - 36) riverGen.center = W - 36 - halfW;

    const topY = slices[slices.length - 1].y - SLICE_H;
    const isRoadBand = riverGen.bridgePending && riverGen.bridgeCountdown >= 6 && riverGen.bridgeCountdown <= 18;

    const slice = {
      y: topY,
      center: riverGen.center,
      width: riverGen.width,
      islandWidth: riverGen.islandWidth,
      isRoad: isRoadBand
    };
    slices.push(slice);

    if (riverGen.bridgePending && riverGen.bridgeCountdown <= 0) {
      riverGen.bridgePending = false;
      state.distanceInSector = 0;
      riverGen.targetWidth = 280;
      state.sector++;
      state.checkpointSector = state.sector;
      addScore(500);
    } else if (!riverGen.bridgePending) {
      spawnRiverEntities(slice);
    }
  }

  function spawnRiverEntities(slice) {
    riverGen.spawnCooldown--;
    if (riverGen.spawnCooldown > 0) return;

    const fromLeft = Math.random() < 0.5;
    entities.push({
      type: 'JET',
      x: fromLeft ? 20 : W - 20,
      y: slice.y,
      w: 30,
      h: 18,
      vx: (fromLeft ? 1 : -1) * (3.2 + Math.min(2.2, state.sector * 0.25)),
      dir: fromLeft ? 1 : -1,
      points: 100
    });

    const baseGap = Math.max(22, 48 - state.sector * 2.5);
    riverGen.spawnCooldown = Math.floor(baseGap + Math.random() * 25);
  }

  function addScore(pts) {
    state.score += pts;
    if (state.score > state.highScore) {
      state.highScore = state.score;
      localStorage.setItem('riverRaidHighScore', String(state.highScore));
    }
  }

  function getSliceAtY(y) {
    let closest = slices[0];
    let bestDiff = 9999;
    for (let i = 0; i < slices.length; i++) {
      const d = Math.abs(slices[i].y - y);
      if (d < bestDiff) {
        bestDiff = d;
        closest = slices[i];
      }
    }
    return closest;
  }

  function isTouchingLand(x, y, halfW) {
    const s = getSliceAtY(y);
    if (!s) return false;
    const leftBank = s.center - s.width / 2;
    const rightBank = s.center + s.width / 2;
    if (x - halfW < leftBank || x + halfW > rightBank) {
      return true;
    }
    if (s.islandWidth > 6) {
      const islLeft = s.center - s.islandWidth / 2;
      const islRight = s.center + s.islandWidth / 2;
      if (x + halfW > islLeft && x - halfW < islRight) {
        return true;
      }
    }
    return false;
  }

  function rectsOverlap(ax, ay, aw, ah, bx, by, bw, bh) {
    return (
      Math.abs(ax - bx) * 2 < aw + bw &&
      Math.abs(ay - by) * 2 < ah + bh
    );
  }

  // --- RL Sensor Observation & Safe Channel Helper ---
  function getSafeCenterForSlice(s, refX) {
    if (!s) return W / 2;
    const leftBank = s.center - s.width / 2;
    const rightBank = s.center + s.width / 2;
    if (s.islandWidth > 6) {
      const islLeft = s.center - s.islandWidth / 2;
      const islRight = s.center + s.islandWidth / 2;
      const leftChannelCenter = (leftBank + islLeft) / 2;
      const rightChannelCenter = (islRight + rightBank) / 2;
      return Math.abs(refX - leftChannelCenter) <= Math.abs(refX - rightChannelCenter)
        ? leftChannelCenter
        : rightChannelCenter;
    }
    return s.center;
  }

  function getSafeMarginAt(x, y) {
    const s = getSliceAtY(y);
    if (!s) return 0;
    const leftBank = s.center - s.width / 2;
    const rightBank = s.center + s.width / 2;
    if (s.islandWidth > 6) {
      const islLeft = s.center - s.islandWidth / 2;
      const islRight = s.center + s.islandWidth / 2;
      if (x < s.center) {
        return Math.min(x - leftBank, islLeft - x);
      }
      return Math.min(x - islRight, rightBank - x);
    }
    return Math.min(x - leftBank, rightBank - x);
  }

  function castSensorRay(angleRad, maxDist = RAY_MAX) {
    const step = RAY_STEP;
    const dx = Math.sin(angleRad);
    const dy = -Math.cos(angleRad);
    for (let d = step; d <= maxDist; d += step) {
      const rx = player.x + dx * d;
      const ry = player.y + dy * d;
      if (rx < 0 || rx > W || ry < 0 || ry > PLAY_H) {
        return { dist: d / maxDist, hitX: rx, hitY: ry, hitType: 'LAND' };
      }
      if (isTouchingLand(rx, ry, 4)) {
        return { dist: d / maxDist, hitX: rx, hitY: ry, hitType: 'LAND' };
      }
      for (let i = 0; i < entities.length; i++) {
        const e = entities[i];
        if (Math.abs(rx - e.x) < e.w * 0.6 && Math.abs(ry - e.y) < e.h * 0.8) {
          return { dist: d / maxDist, hitX: rx, hitY: ry, hitType: 'JET' };
        }
      }
    }
    return {
      dist: 1.0,
      hitX: player.x + dx * maxDist,
      hitY: player.y + dy * maxDist,
      hitType: 'NONE'
    };
  }

  function getRLObservation() {
    const anglesDeg = [-75, -50, -25, 0, 25, 50, 75];
    const rays = [];
    const obs = new Float32Array(13);

    for (let i = 0; i < anglesDeg.length; i++) {
      const rad = (anglesDeg[i] * Math.PI) / 180;
      const r = castSensorRay(rad);
      rays.push(r);
      obs[i] = r.dist;
    }
    rlConfig.lastSensorRays = rays;

    const sliceNow = getSliceAtY(player.y);
    const sliceNear = getSliceAtY(player.y - 70);
    const sliceFar = getSliceAtY(player.y - 160);

    const safeNowX = getSafeCenterForSlice(sliceNow, player.x);
    const safeNearX = getSafeCenterForSlice(sliceNear, safeNowX);
    const safeFarX = getSafeCenterForSlice(sliceFar, safeNearX);
    rlConfig.lastBestTargetX = safeNearX;

    obs[7] = (safeNowX - player.x) / (W * 0.5);
    obs[8] = (safeNearX - player.x) / (W * 0.5);
    obs[9] = (safeFarX - player.x) / (W * 0.5);

    let nearestJet = null;
    let bestDy = 9999;
    for (const e of entities) {
      const dy = player.y - e.y;
      if (dy > -24 && dy < 280 && dy < bestDy) {
        bestDy = dy;
        nearestJet = e;
      }
    }
    rlConfig.lastNearestJet = nearestJet;

    if (nearestJet) {
      obs[10] = (nearestJet.x - player.x) / (W * 0.5);
      obs[11] = (player.y - nearestJet.y) / 280;
    } else {
      obs[10] = 0;
      obs[11] = 1.0;
    }

    obs[12] = (player.speed - player.minSpeed) / (player.maxSpeed - player.minSpeed);
    rlConfig.lastObs = obs;
    return obs;
  }

  function startNewGame() {
    state.score = 0;
    state.sector = 1;
    state.checkpointSector = 1;
    state.distanceInSector = 0;
    state.paused = false;
    player.x = W / 2;
    player.y = START_Y;
    player.speed = player.cruiseSpeed;
    player.bank = 0;
    resetRiverGenerator();
    state.mode = 'PLAYING';
  }

  function stepPhysics() {
    if (state.paused) return;

    const prevState = getRLObservation();
    let action = 0;

    if (rlConfig.enabled) {
      action = agent.act(prevState, rlConfig.learning);

      player.bank = 0;
      if (action === 1) {
        player.x -= STEER_STEP;
        player.bank = -1;
      } else if (action === 2) {
        player.x += STEER_STEP;
        player.bank = 1;
      }

      if (action === 3) {
        player.speed += (player.maxSpeed - player.speed) * 0.12;
      } else if (action === 4) {
        player.speed += (player.minSpeed - player.speed) * 0.14;
      } else {
        player.speed += (player.cruiseSpeed - player.speed) * 0.1;
      }
    } else {
      player.bank = 0;
      if (keys['ArrowLeft'] || keys['KeyA']) {
        player.x -= STEER_STEP;
        player.bank = -1;
      }
      if (keys['ArrowRight'] || keys['KeyD']) {
        player.x += STEER_STEP;
        player.bank = 1;
      }

      if (keys['ArrowUp'] || keys['KeyW']) {
        player.speed += (player.maxSpeed - player.speed) * 0.12;
      } else if (keys['ArrowDown'] || keys['KeyS']) {
        player.speed += (player.minSpeed - player.speed) * 0.14;
      } else {
        player.speed += (player.cruiseSpeed - player.speed) * 0.1;
      }
    }

    const speedRatio = (player.speed - player.minSpeed) / (player.maxSpeed - player.minSpeed);

    if (Math.floor(state.scrollPos / 20) < Math.floor((state.scrollPos + player.speed) / 20)) {
      addScore(10);
    }

    state.scrollPos += player.speed;
    for (let i = 0; i < slices.length; i++) {
      slices[i].y += player.speed;
    }
    while (slices.length > 0 && slices[0].y > PLAY_H + SLICE_H * 2) {
      slices.shift();
    }
    while (slices[slices.length - 1].y > -SLICE_H * 2) {
      generateTopSlice();
    }

    let crashed = isTouchingLand(player.x, player.y, player.w * 0.42);

    for (let i = entities.length - 1; i >= 0; i--) {
      const e = entities[i];
      e.y += player.speed;

      if (e.type === 'JET') {
        e.x += e.vx;
        if (e.x < -30) e.x = W + 30;
        if (e.x > W + 30) e.x = -30;
      }

      if (!crashed && rectsOverlap(player.x, player.y, player.w * 0.78, player.h * 0.78, e.x, e.y, e.w * 0.85, e.h * 0.85)) {
        crashed = true;
      }

      if (e.y > PLAY_H + 60) {
        entities.splice(i, 1);
      }
    }

    if (rlConfig.enabled) {
      if (crashed) {
        if (rlConfig.learning) {
          agent.remember(prevState, action, -12.0, prevState, true);
          agent.replay();
        }
        agent.onEpisodeEnd(state.score);
        startNewGame();
        return;
      } else {
        const nextState = getRLObservation();
        const margin = getSafeMarginAt(player.x, player.y);
        const sliceNear = getSliceAtY(player.y - 65);
        const safeCenter = getSafeCenterForSlice(sliceNear, player.x);
        const centerError = Math.abs(player.x - safeCenter) / 140;

        let reward = 0.25 + (1.0 - Math.min(1.2, centerError)) * 0.35 + speedRatio * 0.1;
        if (margin < 32) {
          reward -= (32 - margin) * 0.04;
        }
        if (rlConfig.lastNearestJet) {
          const j = rlConfig.lastNearestJet;
          const dy = player.y - j.y;
          const dx = Math.abs(player.x - j.x);
          if (dy > 0 && dy < 75 && dx < 42) {
            reward -= 0.45;
          }
        }

        if (rlConfig.learning) {
          agent.remember(prevState, action, reward, nextState, false);
          agent.replay();
        } else {
          agent.episodeReward += reward;
        }
      }
    } else if (crashed) {
      startNewGame();
    }
  }

  // --- Clean Engineering Rendering ---
  function drawTerrain() {
    ctx.fillStyle = COLORS.water;
    ctx.fillRect(0, 0, W, PLAY_H);

    // Subtle engineering coordinate grid
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 50) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, PLAY_H);
      ctx.stroke();
    }
    const offset = Math.floor(state.scrollPos % 50);
    for (let gy = offset; gy < PLAY_H; gy += 50) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(W, gy);
      ctx.stroke();
    }

    for (let i = 0; i < slices.length; i++) {
      const s = slices[i];
      const y = Math.floor(s.y);
      const h = SLICE_H + 1;
      const leftBank = Math.floor(s.center - s.width / 2);
      const rightBank = Math.ceil(s.center + s.width / 2);

      ctx.fillStyle = s.isRoad ? COLORS.road : COLORS.land;
      ctx.fillRect(0, y, leftBank, h);
      ctx.fillRect(rightBank, y, W - rightBank, h);

      ctx.fillStyle = COLORS.shoreline;
      ctx.fillRect(leftBank - 2, y, 2, h);
      ctx.fillRect(rightBank, y, 2, h);

      if (s.islandWidth > 6) {
        const islL = Math.floor(s.center - s.islandWidth / 2);
        const islW = Math.ceil(s.islandWidth);
        ctx.fillStyle = COLORS.land;
        ctx.fillRect(islL, y, islW, h);
        ctx.fillStyle = COLORS.shoreline;
        ctx.fillRect(islL, y, 2, h);
        ctx.fillRect(islL + islW - 2, y, 2, h);
      }
    }
  }

  function drawEntity(e) {
    const x = Math.floor(e.x);
    const y = Math.floor(e.y);

    if (e.type === 'JET') {
      // Clean bounding box + vector shape for obstacle jet
      ctx.strokeStyle = 'rgba(248, 81, 73, 0.45)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x - e.w / 2, y - e.h / 2, e.w, e.h);

      ctx.fillStyle = COLORS.enemyJet;
      ctx.fillRect(x - 12, y - 2, 24, 4);
      ctx.fillRect(x - 3, y - 7, 6, 14);
    }
  }

  function drawRLSensors() {
    if (!rlConfig.showSensors) return;

    for (const r of rlConfig.lastSensorRays) {
      ctx.strokeStyle = r.dist < 0.35
        ? 'rgba(255, 80, 80, 0.9)'
        : r.hitType === 'JET'
          ? 'rgba(255, 220, 40, 0.9)'
          : 'rgba(255, 255, 255, 0.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(player.x, player.y);
      ctx.lineTo(r.hitX, r.hitY);
      ctx.stroke();

      ctx.fillStyle = r.dist < 0.35 ? '#ff3b30' : '#ffffff';
      ctx.fillRect(r.hitX - 2, r.hitY - 2, 4, 4);
    }

    // Target waypoint marker
    const tx = rlConfig.lastBestTargetX;
    const ty = player.y - 70;
    ctx.strokeStyle = '#ffea00';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(tx - 5, ty - 5, 10, 10);
  }

  function drawPlayer() {
    const x = Math.floor(player.x);
    const y = Math.floor(player.y);

    // Agent bounding box + clean triangle geometry
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x - player.w / 2, y - player.h / 2, player.w, player.h);

    ctx.fillStyle = COLORS.jet;
    ctx.beginPath();
    ctx.moveTo(x, y - 12);
    ctx.lineTo(x - 11, y + 10);
    ctx.lineTo(x, y + 5);
    ctx.lineTo(x + 11, y + 10);
    ctx.closePath();
    ctx.fill();
  }

  function updateRLDashboard() {
    elEpisode.textContent = String(agent.episode);
    elEpsilon.textContent = rlConfig.learning ? agent.epsilon.toFixed(3) : '0 (greedy)';
    elReward.textContent = agent.episodeReward.toFixed(1);
    elSimScore.textContent = String(state.score);
    elBestScore.textContent = String(agent.bestScore);
    elSimSector.textContent = String(state.sector);
    elMemory.textContent = String(agent.memory.length);
    elAction.textContent = rlConfig.enabled ? window.ACTION_NAMES[agent.lastAction] : 'MANUAL';

    // Update Q-value bars
    const qVals = agent.lastQValues;
    let minQ = Infinity;
    let maxQ = -Infinity;
    for (let i = 0; i < 5; i++) {
      const v = qVals[i] || 0;
      if (v < minQ) minQ = v;
      if (v > maxQ) maxQ = v;
    }
    const span = Math.max(0.5, maxQ - minQ);

    for (let i = 0; i < 5; i++) {
      const v = qVals[i] || 0;
      const pct = Math.max(4, Math.min(100, ((v - minQ) / span) * 100));
      const bar = document.getElementById(`qBar${i}`);
      const num = document.getElementById(`qVal${i}`);
      if (bar && num) {
        bar.style.width = `${pct.toFixed(0)}%`;
        bar.classList.toggle('best', i === agent.lastAction);
        num.textContent = v.toFixed(2);
      }
    }

    // Format 13D state vector
    const o = rlConfig.lastObs;
    elStateDump.textContent =
      `lidar[0..6] : [${Array.from(o.slice(0, 7)).map(v => v.toFixed(2)).join(', ')}]\n` +
      `ch_offset   : [now:${o[7].toFixed(2)}, near:${o[8].toFixed(2)}, far:${o[9].toFixed(2)}]\n` +
      `jet_delta   : [dx:${o[10].toFixed(2)}, dy:${o[11].toFixed(2)}]\n` +
      `norm_speed  : ${o[12].toFixed(2)}`;
  }

  function loop() {
    const steps = rlConfig.enabled ? rlConfig.warpFactor : 1;
    for (let s = 0; s < steps; s++) {
      stepPhysics();
    }

    drawTerrain();
    for (const e of entities) drawEntity(e);
    drawRLSensors();
    drawPlayer();
    updateRLDashboard();

    requestAnimationFrame(loop);
  }

  syncModeUI();
  syncLearnUI();
  resetRiverGenerator();
  updateRLDashboard();
  requestAnimationFrame(loop);
})();
