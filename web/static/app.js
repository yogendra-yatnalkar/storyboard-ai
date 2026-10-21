// Storyboard AI Studio - Client Application
(function () {
  'use strict';

  // State
  let currentRunId = null;
  let eventSource = null;
  let currentRunState = null;
  let activeTab = 'final_movie';
  let activeScreenplay = [];

  // DOM Elements
  const promptInput = document.getElementById('prompt-input');
  const modeSelect = document.getElementById('mode-select');
  const btnStart = document.getElementById('btn-start-production');
  const systemBadge = document.getElementById('system-badge');
  const systemStatusText = document.getElementById('system-status-text');

  const metaBar = document.getElementById('run-meta-bar');
  const metaRunId = document.getElementById('meta-run-id');
  const metaFolderName = document.getElementById('meta-folder-name');
  const metaAssetCount = document.getElementById('meta-asset-count');

  const hitlPanel = document.getElementById('hitl-panel');
  const hitlPromptText = document.getElementById('hitl-prompt-text');
  const screenplayContainer = document.getElementById('screenplay-container');
  const hitlFeedbackInput = document.getElementById('hitl-feedback-input');
  const btnApproveScreenplay = document.getElementById('btn-approve-screenplay');
  const btnCancelRun = document.getElementById('btn-cancel-run');

  // Episodic Parts Section
  const partsSection = document.getElementById('parts-production-section');
  const partsCardsContainer = document.getElementById('parts-cards-container');
  const partsOverallBadge = document.getElementById('parts-overall-badge');

  // Live Working Output Player (Handles both Video and Image)
  const liveVideoPlayer = document.getElementById('live-video-player');
  const liveVideoSource = document.getElementById('live-video-source');
  const liveImagePlayer = document.getElementById('live-image-player');
  const liveSceneTag = document.getElementById('live-scene-tag');
  const liveSceneCaption = document.getElementById('live-scene-caption');
  const liveOverlayPill = document.getElementById('live-overlay-pill');
  const liveOverlayDot = document.getElementById('live-overlay-dot');
  const liveOverlayText = document.getElementById('live-overlay-text');

  const terminalLogs = document.getElementById('terminal-logs');
  const btnClearLogs = document.getElementById('btn-clear-logs');

  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  const masterShowcase = document.getElementById('master-showcase');
  const masterEmpty = document.getElementById('master-empty');
  const finalMasterPlayer = document.getElementById('final-master-player');
  const finalMasterSource = document.getElementById('final-master-source');
  const masterMeta = document.getElementById('master-meta');
  const btnDownloadMaster = document.getElementById('btn-download-master');

  const partMastersGrid = document.getElementById('part-masters-grid');
  const subscenesGrid = document.getElementById('subscenes-grid');
  const gridsPanelsGrid = document.getElementById('grids-panels-grid');
  const referencesGrid = document.getElementById('references-grid');
  const folderDumpList = document.getElementById('folder-dump-list');

  const badgeMaster = document.getElementById('badge-master');
  const badgeParts = document.getElementById('badge-parts');
  const badgeSubscenes = document.getElementById('badge-subscenes');
  const badgePanels = document.getElementById('badge-panels');
  const badgeRefs = document.getElementById('badge-refs');
  const badgeDump = document.getElementById('badge-dump');

  const historyModal = document.getElementById('history-modal');
  const btnOpenHistory = document.getElementById('btn-open-history');
  const btnCloseHistory = document.getElementById('btn-close-history');
  const historyFoldersList = document.getElementById('history-folders-list');

  const lightboxModal = document.getElementById('lightbox-modal');
  const btnCloseLightbox = document.getElementById('btn-close-lightbox');
  const lightboxBody = document.getElementById('lightbox-body');

  // Budget
  const budgetTokens = document.getElementById('budget-tokens');
  const budgetImages = document.getElementById('budget-images');
  const budgetVeo = document.getElementById('budget-veo');
  const budgetTotal = document.getElementById('budget-total');

  // -------------------------------------------------------------------------
  // Initialization & Setup
  // -------------------------------------------------------------------------
  function init() {
    setupPromptChips();
    setupTabSwitching();
    setupEventListeners();
    fetchConfig();
  }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config');
      if (res.ok) {
        const data = await res.json();
        const pill = document.getElementById('config-pill');
        if (pill) {
          pill.textContent = `${data.output_width}x${data.output_height} | Veo: ${data.enable_veo ? 'ON' : 'OFF'} | SAM 3: ${data.enable_sam ? 'ON' : 'OFF'}`;
        }
      }
    } catch (e) {
      console.warn('Could not fetch config:', e);
    }
  }

  function setupPromptChips() {
    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        promptInput.value = chip.getAttribute('data-prompt');
        promptInput.focus();
      });
    });
  }

  function setupTabSwitching() {
    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        setActiveTab(targetTab);
      });
    });
  }

  function setActiveTab(targetTab) {
    activeTab = targetTab;
    tabButtons.forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === targetTab));
    tabPanes.forEach(p => {
      p.style.display = p.id === `tab-${targetTab}` ? 'block' : 'none';
    });
  }

  function setupEventListeners() {
    btnStart.addEventListener('click', handleStartProduction);
    btnApproveScreenplay.addEventListener('click', handleApproveScreenplay);
    btnCancelRun.addEventListener('click', handleCancelRun);
    btnClearLogs.addEventListener('click', () => { terminalLogs.innerHTML = ''; });

    btnOpenHistory.addEventListener('click', openHistoryModal);
    btnCloseHistory.addEventListener('click', () => { historyModal.style.display = 'none'; });
    btnCloseLightbox.addEventListener('click', () => { lightboxModal.style.display = 'none'; });

    window.addEventListener('click', (e) => {
      if (e.target === historyModal) historyModal.style.display = 'none';
      if (e.target === lightboxModal) lightboxModal.style.display = 'none';
    });
  }

  // -------------------------------------------------------------------------
  // Stepper & Status UI
  // -------------------------------------------------------------------------
  function updateStepper(stepIndex, state) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (!el) continue;
      el.className = 'step-item';
      if (i < stepIndex) {
        el.classList.add('done');
      } else if (i === stepIndex) {
        if (state === 'waiting') el.classList.add('waiting');
        else el.classList.add('active');
      }
    }
  }

  function setSystemStatus(status, label) {
    systemBadge.className = 'badge';
    let text = label || status;

    switch (status) {
      case 'PLANNING':
        systemBadge.classList.add('badge-planning');
        text = 'Planning & Grounding';
        updateStepper(1, 'active');
        btnStart.disabled = true;
        break;
      case 'AWAITING_APPROVAL':
        systemBadge.classList.add('badge-hitl');
        text = 'Waiting for Review';
        updateStepper(2, 'waiting');
        btnStart.disabled = true;
        break;
      case 'PRODUCING_VIDEO':
        systemBadge.classList.add('badge-producing');
        text = 'Producing Scenes';
        updateStepper(3, 'active');
        btnStart.disabled = true;
        break;
      case 'COMPLETED':
        systemBadge.classList.add('badge-completed');
        text = 'Completed';
        updateStepper(4, 'done');
        btnStart.disabled = false;
        break;
      case 'FAILED':
        systemBadge.classList.add('badge-failed');
        text = 'Failed';
        btnStart.disabled = false;
        break;
      case 'CANCELLED':
        systemBadge.classList.add('badge-failed');
        text = 'Cancelled';
        btnStart.disabled = false;
        break;
      default:
        systemBadge.classList.add('badge-idle');
        text = 'Ready';
        btnStart.disabled = false;
    }
    systemStatusText.textContent = text;
  }

  // -------------------------------------------------------------------------
  // API Actions
  // -------------------------------------------------------------------------
  async function handleStartProduction() {
    const topic = promptInput.value.trim();
    if (!topic) {
      alert('Please enter a video topic or script to produce.');
      promptInput.focus();
      return;
    }

    const verbatim = modeSelect.value === 'verbatim';

    btnStart.disabled = true;
    setSystemStatus('PLANNING');
    resetRunUI();

    try {
      const res = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topic,
          verbatim_mode: verbatim,
          requested_parts: null
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to start run.');
      }

      const data = await res.json();
      currentRunId = data.run_id;

      metaBar.style.display = 'flex';
      metaRunId.textContent = currentRunId;
      metaFolderName.textContent = data.folder_name;

      appendLog(`Run initialized: ${currentRunId}`, 'System', 'success');
      appendLog(`Output folder: ${data.output_dir}`, 'System', 'info');

      // Connect SSE
      connectEventStream(currentRunId);

    } catch (e) {
      alert('Error starting production: ' + e.message);
      setSystemStatus('FAILED');
      appendLog(e.message, 'System', 'error');
    }
  }

  async function handleApproveScreenplay() {
    if (!currentRunId) return;

    btnApproveScreenplay.disabled = true;
    const approvalText = hitlFeedbackInput.value.trim() || 'Approved. Proceed with video production and assembly.';

    // Collect any inline edits to narration
    const editedScript = collectScreenplayEdits();

    try {
      const res = await fetch(`/api/runs/${currentRunId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          approval_text: approvalText,
          screenplay: editedScript
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Approval failed.');
      }

      hitlPanel.style.display = 'none';
      setSystemStatus('PRODUCING_VIDEO');
      appendLog(`Screenplay approved by human director. Rendering begins.`, 'User', 'success');

    } catch (e) {
      alert('Approval error: ' + e.message);
      btnApproveScreenplay.disabled = false;
    }
  }

  function collectScreenplayEdits() {
    if (!activeScreenplay || !activeScreenplay.length) return null;

    const cloned = JSON.parse(JSON.stringify(activeScreenplay));
    cloned.forEach((part, pIdx) => {
      (part.subscenes || []).forEach((sub, sIdx) => {
        const input = document.getElementById(`narr-input-${pIdx}-${sIdx}`);
        if (input) {
          sub.narration = input.value.trim();
        }
      });
    });
    return cloned;
  }

  async function handleCancelRun() {
    if (!currentRunId) return;
    if (!confirm('Are you sure you want to halt this production run?')) return;

    try {
      await fetch(`/api/runs/${currentRunId}/cancel`, { method: 'POST' });
      hitlPanel.style.display = 'none';
      setSystemStatus('CANCELLED');
    } catch (e) {
      console.error(e);
    }
  }

  // -------------------------------------------------------------------------
  // Server-Sent Events (SSE) Stream
  // -------------------------------------------------------------------------
  function connectEventStream(runId) {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(`/api/runs/${runId}/stream`);

    eventSource.addEventListener('init', (e) => {
      const data = JSON.parse(e.data);
      currentRunState = data;
      if (data.status) setSystemStatus(data.status);
      if (data.cost_metrics) updateBudgetDisplay(data.cost_metrics);
      if (data.cost_summary) updateBudgetDisplay(data.cost_summary);
      if (data.parts_status) renderAllPartsStatus(data.parts_status, data.requested_parts);
    });

    eventSource.addEventListener('log', (e) => {
      const entry = JSON.parse(e.data);
      if (entry.cost_metrics) updateBudgetDisplay(entry.cost_metrics);
      appendLog(entry.text, entry.author, entry.level, entry.time_str);
    });

    eventSource.addEventListener('hitl_gate', (e) => {
      const data = JSON.parse(e.data);
      if (data.cost_metrics) updateBudgetDisplay(data.cost_metrics);
      showHitlReviewGate(data);
    });

    eventSource.addEventListener('new_asset', (e) => {
      const asset = JSON.parse(e.data);
      if (asset.cost_metrics) updateBudgetDisplay(asset.cost_metrics);
      handleNewAsset(asset);
    });

    eventSource.addEventListener('live_scene_update', (e) => {
      const update = JSON.parse(e.data);
      if (update.cost_metrics) updateBudgetDisplay(update.cost_metrics);
      updateLiveSpotlight(update);
    });

    // Requirement: Show folder structure / state / intermediate output after every part completes
    eventSource.addEventListener('part_completed', (e) => {
      const data = JSON.parse(e.data);
      if (data.cost_metrics) updateBudgetDisplay(data.cost_metrics);
      handlePartCompleted(data.part_number, data.total_parts, data.part_summary);
    });

    eventSource.addEventListener('master_movie_ready', (e) => {
      const asset = JSON.parse(e.data);
      if (asset.cost_metrics) updateBudgetDisplay(asset.cost_metrics);
      showMasterMovieShowcase(asset);
    });

    eventSource.addEventListener('status_change', (e) => {
      const data = JSON.parse(e.data);
      setSystemStatus(data.status);
      if (data.cost_metrics) updateBudgetDisplay(data.cost_metrics);
      if (data.cost_summary) updateBudgetDisplay(data.cost_summary);
    });

    eventSource.onerror = (e) => {
      console.warn('SSE stream notice:', e);
    };
  }

  // -------------------------------------------------------------------------
  // HITL Review Gate UI
  // -------------------------------------------------------------------------
  function showHitlReviewGate(data) {
    setSystemStatus('AWAITING_APPROVAL');
    hitlPanel.style.display = 'block';
    btnApproveScreenplay.disabled = false;

    if (data.prompt_message) {
      hitlPromptText.textContent = data.prompt_message;
    }

    screenplayContainer.innerHTML = '';
    activeScreenplay = data.screenplay || [];

    if (!activeScreenplay.length) {
      screenplayContainer.innerHTML = '<div style="color: var(--text-dim); padding: 1rem;">No screenplay parts found in state.</div>';
      return;
    }

    activeScreenplay.forEach((part, pIdx) => {
      const partCard = document.createElement('div');
      partCard.className = 'part-card';

      const queries = (part.search_queries || []).map(q => 
        `<span class="badge badge-idle" style="font-size: 0.7rem; margin-right: 0.3rem;">👤 ${escapeHtml(q)}</span>`
      ).join('');

      let subscenesHtml = '';
      (part.subscenes || []).forEach((sub, sIdx) => {
        const veoBadge = sub.is_veo_candidate ? '<span class="veo-tag">✨ Veo 8s Motion Clip</span>' : '';
        subscenesHtml += `
          <div class="subscene-row">
            <span class="subscene-badge">${part.part_number}.${sub.subscene_number}</span>
            <div style="flex: 1;">
              <input type="text" id="narr-input-${pIdx}-${sIdx}" class="editable-narration" value="${escapeHtml(sub.narration)}" title="Click to tweak narration before approving" />
              <div class="subscene-visual">Visual: ${escapeHtml(sub.visual_description)}</div>
            </div>
            <div>${veoBadge}</div>
          </div>
        `;
      });

      partCard.innerHTML = `
        <div class="part-card-title">Part ${part.part_number}: ${escapeHtml(part.title || '')}</div>
        <div class="part-card-synopsis">${escapeHtml(part.synopsis || '')}</div>
        ${queries ? `<div style="margin-bottom: 0.6rem;">${queries}</div>` : ''}
        <div class="subscenes-list">${subscenesHtml}</div>
      `;

      screenplayContainer.appendChild(partCard);
    });

    // Smooth scroll to HITL Panel
    hitlPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // -------------------------------------------------------------------------
  // Episodic Parts & Intermediate Outputs UI
  // -------------------------------------------------------------------------
  function renderAllPartsStatus(partsStatusDict, totalParts) {
    if (!partsStatusDict || !Object.keys(partsStatusDict).length) return;
    partsSection.style.display = 'block';

    partsCardsContainer.innerHTML = '';
    const partNums = Object.keys(partsStatusDict).map(Number).sort((a, b) => a - b);
    partsOverallBadge.textContent = `${partNums.length} / ${totalParts || 3} Parts Completed`;
    partsOverallBadge.className = 'badge badge-completed';

    partNums.forEach(pNum => {
      const summary = partsStatusDict[pNum];
      renderPartCard(pNum, totalParts, summary);
    });
  }

  function handlePartCompleted(partNum, totalParts, partSummary) {
    partsSection.style.display = 'block';
    partsOverallBadge.textContent = `${partNum} / ${totalParts || 3} Parts Completed`;
    partsOverallBadge.className = 'badge badge-completed';

    renderPartCard(partNum, totalParts, partSummary);
    appendLog(`Part #${partNum} Intermediate Outputs verified on disk: ${partSummary.summary}`, 'Director', 'success');
  }

  function renderPartCard(partNum, totalParts, partSummary) {
    let card = document.getElementById(`part-card-${partNum}`);
    if (!card) {
      card = document.createElement('div');
      card.id = `part-card-${partNum}`;
      card.className = 'part-progress-card part-completed';
      partsCardsContainer.appendChild(card);
    }

    let chipsHtml = '';

    // 1. Comic Grid Chip
    if (partSummary.grid_image && partSummary.grid_image.url) {
      chipsHtml += `
        <div class="intermediate-output-chip" onclick="window.openLightboxImage('${partSummary.grid_image.url}')">
          <img src="${partSummary.grid_image.url}" class="intermediate-thumb-preview" alt="Grid">
          <span>🖼️ 3x2 Comic Grid</span>
        </div>
      `;
    }

    // 2. Panels Chip
    if (partSummary.panels_count > 0) {
      chipsHtml += `
        <div class="intermediate-output-chip" onclick="window.switchTab('grids_panels')">
          <span>🧩 ${partSummary.panels_count}/6 Panels Sliced</span>
        </div>
      `;
    }

    // 3. Subscenes Chip
    if (partSummary.subscenes_count > 0) {
      chipsHtml += `
        <div class="intermediate-output-chip" onclick="window.switchTab('subscenes')">
          <span>🎥 ${partSummary.subscenes_count}/6 Subscenes Done</span>
        </div>
      `;
    }

    // 4. Part Master Video Chip
    if (partSummary.part_master && partSummary.part_master.url) {
      chipsHtml += `
        <div class="intermediate-output-chip" onclick="window.openLightboxVideo('${partSummary.part_master.url}')">
          <span>🎞️ Part Master (${partSummary.part_master.duration.toFixed(1)}s) ▶</span>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="part-progress-header">
        <div class="part-progress-title">
          <span>Part ${partNum}: ${escapeHtml(partSummary.title || '')}</span>
        </div>
        <span class="badge badge-completed">✅ Part ${partNum} Complete 100%</span>
      </div>
      <div style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(partSummary.summary || '')}</div>
      <div class="intermediate-outputs-row">
        <span style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase;">Intermediate Outputs:</span>
        ${chipsHtml}
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // Live Working Scene Spotlight (Handles both LIVE IMAGES & LIVE VIDEOS)
  // -------------------------------------------------------------------------
  function updateLiveSpotlight(update) {
    const asset = update.asset;
    if (!asset || !asset.url) return;

    liveOverlayPill.style.display = 'flex';
    const isImage = (update.media_type === 'image') ||
      ['grid', 'panel', 'reference'].includes(asset.category) ||
      /\.(png|jpg|jpeg|webp)$/i.test(asset.filename || '');

    if (isImage) {
      // Switch player to image mode
      liveVideoPlayer.pause();
      liveVideoPlayer.style.display = 'none';
      liveImagePlayer.style.display = 'block';
      liveImagePlayer.src = asset.url;

      if (liveOverlayDot) liveOverlayDot.style.background = 'var(--accent-cyan)';
      liveOverlayText.textContent = asset.label || 'Live Image';

      liveSceneTag.className = 'badge badge-planning';
      liveSceneTag.textContent = asset.label || 'Working Image';

      liveSceneCaption.innerHTML = `
        <strong>${escapeHtml(asset.label)}</strong> — High-res visual asset created on disk (${((asset.size_bytes || 0) / 1024).toFixed(1)} KB).
      `;
    } else {
      // Switch player to video mode
      liveImagePlayer.style.display = 'none';
      liveVideoPlayer.style.display = 'block';

      if (liveVideoSource.src !== asset.url) {
        liveVideoSource.src = asset.url;
        liveVideoPlayer.load();
        liveVideoPlayer.play().catch(() => {});
      }

      if (liveOverlayDot) liveOverlayDot.style.background = 'var(--accent-rose)';
      liveOverlayText.textContent = asset.label || 'Live Video';

      liveSceneTag.className = 'badge badge-producing';
      liveSceneTag.textContent = asset.label || 'Working Scene';

      liveSceneCaption.innerHTML = `
        <strong>${escapeHtml(asset.label)}</strong> — Native 720p H.264 video rendered &amp; verified on disk.
      `;
    }
  }

  function showMasterMovieShowcase(asset) {
    masterEmpty.style.display = 'none';
    masterShowcase.style.display = 'block';

    finalMasterSource.src = asset.url;
    finalMasterPlayer.load();
    btnDownloadMaster.href = asset.url;
    btnDownloadMaster.setAttribute('download', asset.filename || 'movie_master.mp4');

    masterMeta.textContent = `File: ${asset.filename} | Native 1280x720 H.264`;

    // Auto-switch to Final Movie Tab
    setActiveTab('final_movie');
  }

  // -------------------------------------------------------------------------
  // Assets & Gallery Manager
  // -------------------------------------------------------------------------
  let assetCounters = {
    reference: 0,
    grid: 0,
    panel: 0,
    subscene: 0,
    part_master: 0,
    master_movie: 0,
    total: 0
  };

  function handleNewAsset(asset) {
    assetCounters.total++;
    metaAssetCount.textContent = `${assetCounters.total} Assets`;

    const cat = asset.category || 'other';
    if (assetCounters[cat] !== undefined) assetCounters[cat]++;

    // Update badges
    badgeMaster.textContent = assetCounters.master_movie;
    badgeParts.textContent = assetCounters.part_master;
    badgeSubscenes.textContent = assetCounters.subscene;
    badgePanels.textContent = assetCounters.panel + assetCounters.grid;
    badgeRefs.textContent = assetCounters.reference;
    badgeDump.textContent = assetCounters.total;

    // Render into target gallery
    if (cat === 'master_movie') {
      showMasterMovieShowcase(asset);
    } else if (cat === 'part_master') {
      renderPartMasterCard(asset);
    } else if (cat === 'subscene') {
      renderSubsceneCard(asset);
    } else if (cat === 'grid' || cat === 'panel') {
      renderGridPanelCard(asset);
    } else if (cat === 'reference') {
      renderReferenceCard(asset);
    }

    // Always add to output dump list
    renderDumpEntry(asset);
  }

  function renderPartMasterCard(asset) {
    if (partMastersGrid.querySelector('.empty-notice') || assetCounters.part_master === 1) {
      partMastersGrid.innerHTML = '';
    }

    const card = document.createElement('div');
    card.className = 'asset-card';
    card.innerHTML = `
      <video class="asset-thumb" src="${asset.url}" controls playsinline preload="metadata"></video>
      <div class="asset-meta">
        <div class="asset-title">${escapeHtml(asset.label || asset.filename)}</div>
        <div class="asset-sub">Part Master &bull; 720p</div>
      </div>
    `;
    partMastersGrid.prepend(card);
  }

  function renderSubsceneCard(asset) {
    if (subscenesGrid.querySelector('.empty-notice') || assetCounters.subscene === 1) {
      subscenesGrid.innerHTML = '';
    }

    const card = document.createElement('div');
    card.className = 'asset-card';
    card.innerHTML = `
      <video class="asset-thumb" src="${asset.url}" controls playsinline preload="metadata"></video>
      <div class="asset-meta">
        <div class="asset-title">${escapeHtml(asset.label || asset.filename)}</div>
        <div class="asset-sub">Subtitled 720p Scene</div>
      </div>
    `;
    subscenesGrid.prepend(card);
  }

  function renderGridPanelCard(asset) {
    if (gridsPanelsGrid.querySelector('.empty-notice') || (assetCounters.panel + assetCounters.grid) === 1) {
      gridsPanelsGrid.innerHTML = '';
    }

    const card = document.createElement('div');
    card.className = 'asset-card';
    card.innerHTML = `
      <img class="asset-thumb" src="${asset.url}" alt="${escapeHtml(asset.label)}" loading="lazy">
      <div class="asset-meta">
        <div class="asset-title">${escapeHtml(asset.label)}</div>
        <div class="asset-sub">${asset.filename}</div>
      </div>
    `;

    card.querySelector('img').addEventListener('click', () => {
      openLightbox(`<img src="${asset.url}" style="max-width: 100%; border-radius: 8px;">`);
    });

    gridsPanelsGrid.prepend(card);
  }

  function renderReferenceCard(asset) {
    if (referencesGrid.querySelector('.empty-notice') || assetCounters.reference === 1) {
      referencesGrid.innerHTML = '';
    }

    const card = document.createElement('div');
    card.className = 'asset-card';
    card.innerHTML = `
      <img class="asset-thumb" src="${asset.url}" alt="${escapeHtml(asset.label)}" loading="lazy">
      <div class="asset-meta">
        <div class="asset-title">${escapeHtml(asset.label)}</div>
        <div class="asset-sub">Character Portrait</div>
      </div>
    `;

    card.querySelector('img').addEventListener('click', () => {
      openLightbox(`<img src="${asset.url}" style="max-width: 100%; max-height: 80vh; border-radius: 8px;">`);
    });

    referencesGrid.prepend(card);
  }

  function renderDumpEntry(asset) {
    if (folderDumpList.querySelector('.empty-notice') || assetCounters.total === 1) {
      folderDumpList.innerHTML = '';
    }

    const item = document.createElement('div');
    item.style.display = 'flex';
    item.style.justifyContent = 'space-between';
    item.style.padding = '0.35rem 0.6rem';
    item.style.background = 'var(--bg-secondary)';
    item.style.borderRadius = '4px';

    const sizeKb = ((asset.size_bytes || 0) / 1024).toFixed(1);
    item.innerHTML = `
      <div>
        <a href="${asset.url}" target="_blank" style="color: var(--accent-cyan); text-decoration: none;">📄 ${escapeHtml(asset.filename)}</a>
        <span style="color: var(--text-dim); margin-left: 0.5rem;">[${escapeHtml(asset.category)}]</span>
      </div>
      <div style="color: var(--text-dim);">${sizeKb} KB</div>
    `;
    folderDumpList.prepend(item);
  }

  function openLightbox(html) {
    lightboxBody.innerHTML = html;
    lightboxModal.style.display = 'flex';
  }

  // Global helpers attached to window for inline onclick handlers
  window.openLightboxImage = function (url) {
    openLightbox(`<img src="${url}" style="max-width: 100%; max-height: 80vh; border-radius: 8px;">`);
  };

  window.openLightboxVideo = function (url) {
    openLightbox(`<video src="${url}" controls autoplay playsinline style="max-width: 100%; max-height: 80vh; border-radius: 8px;"></video>`);
  };

  window.switchTab = function (tabName) {
    setActiveTab(tabName);
  };

  // -------------------------------------------------------------------------
  // Past Runs / History Modal
  // -------------------------------------------------------------------------
  async function openHistoryModal() {
    historyModal.style.display = 'flex';
    historyFoldersList.innerHTML = '<div style="color: var(--text-dim); padding: 1rem;">Loading past runs...</div>';

    try {
      const res = await fetch('/api/history');
      if (!res.ok) throw new Error('Failed to load past runs.');
      const data = await res.json();

      if (!data || !data.length) {
        historyFoldersList.innerHTML = '<div style="color: var(--text-dim); padding: 1rem;">No past runs found in output/ folder.</div>';
        return;
      }

      historyFoldersList.innerHTML = '';
      data.forEach(item => {
        const card = document.createElement('div');
        card.style.background = 'var(--bg-tertiary)';
        card.style.border = '1px solid var(--border-color)';
        card.style.borderRadius = '8px';
        card.style.padding = '0.75rem 1rem';
        card.style.display = 'flex';
        card.style.justifyContent = 'space-between';
        card.style.alignItems = 'center';

        const finalBadge = item.has_final_movie ? '<span class="badge badge-completed">🎬 Movie Ready</span>' : '';
        const playBtn = item.final_movie_url ? 
          `<a href="${item.final_movie_url}" target="_blank" class="btn-secondary" style="font-size: 0.75rem;">▶ Watch</a>` : '';

        card.innerHTML = `
          <div>
            <div style="font-weight: 600; color: var(--text-main); font-family: var(--font-mono); font-size: 0.82rem;">${escapeHtml(item.folder_name)}</div>
            <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.2rem;">
              ${item.total_files} files &bull; ${item.parts_count} Part Masters &bull; ${item.grid_count} Grids
            </div>
          </div>
          <div style="display: flex; gap: 0.5rem; align-items: center;">
            ${finalBadge}
            ${playBtn}
          </div>
        `;

        historyFoldersList.appendChild(card);
      });

    } catch (e) {
      historyFoldersList.innerHTML = `<div style="color: var(--accent-rose); padding: 1rem;">${escapeHtml(e.message)}</div>`;
    }
  }

  // -------------------------------------------------------------------------
  // Terminal Logs & Budget Helpers
  // -------------------------------------------------------------------------
  function appendLog(text, author = 'Director', level = 'info', timeStr = null) {
    const time = timeStr || (new Date()).toLocaleTimeString();
    const div = document.createElement('div');
    div.className = `log-line log-level-${level}`;
    div.innerHTML = `
      <span class="log-time">[${time}]</span>
      <span class="log-author">[${escapeHtml(author)}]</span>
      <span class="log-text">${escapeHtml(text)}</span>
    `;
    terminalLogs.appendChild(div);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
  }

  function updateBudgetDisplay(data) {
    if (!data) return;
    if (typeof data === 'object') {
      if (data.total_tokens !== undefined) {
        budgetTokens.textContent = `${data.total_tokens.toLocaleString()} tokens`;
      }
      if (data.images_generated !== undefined) {
        budgetImages.textContent = `${data.images_generated} calls ($${(data.images_generated * 0.035).toFixed(2)})`;
      }
      if (data.videos_generated !== undefined) {
        budgetVeo.textContent = `${data.videos_generated} clips`;
      }
      if (data.estimated_cost_usd !== undefined) {
        budgetTotal.textContent = `$${parseFloat(data.estimated_cost_usd).toFixed(3)}`;
      }
      return;
    }
    const matchTokens = data.match(/Total Tokens:\s*([0-9,]+)/i);
    if (matchTokens) budgetTokens.textContent = `${matchTokens[1]} tokens`;

    const matchImages = data.match(/Images Generated:\s*(\d+)/i);
    if (matchImages) budgetImages.textContent = `${matchImages[1]} calls`;

    const matchVideos = data.match(/Videos Generated:\s*(\d+)/i);
    if (matchVideos) budgetVeo.textContent = `${matchVideos[1]} clips`;

    const matchTotal = data.match(/(?:ESTIMATED TOTAL COST|Total Estimated Cost):\s*\$([0-9.]+)/i);
    if (matchTotal) {
      budgetTotal.textContent = `$${parseFloat(matchTotal[1]).toFixed(3)}`;
    }
  }

  function resetRunUI() {
    assetCounters = { reference: 0, grid: 0, panel: 0, subscene: 0, part_master: 0, master_movie: 0, total: 0 };
    badgeMaster.textContent = '0';
    badgeParts.textContent = '0';
    badgeSubscenes.textContent = '0';
    badgePanels.textContent = '0';
    badgeRefs.textContent = '0';
    badgeDump.textContent = '0';

    partMastersGrid.innerHTML = '<div class="empty-notice" style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-dim);">No Part masters assembled yet.</div>';
    subscenesGrid.innerHTML = '<div class="empty-notice" style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-dim);">No working scenes rendered yet.</div>';
    gridsPanelsGrid.innerHTML = '<div class="empty-notice" style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-dim);">No 3x2 comic grids or panel slices generated yet.</div>';
    referencesGrid.innerHTML = '<div class="empty-notice" style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-dim);">No historical portrait references acquired yet.</div>';
    folderDumpList.innerHTML = '<div class="empty-notice" style="color: var(--text-dim); padding: 1rem;">No files in output directory yet.</div>';

    partsCardsContainer.innerHTML = '';
    partsSection.style.display = 'none';

    masterShowcase.style.display = 'none';
    masterEmpty.style.display = 'block';
    hitlPanel.style.display = 'none';

    liveImagePlayer.style.display = 'none';
    liveVideoPlayer.style.display = 'block';
    liveVideoSource.src = '';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Run on load
  document.addEventListener('DOMContentLoaded', init);
})();
