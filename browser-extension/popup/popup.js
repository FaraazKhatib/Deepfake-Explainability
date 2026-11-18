/* ─────────────────────────────────────────────────────────────────
   VeriTrust AI — Chrome Extension Popup Logic
   Calls POST /api/v1/text/check-news
   - Scans and analyses run in background service worker (background.js)
     so operations continue even if the popup closes due to website
     notification dialogs, subscription modals, or lost focus.
   - Real-time synchronization via chrome.storage.onChanged
   - Supports 1-click docking into persistent Chrome Side Panel
   ───────────────────────────────────────────────────────────────── */

// ── DOM helpers ──────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DEFAULT_API_URL = 'http://127.0.0.1:8001';

// ── Storage helpers (safe fallback to localStorage) ──────────────
async function getStoredApiUrl() {
  try {
    if (typeof chrome !== 'undefined' && chrome?.storage?.local) {
      const stored = await chrome.storage.local.get('apiUrl');
      if (stored && stored.apiUrl) return stored.apiUrl;
    }
  } catch (err) {
    console.warn('[VeriTrust] chrome.storage get error:', err);
  }
  try {
    return localStorage.getItem('veritrust_api_url') || DEFAULT_API_URL;
  } catch {
    return DEFAULT_API_URL;
  }
}

async function loadState() {
  try {
    if (typeof chrome !== 'undefined' && chrome?.storage?.local) {
      const data = await chrome.storage.local.get('vt_state');
      if (data && data.vt_state) return data.vt_state;
    }
  } catch (err) {
    console.warn('[VeriTrust] chrome.storage get error:', err);
  }
  try {
    const raw = localStorage.getItem('vt_saved_state');
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

async function persistState(patch) {
  try {
    const current = (await loadState()) || {};
    const updated = { ...current, ...patch, updatedAt: Date.now() };

    if (typeof chrome !== 'undefined' && chrome?.storage?.local) {
      await chrome.storage.local.set({ vt_state: updated });
    }
    try {
      localStorage.setItem('vt_saved_state', JSON.stringify(updated));
    } catch {}
  } catch (err) {
    console.warn('[VeriTrust] Error persisting state:', err);
  }
}

async function clearPersistedState() {
  try {
    if (typeof chrome !== 'undefined' && chrome?.storage?.local) {
      await chrome.storage.local.remove('vt_state');
    }
    try {
      localStorage.removeItem('vt_saved_state');
    } catch {}
  } catch (err) {
    console.warn('[VeriTrust] Error clearing state:', err);
  }
}

// ── Main Initialization ──────────────────────────────────────────
async function initPopup() {
  console.log('[VeriTrust] Initializing popup...');

  const inputText             = $('#inputText');
  const analyzeBtn            = $('#analyzeBtn');
  const translateBtn          = $('#translateBtn');
  const scanBtn               = $('#scanBtn');
  const extractedPreview      = $('#extractedPreview');
  const resultsSection        = $('#results');
  const translationBanner     = $('#translationBanner');
  const translationText       = $('#translationText');
  const viewOriginalBtn       = $('#viewOriginalBtn');
  const originalTextContainer = $('#originalTextContainer');
  const verdictBanner         = $('#verdictBanner');
  const summaryText           = $('#summaryText');
  const explanationText       = $('#explanationText');
  const claimsList            = $('#claimsList');
  const analysisNotes         = $('#analysisNotes');
  const errorBanner           = $('#errorBanner');
  const statusBadge           = $('#statusBadge');
  const resetBtn              = $('#resetBtn');
  const sidePanelBtn          = $('#sidePanelBtn');

  // Multi-tab context elements
  const scannedSourceBanner   = $('#scannedSourceBanner');
  const sourceTabIndicator    = $('#sourceTabIndicator');
  const sourcePageLink        = $('#sourcePageLink');
  const goToScannedTabBtn     = $('#goToScannedTabBtn');
  const scanCurrentTabBtn     = $('#scanCurrentTabBtn');
  const bgScanProgressBox     = $('#bgScanProgressBox');
  const bgScanStatusText      = $('#bgScanStatusText');
  const bgScanTargetText      = $('#bgScanTargetText');
  const bgScanGoToTabBtn      = $('#bgScanGoToTabBtn');
  const bgScanCancelBtn       = $('#bgScanCancelBtn');

  // Verification mode single toggle
  const strictModeToggle      = $('#strictModeToggle');
  const modeName              = $('#modeName');
  const modeDesc              = $('#modeDesc');

  function updateModeUI(isStrict) {
    if (strictModeToggle) strictModeToggle.checked = isStrict;
    if (modeName) {
      modeName.textContent = isStrict ? 'Strict Tier-1 Audit' : 'Standard (Fast Live News)';
      modeName.classList.toggle('strict', isStrict);
    }
    if (modeDesc) {
      modeDesc.textContent = isStrict
        ? 'Enforces Tier-1 wire consensus (BBC, Reuters, AP) & audits secondary claims.'
        : 'Verifies primary story against live indexed news in ~8-12s.';
    }
  }

  // Load and persist toggle preferences
  try {
    if (typeof chrome !== 'undefined' && chrome.storage?.local) {
      chrome.storage.local.get(['vt_strict_mode']).then((prefs) => {
        const isStrict = Boolean(prefs?.vt_strict_mode);
        updateModeUI(isStrict);
      }).catch(() => {});
    }
  } catch {}

  strictModeToggle?.addEventListener('change', () => {
    const isStrict = strictModeToggle.checked;
    updateModeUI(isStrict);
    chrome.storage?.local?.set({ vt_strict_mode: isStrict }).catch(() => {});
  });

  // Side Panel open handler
  if (sidePanelBtn) {
    if (window.innerWidth > 430) {
      sidePanelBtn.style.display = 'none'; // already docked inside side panel
    } else {
      sidePanelBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        try {
          if (typeof chrome !== 'undefined' && chrome.sidePanel?.open) {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab && tab.windowId) {
              await chrome.sidePanel.open({ windowId: tab.windowId });
              window.close();
            }
          }
        } catch (err) {
          console.warn('[VeriTrust] Error opening side panel:', err);
        }
      });
    }
  }

  // Tab switching
  function switchTab(tabTarget) {
    $$('.tab').forEach((t) => t.classList.remove('active'));
    $$('.tab-content').forEach((c) => c.classList.remove('active'));

    const tabBtn = $(`button[data-tab="${tabTarget}"]`);
    if (tabBtn) tabBtn.classList.add('active');

    const targetContent = $(`#tab-${tabTarget}`);
    if (targetContent) targetContent.classList.add('active');
  }

  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const tabTarget = tab.dataset.tab;
      switchTab(tabTarget);
      persistState({ activeTab: tabTarget });
    });
  });

  let pollInterval = null;
  function startStatusPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
      try {
        const current = await loadState();
        if (current && current.vt_status === 'done' && current.resultData) {
          stopStatusPolling();
          applyState(current);
        } else if (current && current.vt_status === 'error') {
          stopStatusPolling();
          applyState(current);
        } else if (current?.scanStartTime && (Date.now() - current.scanStartTime > 75000)) {
          stopStatusPolling();
          applyState({ ...current, vt_status: 'ready', error: null });
        }
      } catch {}
    }, 1000);
  }

  function stopStatusPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  // State application helper
  function applyState(saved) {
    if (!saved) return;

    if (saved.activeTab === 'scan') {
      switchTab('scan');
    } else if (saved.activeTab === 'paste') {
      switchTab('paste');
    }

    if (saved.inputText && inputText && !inputText.value) {
      inputText.value = saved.inputText;
    }

    if (saved.extractedText && extractedPreview) {
      extractedPreview.textContent = saved.extractedText.length > 500
        ? saved.extractedText.slice(0, 500) + '…'
        : saved.extractedText;
      extractedPreview.classList.remove('hidden');
    }

    if (saved.vt_status === 'scanning') {
      const elapsed = Date.now() - (saved.scanStartTime || 0);
      if (elapsed > 75000) {
        // If an old scan was left dangling, reset smoothly to ready state
        stopStatusPolling();
        saved.vt_status = 'ready';
        saved.error = null;
        persistState(saved);
        setLoading(scanBtn, false);
        setLoading(analyzeBtn, false);
        if (statusBadge) statusBadge.textContent = 'Ready';
        return;
      }

      const activeBtn = saved.activeTab === 'paste' ? analyzeBtn : scanBtn;
      setLoading(activeBtn, true);
      hideResults();
      hideError();
      if (statusBadge) statusBadge.textContent = 'Verifying story… (~8-12s)';
      if (resetBtn) resetBtn.style.display = 'inline-block';
      startStatusPolling();
    } else {
      stopStatusPolling();
      if (saved.vt_status === 'done' && saved.resultData) {
        setLoading(scanBtn, false);
        setLoading(analyzeBtn, false);
        renderResults(saved.resultData);
        if (statusBadge) statusBadge.textContent = 'Done';
        if (resetBtn) resetBtn.style.display = 'inline-block';
      } else if (saved.vt_status === 'error' && saved.error) {
        setLoading(scanBtn, false);
        setLoading(analyzeBtn, false);
        showError(saved.error);
        if (statusBadge) statusBadge.textContent = 'Error';
        if (resetBtn) resetBtn.style.display = 'inline-block';
      } else if (saved.resultData) {
        renderResults(saved.resultData);
        if (statusBadge) statusBadge.textContent = 'Done';
        if (resetBtn) resetBtn.style.display = 'inline-block';
      }
    }

    // Update tab-awareness banner
    updateTabContextBanner(saved);
  }

  // ── Multi-Tab Context Banner Helper ────────────────────────────
  async function updateTabContextBanner(saved) {
    if (!saved || (!saved.pageUrl && !saved.tabTitle)) {
      if (scannedSourceBanner) scannedSourceBanner.classList.add('hidden');
      if (bgScanProgressBox) bgScanProgressBox.classList.add('hidden');
      return;
    }

    let currentTab = null;
    try {
      if (typeof chrome !== 'undefined' && chrome.tabs?.query) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        currentTab = tab;
      }
    } catch {}

    const isSameTab = currentTab && (
      (saved.tabId && currentTab.id === saved.tabId) ||
      (saved.pageUrl && currentTab.url === saved.pageUrl)
    );

    let domain = '';
    try {
      if (saved.pageUrl) domain = new URL(saved.pageUrl).hostname.replace('www.', '');
    } catch {}

    const pageLabel = saved.tabTitle || domain || 'Source Article';

    // 1. If currently scanning in background:
    if (saved.vt_status === 'scanning') {
      if (bgScanProgressBox) {
        bgScanProgressBox.classList.remove('hidden');
        if (bgScanTargetText) {
          bgScanTargetText.textContent = isSameTab
            ? `Scanning current tab: ${pageLabel}`
            : `Scanning in background: ${pageLabel}`;
        }
        if (bgScanGoToTabBtn) {
          bgScanGoToTabBtn.style.display = (isSameTab || !saved.tabId) ? 'none' : 'inline-flex';
        }
      }
      if (scannedSourceBanner) scannedSourceBanner.classList.add('hidden');
      return;
    } else {
      if (bgScanProgressBox) bgScanProgressBox.classList.add('hidden');
    }

    // 2. If scan is done and results exist:
    if (saved.vt_status === 'done' && saved.resultData) {
      if (scannedSourceBanner) {
        scannedSourceBanner.classList.remove('hidden');
        if (sourcePageLink) {
          sourcePageLink.href = saved.pageUrl || '#';
          sourcePageLink.textContent = pageLabel;
        }
        if (sourceTabIndicator) {
          sourceTabIndicator.textContent = isSameTab ? '● Current Tab:' : '● Scanned from other tab:';
          sourceTabIndicator.style.color = isSameTab ? '#10b981' : '#f59e0b';
        }
        if (goToScannedTabBtn) {
          goToScannedTabBtn.classList.toggle('hidden', isSameTab || !saved.tabId);
        }
        if (scanCurrentTabBtn) {
          const canScanCurrent = currentTab && !isSameTab && currentTab.url && !currentTab.url.startsWith('chrome://');
          scanCurrentTabBtn.classList.toggle('hidden', !canScanCurrent);
          if (canScanCurrent) {
            let curDomain = 'Current Tab';
            try { curDomain = new URL(currentTab.url).hostname.replace('www.', ''); } catch {}
            scanCurrentTabBtn.textContent = `Scan ${curDomain}`;
          }
        }
      }
    }
  }

  // Restore saved state on initial load
  try {
    const saved = await loadState();
    if (saved) {
      applyState(saved);
    }
  } catch (e) {
    console.warn('[VeriTrust] Failed to restore saved state:', e);
  }

  // Listen to background service worker state changes in real-time
  if (typeof chrome !== 'undefined' && chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'local' && changes.vt_state?.newValue) {
        applyState(changes.vt_state.newValue);
      }
    });
  }

  // Save text changes on input
  let inputTimer = null;
  if (inputText) {
    inputText.addEventListener('input', () => {
      clearTimeout(inputTimer);
      inputTimer = setTimeout(() => {
        const text = inputText.value;
        persistState({ inputText: text });
        if (resetBtn) resetBtn.style.display = text.trim() ? 'inline-block' : 'none';
      }, 250);
    });
  }

  // Reset / New Check
  if (resetBtn) {
    resetBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      stopStatusPolling();
      await clearPersistedState();

      try {
        if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
          chrome.runtime.sendMessage({ action: 'CLEAR_BADGE' }).catch(() => {});
        }
      } catch {}

      if (inputText) inputText.value = '';
      if (extractedPreview) {
        extractedPreview.textContent = '';
        extractedPreview.classList.add('hidden');
      }

      setLoading(scanBtn, false);
      setLoading(analyzeBtn, false);
      hideResults();
      hideError();
      if (scannedSourceBanner) scannedSourceBanner.classList.add('hidden');
      if (bgScanProgressBox) bgScanProgressBox.classList.add('hidden');

      if (statusBadge) statusBadge.textContent = 'Ready';
      resetBtn.style.display = 'none';
      switchTab('paste');
    });
  }

  // Multi-tab navigation & action handlers
  if (goToScannedTabBtn) {
    goToScannedTabBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      try {
        const saved = await loadState();
        if (saved?.tabId && chrome.tabs) {
          await chrome.tabs.update(saved.tabId, { active: true });
        }
      } catch (err) {
        console.warn('[VeriTrust] Switch tab error:', err);
      }
    });
  }

  if (bgScanGoToTabBtn) {
    bgScanGoToTabBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      try {
        const saved = await loadState();
        if (saved?.tabId && chrome.tabs) {
          await chrome.tabs.update(saved.tabId, { active: true });
        }
      } catch (err) {
        console.warn('[VeriTrust] Switch tab error:', err);
      }
    });
  }

  if (bgScanCancelBtn) {
    bgScanCancelBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      try {
        if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
          await chrome.runtime.sendMessage({ action: 'CANCEL_ANALYSIS' }).catch(() => {});
        }
        await clearPersistedState();
        hideError();
        hideResults();
        if (bgScanProgressBox) bgScanProgressBox.classList.add('hidden');
        if (scannedSourceBanner) scannedSourceBanner.classList.add('hidden');
        setLoading(scanBtn, false);
        setLoading(analyzeBtn, false);
        if (statusBadge) statusBadge.textContent = 'Ready';
        switchTab('scan');
        await scanCurrentPage();
      } catch (err) {
        console.warn('[VeriTrust] Cancel & rescan error:', err);
      }
    });
  }

  if (scanCurrentTabBtn) {
    scanCurrentTabBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      try {
        await clearPersistedState();
        hideResults();
        hideError();
        if (scannedSourceBanner) scannedSourceBanner.classList.add('hidden');
        switchTab('scan');
        await scanCurrentPage();
      } catch (err) {
        console.warn('[VeriTrust] Scan current tab error:', err);
      }
    });
  }

  // Analyze (paste) button
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      const text = inputText ? inputText.value.trim() : '';
      if (!text) {
        showError('Please paste some text, news claim, or article to analyze.');
        if (inputText) inputText.focus();
        return;
      }
      await runAnalysis(text, analyzeBtn, '', '', 'paste');
    });
  }

  // Translate (paste) button
  if (translateBtn) {
    translateBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      const text = inputText ? inputText.value.trim() : '';
      if (!text) {
        showError('Please paste some text first to translate.');
        if (inputText) inputText.focus();
        return;
      }
      setLoading(translateBtn, true);
      hideError();
      if (statusBadge) statusBadge.textContent = 'Translating…';

      try {
        const apiBase = await getStoredApiUrl();
        const res = await fetch(`${apiBase}/api/v1/text/translate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error('Translation request failed.');
        const data = await res.json();
        if (data.was_translated && data.text) {
          inputText.value = data.text;
          persistState({ inputText: data.text });
          if (statusBadge) statusBadge.textContent = `Translated (${data.source_language_name})`;
        } else {
          if (statusBadge) statusBadge.textContent = 'Already English';
        }
      } catch (err) {
        showError(err.message || 'Translation failed.');
        if (statusBadge) statusBadge.textContent = 'Error';
      } finally {
        setLoading(translateBtn, false);
      }
    });
  }

  // Scan page button
  if (scanBtn) {
    scanBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      await scanCurrentPage();
    });
  }

  // ── Scan the current page ───────────────────────────────────────
  async function scanCurrentPage() {
    setLoading(scanBtn, true);
    hideError();
    hideResults();
    if (statusBadge) statusBadge.textContent = 'Scanning… (~1-2m)';

    try {
      if (typeof chrome === 'undefined' || !chrome.tabs) {
        throw new Error('Chrome extension APIs not available in this context.');
      }

      let [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
      if (!tab) {
        [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      }

      if (!tab || !tab.id) {
        throw new Error('Cannot access the current tab. Please ensure the page is active.');
      }

      // Guard against internal pages
      if (
        tab.url &&
        (tab.url.startsWith('chrome://') ||
          tab.url.startsWith('chrome-extension://') ||
          tab.url.startsWith('about:') ||
          tab.url.startsWith('edge://'))
      ) {
        throw new Error('Cannot scan browser internal pages. Navigate to a news article and try again.');
      }

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractPageText,
      });

      const rawResult = results?.[0]?.result;
      const pageText = typeof rawResult === 'object' && rawResult !== null ? rawResult.text : rawResult;
      const detectedPageLang = typeof rawResult === 'object' && rawResult !== null ? rawResult.language : '';

      if (!pageText || pageText.trim().length < 20) {
        throw new Error('Could not extract enough text from this page. You can also highlight/select text on the page with your cursor and click "Scan This Page" again!');
      }

      // Show preview of extracted text
      if (extractedPreview) {
        extractedPreview.textContent = pageText.length > 500
          ? pageText.slice(0, 500) + '…'
          : pageText;
        extractedPreview.classList.remove('hidden');
      }

      const now = Date.now();
      await persistState({
        activeTab: 'scan',
        vt_status: 'scanning',
        scanStartTime: now,
        extractedText: pageText,
        pageUrl: tab.url || '',
        tabTitle: tab.title || '',
        tabId: tab.id || null,
        windowId: tab.windowId || null,
        error: null,
      });

      if (resetBtn) resetBtn.style.display = 'inline-block';

      await runAnalysis(
        pageText,
        scanBtn,
        detectedPageLang,
        tab.url || '',
        'scan',
        tab.title || '',
        tab.id || null,
        tab.windowId || null
      );
    } catch (err) {
      showError(err.message || 'Error scanning page');
      setLoading(scanBtn, false);
      if (statusBadge) statusBadge.textContent = 'Error';
    }
  }

  // ── Run Analysis ─────────────────────────────────────────────────
  async function runAnalysis(
    text,
    triggerBtn,
    sourceLanguage = '',
    pageUrl = '',
    activeTab = 'scan',
    tabTitle = '',
    tabId = null,
    windowId = null
  ) {
    const isStrictMode = strictModeToggle ? strictModeToggle.checked : false;
    // In Standard Mode (default): focus on primary story claim for ultra-fast 8-12s response
    // In Strict Mode (when toggled): deep multi-claim evaluation with Tier-1 strictness
    const isSingleClaim = !isStrictMode;

    setLoading(triggerBtn, true);
    hideError();
    hideResults();
    if (statusBadge) statusBadge.textContent = isStrictMode ? 'Deep auditing (Strict Tier-1)…' : 'Verifying story… (~8-12s)';
    if (resetBtn) resetBtn.style.display = 'inline-block';

    const now = Date.now();
    await persistState({
      vt_status: 'scanning',
      scanStartTime: now,
      activeTab,
      extractedText: text,
      pageUrl,
      tabTitle,
      tabId,
      windowId,
      strictMode: isStrictMode,
      singleClaim: isSingleClaim,
      error: null,
    });

    startStatusPolling();

    // Trigger background service worker for badge and desktop notification
    if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
      chrome.runtime.sendMessage({
        action: 'START_ANALYSIS',
        payload: {
          text,
          activeTab,
          sourceLanguage,
          pageUrl,
          tabTitle,
          tabId,
          windowId,
          strictMode: isStrictMode,
          singleClaim: isSingleClaim,
        },
      }).then((resp) => {
        if (resp?.status === 'done' && resp?.data) {
          stopStatusPolling();
          renderResults(resp.data);
          if (statusBadge) statusBadge.textContent = 'Done';
          setLoading(triggerBtn, false);
        }
      }).catch(() => {});
    }

    // Direct fetch in parallel so open popup receives immediate response without IPC lag
    await runDirectFetch(text, triggerBtn, sourceLanguage, isStrictMode, isSingleClaim);
  }

  // Direct fetch execution
  async function runDirectFetch(text, triggerBtn, sourceLanguage = '', strictMode = false, singleClaim = false) {
    const MAX_CHARS = 5000;
    const trimmedText = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;
    const apiBase = await getStoredApiUrl();

    // Abort controller with 75-second safety limit
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 75000);

    try {
      const res = await fetch(`${apiBase}/api/v1/text/check-news`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: trimmedText,
          max_claims: singleClaim ? 1 : 3,
          single_claim: Boolean(singleClaim),
          strict_mode: Boolean(strictMode),
          include_evidence: true,
          ...(sourceLanguage ? { source_language: sourceLanguage } : {}),
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(errBody.message || errBody.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      stopStatusPolling();
      renderResults(data);

      if (statusBadge) statusBadge.textContent = 'Done';
      if (resetBtn) resetBtn.style.display = 'inline-block';

      await persistState({
        vt_status: 'done',
        resultData: data,
        scanCompletedTime: Date.now(),
        error: null,
      });
    } catch (err) {
      clearTimeout(timeoutId);
      stopStatusPolling();
      console.error('[VeriTrust] Direct analysis error:', err);
      if (err.name === 'AbortError') {
        showError('Verification request timed out. Please try again.');
      } else if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
        showError('Cannot connect to VeriTrust AI backend. Please verify the server is running on port 8001.');
      } else {
        showError(err.message || 'Analysis failed.');
      }
      if (statusBadge) statusBadge.textContent = 'Error';
    } finally {
      setLoading(triggerBtn, false);
    }
  }

  // ── Render results ──────────────────────────────────────────────
  function renderResults(data) {
    if (!resultsSection) return;

    // Multilingual translation banner
    if (translationBanner && data.translation?.was_translated) {
      const srcName = data.translation.source_language_name || 'non-English';
      if (translationText) {
        translationText.textContent = `Translated from ${srcName} to English for fact-checking`;
      }
      translationBanner.classList.remove('hidden');

      if (viewOriginalBtn && originalTextContainer) {
        let showingOrig = false;
        originalTextContainer.textContent = data.translation.original_text || '';
        originalTextContainer.classList.add('hidden');
        viewOriginalBtn.textContent = 'View Original';

        viewOriginalBtn.onclick = (e) => {
          e.preventDefault();
          showingOrig = !showingOrig;
          if (showingOrig) {
            originalTextContainer.classList.remove('hidden');
            viewOriginalBtn.textContent = 'Hide Original';
          } else {
            originalTextContainer.classList.add('hidden');
            viewOriginalBtn.textContent = 'View Original';
          }
        };
      }
    } else if (translationBanner) {
      translationBanner.classList.add('hidden');
      if (originalTextContainer) originalTextContainer.classList.add('hidden');
    }

    // Verdict banner
    const verdict = data.verdict || 'Analysis Complete';
    const verdictClass = getVerdictClass(verdict);
    if (verdictBanner) {
      verdictBanner.className = verdictClass;
      if (data.meta?.strict_mode) {
        verdictBanner.innerHTML = `${escapeHtml(verdict)} <span class="mode-pill">Strict Mode</span>`;
      } else {
        verdictBanner.textContent = verdict;
      }
    }

    // Summary & explanation
    if (summaryText) summaryText.textContent = data.summary || data.rationale || '';
    if (explanationText) explanationText.textContent = data.explanation || '';

    // Claims
    if (claimsList) {
      claimsList.innerHTML = '';
      if (data.claims && data.claims.length > 0) {
        for (const claim of data.claims) {
          claimsList.appendChild(buildClaimCard(claim));
        }
      }
    }

    // Analysis notes (from meta)
    if (analysisNotes) {
      if (data.meta?.notes) {
        const notesArr = Array.isArray(data.meta.notes) ? data.meta.notes : [data.meta.notes];
        analysisNotes.textContent = notesArr.join('\n');
        analysisNotes.classList.remove('hidden');
      } else {
        analysisNotes.classList.add('hidden');
      }
    }

    resultsSection.classList.remove('hidden');
  }

  function buildClaimCard(claim) {
    const card = document.createElement('div');
    card.className = 'claim-card';

    const verdictClass = getVerdictClass(claim.verdict);

    // Combine all sources
    const sources = [
      ...(claim.supporting_sources || []),
      ...(claim.refuting_sources || []),
    ];

    let sourcesHtml = '';
    if (sources.length > 0) {
      const items = sources.map((s) => {
        const label = s.title || s.outlet || 'Source';
        const link = s.url && /^https?:\/\//i.test(s.url)
          ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`
          : escapeHtml(label);
        const stance = s.stance ? `<span class="source-stance"> — ${escapeHtml(s.stance)}</span>` : '';
        return `<li>${link}${stance}</li>`;
      }).join('');
      sourcesHtml = `<ul class="source-list">${items}</ul>`;
    }

    card.innerHTML = `
      <p class="claim-text">"${escapeHtml(claim.claim || '')}"</p>
      <div class="claim-meta">
        <span class="verdict-pill ${verdictClass}">${escapeHtml(claim.verdict || 'Unverified')}</span>
        ${claim.confidence ? `<span class="confidence-label">Evidence: ${escapeHtml(claim.confidence)}</span>` : ''}
      </div>
      ${claim.explanation ? `<p class="claim-explanation">${escapeHtml(claim.explanation)}</p>` : ''}
      ${sourcesHtml}
    `;

    return card;
  }

  // ── Helpers ──────────────────────────────────────────────────────
  function getVerdictClass(verdict) {
    if (!verdict) return 'unverified';
    const v = String(verdict).toLowerCase();
    if (v.includes('supported') || v.includes('confirmed') || v.includes('true')) return 'supported';
    if (v.includes('contradicted') || v.includes('refuted') || v.includes('false')) return 'contradicted';
    if (v.includes('disputed') || v.includes('mixed')) return 'disputed';
    return 'unverified';
  }

  function setLoading(btn, loading) {
    if (!btn) return;
    const label = btn.querySelector('.btn-label');
    const spinner = btn.querySelector('.spinner');
    if (loading) {
      btn.disabled = true;
      if (label) label.textContent = 'Scanning… (~1-2m)';
      if (spinner) spinner.classList.remove('hidden');
    } else {
      btn.disabled = false;
      if (label) {
        if (btn === analyzeBtn) label.textContent = 'Analyze Text';
        else if (btn === translateBtn) label.textContent = '🌐 Translate';
        else label.textContent = 'Scan This Page';
      }
      if (spinner) spinner.classList.add('hidden');
    }
  }

  function showError(msg) {
    if (!errorBanner) return;
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function hideError() {
    if (!errorBanner) return;
    errorBanner.classList.add('hidden');
    errorBanner.textContent = '';
  }

  function hideResults() {
    if (resultsSection) resultsSection.classList.add('hidden');
    if (claimsList) claimsList.innerHTML = '';
    if (analysisNotes) analysisNotes.classList.add('hidden');
    if (extractedPreview) extractedPreview.classList.add('hidden');
    if (translationBanner) translationBanner.classList.add('hidden');
    if (originalTextContainer) originalTextContainer.classList.add('hidden');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }
}

// ── Extract visible article text from the page (injected) ─────────
function extractPageText() {
  // 1. First priority: User highlight / selection
  try {
    const selectedText = window.getSelection()?.toString()?.trim() || '';
    if (selectedText.length >= 25) {
      return {
        text: selectedText,
        language: (document.documentElement.lang || '').slice(0, 2).toLowerCase(),
      };
    }
  } catch {}

  // 2. Structured Metadata (Schema.org JSON-LD NewsArticle / Article)
  let jsonLdBody = '';
  let jsonLdHeadline = '';
  try {
    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
    for (const s of scripts) {
      try {
        const raw = s.textContent?.trim();
        if (!raw) continue;
        const data = JSON.parse(raw);
        const items = Array.isArray(data) ? data : (data['@graph'] || [data]);
        for (const item of items) {
          const type = (item['@type'] || '').toString();
          if (type.includes('Article') || type.includes('News') || type.includes('Report')) {
            if (item.headline && !jsonLdHeadline) jsonLdHeadline = String(item.headline).trim();
            if (item.articleBody && item.articleBody.length > 50 && !jsonLdBody) {
              jsonLdBody = String(item.articleBody).trim();
              break;
            } else if (item.description && item.description.length > 50 && !jsonLdBody) {
              jsonLdBody = String(item.description).trim();
            }
          }
        }
      } catch {}
      if (jsonLdBody) break;
    }
  } catch {}

  // 3. Headline extraction
  let headline = jsonLdHeadline;
  if (!headline) {
    const h1 = document.querySelector('article h1, main h1, [role="main"] h1, [itemprop="headline"], .story__headline, .article-title, h1');
    if (h1 && h1.innerText?.trim()) {
      headline = h1.innerText.trim();
    } else {
      const ogTitle = document.querySelector('meta[property="og:title"]')?.content ||
                      document.querySelector('meta[name="twitter:title"]')?.content;
      if (ogTitle && ogTitle.trim()) {
        headline = ogTitle.trim();
      } else if (document.title) {
        headline = document.title.replace(/\s*[-–—|]\s*[^ -–—|]+$/, '').trim();
      }
    }
  }

  // 4. Meta description / lead summary
  const metaDesc = (
    document.querySelector('meta[property="og:description"]')?.content ||
    document.querySelector('meta[name="twitter:description"]')?.content ||
    document.querySelector('meta[name="description"]')?.content ||
    ''
  ).trim();

  // 5. Semantic content candidates
  const candidates = [
    document.querySelector('[itemprop="articleBody"]'),
    document.querySelector('.story__content'),
    document.querySelector('.story_details'),
    document.querySelector('.article__content'),
    document.querySelector('.article-body'),
    document.querySelector('.story-body'),
    document.querySelector('.story-content'),
    document.querySelector('.post-content'),
    document.querySelector('.entry-content'),
    document.querySelector('.instory_body'),
    document.querySelector('#story'),
    document.querySelector('#article-body'),
    ...document.querySelectorAll('article'),
    document.querySelector('main'),
    document.querySelector('[role="main"]'),
  ].filter(Boolean);

  let container = null;
  let maxLength = 0;

  for (const el of candidates) {
    const txt = el.innerText?.trim() || '';
    if (txt.length > maxLength) {
      maxLength = txt.length;
      container = el;
    }
  }

  if (!container || maxLength < 80) {
    container = document.body;
  }

  // 6. Collect content paragraphs
  const junkTerms = [
    'copyright', 'all rights reserved', 'terms of service', 'privacy policy',
    'sign up', 'subscribe to', 'follow us on', 'advertisement', 'recommended stories'
  ];

  let paragraphs = Array.from(container.querySelectorAll('p, div.paragraph, div.story-paragraph, div[class*="content"] > p'))
    .map((p) => p.innerText?.trim())
    .filter((t) => {
      if (!t || t.length < 20) return false;
      const lower = t.toLowerCase();
      if (junkTerms.some(term => lower.includes(term))) return false;
      if (lower.startsWith('read more') || lower.startsWith('click here') || lower.startsWith('photo:')) return false;
      return true;
    });

  // If few paragraphs found, augment with JSON-LD body or meta description
  if (paragraphs.length === 0 && jsonLdBody) {
    paragraphs = [jsonLdBody];
  } else if (paragraphs.length === 0 && metaDesc && metaDesc.length >= 25) {
    paragraphs = [metaDesc];
  }

  const parts = [];
  if (headline) parts.push(headline);
  if (metaDesc && metaDesc !== headline && !paragraphs.some(p => p.includes(metaDesc.slice(0, 30)))) {
    parts.push(metaDesc);
  }
  parts.push(...paragraphs.slice(0, 10));

  const docLang = (
    document.documentElement.lang ||
    document.querySelector('meta[property="og:locale"]')?.content ||
    document.querySelector('meta[name="language"]')?.content ||
    document.querySelector('meta[http-equiv="content-language"]')?.content ||
    ''
  ).toLowerCase().split('-')[0].split('_')[0].trim();

  let extractedText = parts.length > 0 ? parts.join('\n\n').trim() : '';

  // Ultimate fallback if still short
  if (extractedText.length < 25) {
    const rawBody = (container.innerText || document.body?.innerText || '').trim();
    // take top lines from rawBody
    const lines = rawBody.split('\n').map(l => l.trim()).filter(l => l.length > 25);
    extractedText = lines.slice(0, 6).join('\n\n');
  }

  return {
    text: extractedText,
    language: docLang,
  };
}

// ── Bootstrap ───────────────────────────────────────────────────
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initPopup);
} else {
  initPopup();
}
