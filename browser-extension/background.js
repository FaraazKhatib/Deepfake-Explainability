/* ─────────────────────────────────────────────────────────────────
   VeriTrust AI — Background Service Worker (Manifest V3)
   - Performs long-running page scans and text analyses in the background
   - Persists state in chrome.storage.local so scans never vanish when
     popups close or when users dismiss website modals/notifications
   - Enforces a 180-second fetch timeout to allow full CPU evidence evaluation
   - Updates the extension toolbar badge with scan progress & results
   ───────────────────────────────────────────────────────────────── */

const DEFAULT_API_URL = 'http://127.0.0.1:8001';
let currentAbortController = null;

async function getApiUrl() {
  try {
    const stored = await chrome.storage.local.get('apiUrl');
    return stored?.apiUrl || DEFAULT_API_URL;
  } catch {
    return DEFAULT_API_URL;
  }
}

// ── Message Listener ─────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'START_ANALYSIS') {
    handleAnalysis(message.payload).then((data) => {
      try {
        sendResponse({ status: 'done', data });
      } catch {}
    }).catch((err) => {
      try {
        sendResponse({ status: 'error', error: err?.message || 'Analysis failed' });
      } catch {}
    });
    return true; // Keep message channel open and service worker active
  }

  if (message.action === 'CANCEL_ANALYSIS' || message.action === 'CLEAR_BADGE') {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
    chrome.action.setBadgeText({ text: '' }).catch(() => {});
    if (message.action === 'CANCEL_ANALYSIS') {
      chrome.storage.local.remove('vt_state').catch(() => {});
    }
    sendResponse({ status: 'cleared' });
    return false;
  }

  return false;
});

// ── Background Scan Execution ────────────────────────────────────
async function handleAnalysis({
  text,
  activeTab,
  sourceLanguage,
  pageUrl,
  tabTitle,
  tabId,
  windowId,
  strictMode = false,
  singleClaim = false,
}) {
  if (!text || !text.trim()) return;

  // Cancel any prior in-flight request if a new one is requested
  if (currentAbortController) {
    try {
      currentAbortController.abort();
    } catch {}
  }
  currentAbortController = new AbortController();
  const signal = currentAbortController.signal;

  const MAX_CHARS = 5000;
  const trimmedText = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;

  // 1. Mark status as 'scanning' in chrome.storage.local
  const scanningState = {
    vt_status: 'scanning',
    activeTab: activeTab || 'scan',
    extractedText: text,
    pageUrl: pageUrl || '',
    tabTitle: tabTitle || '',
    tabId: tabId || null,
    windowId: windowId || null,
    strictMode: Boolean(strictMode),
    singleClaim: Boolean(singleClaim),
    scanStartTime: Date.now(),
    resultData: null,
    error: null,
  };

  try {
    await chrome.storage.local.set({ vt_state: scanningState });
  } catch (err) {
    console.warn('[VeriTrust BG] Storage write error:', err);
  }

  // Set toolbar badge to 'SCAN' (blue)
  try {
    await chrome.action.setBadgeText({ text: 'SCAN' });
    await chrome.action.setBadgeBackgroundColor({ color: '#3b82f6' });
  } catch {}

  // 2. Perform the fetch with a 90-second safety timeout
  const timeoutId = setTimeout(() => {
    try {
      if (currentAbortController) {
        currentAbortController.abort();
      }
    } catch {}
  }, 90000);

  try {
    const apiBase = await getApiUrl();
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
      signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(errBody.message || errBody.detail || `Server error (${res.status})`);
    }

    const data = await res.json();

    // 3. Persist completed state
    const doneState = {
      ...scanningState,
      vt_status: 'done',
      resultData: data,
      scanCompletedTime: Date.now(),
      error: null,
    };

    await chrome.storage.local.set({ vt_state: doneState });

    // Update badge based on verdict
    try {
      const v = (data.verdict || '').toLowerCase();
      if (v.includes('supported') || v.includes('true')) {
        await chrome.action.setBadgeText({ text: 'TRUE' });
        await chrome.action.setBadgeBackgroundColor({ color: '#10b981' }); // Green
      } else if (v.includes('contradicted') || v.includes('false')) {
        await chrome.action.setBadgeText({ text: 'FAKE' });
        await chrome.action.setBadgeBackgroundColor({ color: '#ef4444' }); // Red
      } else {
        await chrome.action.setBadgeText({ text: 'DONE' });
        await chrome.action.setBadgeBackgroundColor({ color: '#6366f1' }); // Indigo
      }
    } catch {}

    // Show Chrome desktop notification so user knows result even if browsing another tab
    try {
      if (typeof chrome !== 'undefined' && chrome.notifications?.create) {
        const v = (data.verdict || '').toLowerCase();
        let statusTitle = 'Scan Complete';
        if (v.includes('supported') || v.includes('true')) {
          statusTitle = 'Verified (Likely True)';
        } else if (v.includes('contradicted') || v.includes('false')) {
          statusTitle = 'Contradicted (Likely False)';
        } else {
          statusTitle = 'Unverified / Developing';
        }

        let domain = 'News Article';
        try {
          if (pageUrl) domain = new URL(pageUrl).hostname.replace('www.', '');
        } catch {}

        const displayHeadline = tabTitle ? `"${tabTitle.slice(0, 55)}..."` : domain;
        chrome.notifications.create('vt_scan_done_' + Date.now(), {
          type: 'basic',
          iconUrl: chrome.runtime.getURL('icons/icon128.png'),
          title: `VeriTrust AI: ${statusTitle}`,
          message: `${displayHeadline}\n${data.summary ? data.summary.slice(0, 110) : 'Click to inspect sources & evidence.'}`,
          priority: 2,
        });
      }
    } catch (notifErr) {
      console.warn('[VeriTrust BG] Notification error:', notifErr);
    }

    return data;

  } catch (err) {
    clearTimeout(timeoutId);
    console.error('[VeriTrust BG] Analysis error:', err);

    let errMsg = err.message || 'Analysis failed.';
    if (err.name === 'AbortError') {
      errMsg = 'Scan timed out. The backend server took too long to retrieve and process evidence.';
    } else if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
      errMsg = 'Cannot connect to VeriTrust AI backend. Please verify the server is running on port 8001.';
    }

    const errorState = {
      ...scanningState,
      vt_status: 'error',
      error: errMsg,
      scanCompletedTime: Date.now(),
    };

    try {
      await chrome.storage.local.set({ vt_state: errorState });
      await chrome.action.setBadgeText({ text: 'ERR' });
      await chrome.action.setBadgeBackgroundColor({ color: '#ef4444' });
    } catch {}
  } finally {
    currentAbortController = null;
  }
}

// ── Notification Click: Switch back to scanned tab ───────────────
if (typeof chrome !== 'undefined' && chrome.notifications?.onClicked) {
  chrome.notifications.onClicked.addListener(async () => {
    try {
      const stored = await chrome.storage.local.get('vt_state');
      const state = stored?.vt_state;
      if (state?.tabId) {
        await chrome.tabs.update(state.tabId, { active: true });
      }
      if (state?.windowId) {
        await chrome.windows.update(state.windowId, { focused: true });
        if (chrome.sidePanel?.open) {
          await chrome.sidePanel.open({ windowId: state.windowId });
        }
      }
    } catch (err) {
      console.warn('[VeriTrust BG] Notification click navigation error:', err);
    }
  });
}
