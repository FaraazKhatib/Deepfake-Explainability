const DEFAULT_API_URL = 'http://127.0.0.1:8001';
const SCAN_TIMEOUT_MS = 90_000;
let activeScan = null;

// Serialize state, badge and notification changes. Network requests run outside
// this queue so a slow request never delays cancellation or its replacement.
let transitions = Promise.resolve();
function transition(operation) {
  const result = transitions.then(operation);
  transitions = result.catch(() => {});
  return result;
}

async function badge(text, color) {
  try {
    await chrome.action.setBadgeText({ text });
    if (color) await chrome.action.setBadgeBackgroundColor({ color });
  } catch {}
}

async function getState() {
  const { vt_state: state } = await chrome.storage.local.get('vt_state');
  if (state?.vt_status === 'scanning' && !activeScan) {
    // A worker restart loses its fetch, including scans from older versions.
    const interrupted = { ...state, vt_status: 'error', error: 'Scan was interrupted. Please try again.' };
    await chrome.storage.local.set({ vt_state: interrupted });
    await badge('ERR', '#ef4444');
    return interrupted;
  }
  return state || null;
}

async function analyze(job, payload) {
  const { text, singleClaim, strictMode, sourceLanguage, pageUrl } = payload;
  const targetPageUrl = pageUrl || job.state?.pageUrl || '';
  const timeout = setTimeout(() => job.controller.abort(), SCAN_TIMEOUT_MS);
  try {
    const { apiUrl } = await chrome.storage.local.get('apiUrl');
    const response = await fetch((apiUrl || DEFAULT_API_URL) + '/api/v1/text/check-news', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: job.controller.signal,
      body: JSON.stringify({
        text: text.slice(0, 5000), max_claims: singleClaim ? 1 : 3,
        single_claim: Boolean(singleClaim), strict_mode: Boolean(strictMode),
        include_evidence: true, ...(sourceLanguage ? { source_language: sourceLanguage } : {}),
        page_url: targetPageUrl,
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.message || body.detail || `Server error (${response.status})`);
    }
    const data = await response.json();
    await transition(async () => {
      if (activeScan?.id !== job.id) return;
      await chrome.storage.local.set({ vt_state: { ...job.state, vt_status: 'done', resultData: data, scanCompletedTime: Date.now() } });
      activeScan = null;
      const verdict = (data.verdict || '').toLowerCase();
      const supported = verdict.includes('supported') || verdict.includes('true');
      const contradicted = verdict.includes('contradicted') || verdict.includes('false');
      await badge(supported ? 'TRUE' : contradicted ? 'FAKE' : 'DONE', supported ? '#10b981' : contradicted ? '#ef4444' : '#6366f1');
      const title = supported ? 'Verified (Likely True)' : contradicted ? 'Contradicted (Likely False)' : 'Unverified / Developing';
      let headline = 'News Article';
      try { headline = new URL(job.state.pageUrl).hostname.replace(/^www\./, ''); } catch {}
      if (job.state.tabTitle) headline = job.state.tabTitle.slice(0, 55);
      await chrome.notifications?.create('vt_scan_done_' + job.id, {
        type: 'basic', iconUrl: chrome.runtime.getURL('icons/icon128.png'),
        title: 'VeriTrust AI: ' + title, priority: 2,
        message: headline + '\n' + (data.summary?.slice(0, 110) || 'Click to inspect sources & evidence.'),
      }).catch(() => {});
    });
  } catch (err) {
    await transition(async () => {
      if (activeScan?.id !== job.id) return;
      let error = err.message || 'Analysis failed.';
      if (err.name === 'AbortError') error = 'Scan timed out while waiting for the backend.';
      else if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
        error = 'Cannot connect to VeriTrust AI backend. Please verify the server is running on port 8001.';
      }
      await chrome.storage.local.set({ vt_state: { ...job.state, vt_status: 'error', error, scanCompletedTime: Date.now() } });
      activeScan = null;
      await badge('ERR', '#ef4444');
    });
  } finally {
    clearTimeout(timeout);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const { action, scanId, payload } = message;
  if (!['START_ANALYSIS', 'CANCEL_ANALYSIS', 'CLEAR_BADGE', 'GET_SCAN_STATE'].includes(action)) return false;
  const result = transition(async () => {
    if (action === 'GET_SCAN_STATE') return { state: await getState() };
    if (action !== 'START_ANALYSIS') {
      const state = await getState();
      if (scanId !== undefined && scanId !== (state?.scanId ?? null)) return { status: 'superseded' };
      activeScan?.controller.abort();
      activeScan = null;
      await chrome.storage.local.remove('vt_state');
      await badge('');
      return { status: 'cleared' };
    }
    if (!payload?.text?.trim()) throw new Error('Please enter text to analyze.');
    activeScan?.controller.abort();
    activeScan = null;
    const id = scanId || crypto.randomUUID();
    const state = {
      scanId: id, vt_status: 'scanning', activeTab: payload.activeTab || 'scan',
      extractedText: payload.text, pageUrl: payload.pageUrl || '', tabTitle: payload.tabTitle || '',
      tabId: payload.tabId || null, windowId: payload.windowId || null,
      strictMode: Boolean(payload.strictMode), singleClaim: Boolean(payload.singleClaim),
      scanStartTime: Date.now(), deadline: Date.now() + SCAN_TIMEOUT_MS, resultData: null, error: null,
    };
    await chrome.storage.local.set({ vt_state: state });
    const job = { id, state, controller: new AbortController() };
    activeScan = job;
    await badge('SCAN', '#3b82f6');
    return { completion: analyze(job, payload) };
  });
  // Keep the message event alive while the job runs. Its outcome is published
  // in storage even if the popup that sent the message has since closed.
  result.then(async (reply) => {
    if (!reply.completion) return reply;
    await reply.completion;
    return { status: 'finished' };
  }).then(sendResponse, err => sendResponse({ status: 'error', error: err.message }));
  return true;
});

chrome.notifications?.onClicked.addListener(async (notificationId) => {
  try {
    const { vt_state: state } = await chrome.storage.local.get('vt_state');
    if (state?.scanId && notificationId !== 'vt_scan_done_' + state.scanId) return;
    if (state?.tabId) await chrome.tabs.update(state.tabId, { active: true });
    if (state?.windowId) {
      await chrome.windows.update(state.windowId, { focused: true });
      await chrome.sidePanel?.open({ windowId: state.windowId });
    }
  } catch (err) {
    console.warn('[VeriTrust BG] Notification click navigation error:', err);
  }
});
