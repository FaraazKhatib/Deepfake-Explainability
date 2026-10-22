/* ─────────────────────────────────────────────────────────────────
   VeriTrust AI — Chrome Extension Popup Logic
   Calls the same POST /api/v1/text/check-news endpoint as the web app.
   ───────────────────────────────────────────────────────────────── */

// ── DOM helper ───────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

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
    return localStorage.getItem('veritrust_api_url') || 'http://127.0.0.1:8001';
  } catch {
    return 'http://127.0.0.1:8001';
  }
}

async function setStoredApiUrl(url) {
  try {
    if (typeof chrome !== 'undefined' && chrome?.storage?.local) {
      await chrome.storage.local.set({ apiUrl: url });
    }
  } catch (err) {
    console.warn('[VeriTrust] chrome.storage set error:', err);
  }
  try {
    localStorage.setItem('veritrust_api_url', url);
  } catch {}
}

// ── Main Initialization ──────────────────────────────────────────
async function initPopup() {
  console.log('[VeriTrust] Initializing popup...');

  const inputText       = $('#inputText');
  const analyzeBtn      = $('#analyzeBtn');
  const scanBtn         = $('#scanBtn');
  const extractedPreview = $('#extractedPreview');
  const resultsSection  = $('#results');
  const verdictBanner   = $('#verdictBanner');
  const summaryText     = $('#summaryText');
  const explanationText = $('#explanationText');
  const claimsList      = $('#claimsList');
  const analysisNotes   = $('#analysisNotes');
  const errorBanner     = $('#errorBanner');
  const statusBadge     = $('#statusBadge');
  const apiUrlInput     = $('#apiUrl');

  // Restore saved backend URL
  try {
    const savedUrl = await getStoredApiUrl();
    if (savedUrl && apiUrlInput) {
      apiUrlInput.value = savedUrl;
    }
  } catch (e) {
    console.warn('[VeriTrust] Failed to restore apiUrl:', e);
  }

  // Save backend URL on change
  if (apiUrlInput) {
    apiUrlInput.addEventListener('change', () => {
      const val = apiUrlInput.value.trim();
      setStoredApiUrl(val);
    });
  }

  // Tab switching
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const tabTarget = tab.dataset.tab;
      $$('.tab').forEach((t) => t.classList.remove('active'));
      $$('.tab-content').forEach((c) => c.classList.remove('active'));

      tab.classList.add('active');
      const targetContent = $(`#tab-${tabTarget}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });

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
      await runAnalysis(text, analyzeBtn);
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

    try {
      if (typeof chrome === 'undefined' || !chrome.tabs) {
        throw new Error('Chrome extension APIs not available in this context.');
      }

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.id) {
        throw new Error('Cannot access the current tab.');
      }

      // Guard against internal pages
      if (tab.url && (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('about:') || tab.url.startsWith('edge://'))) {
        throw new Error('Cannot scan browser internal pages. Navigate to a news article and try again.');
      }

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: extractPageText,
      });

      const pageText = results?.[0]?.result;
      if (!pageText || pageText.trim().length < 20) {
        throw new Error('Could not extract enough text from this page. Try copying and pasting the article text into the "Paste Text" tab instead.');
      }

      // Show preview of extracted text
      if (extractedPreview) {
        extractedPreview.textContent = pageText.length > 500
          ? pageText.slice(0, 500) + '…'
          : pageText;
        extractedPreview.classList.remove('hidden');
      }

      await runAnalysis(pageText, scanBtn);
    } catch (err) {
      showError(err.message || 'Error scanning page');
      setLoading(scanBtn, false);
    }
  }

  // ── Call the backend ────────────────────────────────────────────
  async function runAnalysis(text, triggerBtn) {
    setLoading(triggerBtn, true);
    hideError();
    hideResults();
    if (statusBadge) statusBadge.textContent = 'Analyzing…';

    // Truncate very long text to avoid overwhelming the backend
    const MAX_CHARS = 5000;
    const trimmedText = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;

    const apiBase = apiUrlInput ? apiUrlInput.value.trim().replace(/\/+$/, '') : 'http://127.0.0.1:8001';

    try {
      const res = await fetch(`${apiBase}/api/v1/text/check-news`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: trimmedText,
          max_claims: 3,
          single_claim: false,
          include_evidence: true,
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(errBody.message || errBody.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      renderResults(data);
      if (statusBadge) statusBadge.textContent = 'Done';
    } catch (err) {
      console.error('[VeriTrust] Analysis error:', err);
      if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
        showError(`Cannot connect to VeriTrust server at ${apiBase}.\nMake sure the backend is running (python -m uvicorn app.main:app --port 8001).`);
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

    // Verdict banner
    const verdict = data.verdict || 'Analysis Complete';
    const verdictClass = getVerdictClass(verdict);
    if (verdictBanner) {
      verdictBanner.textContent = verdict;
      verdictBanner.className = verdictClass;
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
      if (label) label.textContent = 'Analyzing…';
      if (spinner) spinner.classList.remove('hidden');
    } else {
      btn.disabled = false;
      if (label) {
        label.textContent = btn === analyzeBtn ? 'Analyze Text' : 'Scan This Page';
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
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  console.log('[VeriTrust] Popup successfully initialized and event listeners bound.');
}

// ── Extract visible article text from the page (injected) ─────────
function extractPageText() {
  const candidates = [
    ...document.querySelectorAll('article'),
    document.querySelector('main'),
    document.querySelector('[role="main"]'),
    document.querySelector('.post-content'),
    document.querySelector('.article-body'),
    document.querySelector('.story-body'),
    document.querySelector('.entry-content'),
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

  if (!container || maxLength < 100) {
    container = document.body;
  }

  const paragraphs = container.querySelectorAll('p');
  if (paragraphs.length > 3) {
    const texts = Array.from(paragraphs)
      .map((p) => p.innerText?.trim())
      .filter((t) => t && t.length > 20);
    if (texts.length > 0) {
      return texts.join('\n\n');
    }
  }

  return container.innerText?.trim() || '';
}

// ── Bootstrap: run whether DOM is loading or already interactive ──
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initPopup);
} else {
  initPopup();
}
