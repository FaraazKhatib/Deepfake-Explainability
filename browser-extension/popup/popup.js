/* ─────────────────────────────────────────────────────────────────
   VeriTrust AI — Chrome Extension Popup Logic
   Calls POST /api/v1/text/check-news
   - Backend URL is handled internally (not shown in the UI)
   - State persisted across tab navigation / page opening
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

  const inputText        = $('#inputText');
  const analyzeBtn       = $('#analyzeBtn');
  const translateBtn     = $('#translateBtn');
  const scanBtn          = $('#scanBtn');
  const extractedPreview = $('#extractedPreview');
  const resultsSection   = $('#results');
  const translationBanner = $('#translationBanner');
  const translationText  = $('#translationText');
  const viewOriginalBtn  = $('#viewOriginalBtn');
  const originalTextContainer = $('#originalTextContainer');
  const verdictBanner    = $('#verdictBanner');
  const summaryText      = $('#summaryText');
  const explanationText  = $('#explanationText');
  const claimsList       = $('#claimsList');
  const analysisNotes    = $('#analysisNotes');
  const errorBanner      = $('#errorBanner');
  const statusBadge      = $('#statusBadge');
  const resetBtn         = $('#resetBtn');

  // Restore saved state (so popup does not reset when tabs/pages switch)
  try {
    const saved = await loadState();
    if (saved) {
      if (saved.activeTab === 'scan') {
        switchTab('scan');
      } else {
        switchTab('paste');
      }

      if (saved.inputText && inputText) {
        inputText.value = saved.inputText;
      }

      if (saved.extractedText && extractedPreview) {
        extractedPreview.textContent = saved.extractedText;
        extractedPreview.classList.remove('hidden');
      }

      if (saved.resultData) {
        renderResults(saved.resultData);
        if (statusBadge) statusBadge.textContent = 'Done';
      }

      if (saved.inputText || saved.extractedText || saved.resultData) {
        if (resetBtn) resetBtn.style.display = 'inline-block';
      }
    }
  } catch (e) {
    console.warn('[VeriTrust] Failed to restore saved state:', e);
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

  // Reset / New Check
  if (resetBtn) {
    resetBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      await clearPersistedState();

      if (inputText) inputText.value = '';
      if (extractedPreview) {
        extractedPreview.textContent = '';
        extractedPreview.classList.add('hidden');
      }

      hideResults();
      hideError();

      if (statusBadge) statusBadge.textContent = 'Ready';
      resetBtn.style.display = 'none';
      switchTab('paste');
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
      await runAnalysis(text, analyzeBtn);
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
    if (statusBadge) statusBadge.textContent = 'Scanning…';

    try {
      if (typeof chrome === 'undefined' || !chrome.tabs) {
        throw new Error('Chrome extension APIs not available in this context.');
      }

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.id) {
        throw new Error('Cannot access the current tab.');
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
        throw new Error('Could not extract enough text from this page. Try copying and pasting the article into the "Paste Text" tab instead.');
      }

      // Show preview of extracted text
      if (extractedPreview) {
        extractedPreview.textContent = pageText.length > 500
          ? pageText.slice(0, 500) + '…'
          : pageText;
        extractedPreview.classList.remove('hidden');
      }

      await persistState({
        activeTab: 'scan',
        extractedText: pageText,
        pageUrl: tab.url || '',
      });

      if (resetBtn) resetBtn.style.display = 'inline-block';

      await runAnalysis(pageText, scanBtn, detectedPageLang);
    } catch (err) {
      showError(err.message || 'Error scanning page');
      setLoading(scanBtn, false);
      if (statusBadge) statusBadge.textContent = 'Error';
    }
  }

  // ── Call the backend ────────────────────────────────────────────
  async function runAnalysis(text, triggerBtn, sourceLanguage = '') {
    setLoading(triggerBtn, true);
    hideError();
    hideResults();
    if (statusBadge) statusBadge.textContent = 'Analyzing…';

    // Truncate very long text to avoid overwhelming the backend
    const MAX_CHARS = 5000;
    const trimmedText = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;
    const apiBase = await getStoredApiUrl();

    try {
      const res = await fetch(`${apiBase}/api/v1/text/check-news`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: trimmedText,
          max_claims: 3,
          single_claim: false,
          include_evidence: true,
          ...(sourceLanguage ? { source_language: sourceLanguage } : {}),
        }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(errBody.message || errBody.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      renderResults(data);

      if (statusBadge) statusBadge.textContent = 'Done';
      if (resetBtn) resetBtn.style.display = 'inline-block';

      // Persist results across tab/page switches
      await persistState({
        resultData: data,
      });
    } catch (err) {
      console.error('[VeriTrust] Analysis error:', err);
      if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
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
  let headline = '';
  const h1 = document.querySelector('article h1, main h1, [role="main"] h1, h1');
  if (h1 && h1.innerText?.trim()) {
    headline = h1.innerText.trim();
  } else {
    const ogTitle = document.querySelector('meta[property="og:title"]')?.content;
    if (ogTitle && ogTitle.trim()) {
      headline = ogTitle.trim();
    } else if (document.title) {
      headline = document.title.replace(/\s*[-–—|]\s*[^ -–—|]+$/, '').trim();
    }
  }

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

  const paragraphs = Array.from(container.querySelectorAll('p'))
    .map((p) => p.innerText?.trim())
    .filter((t) => {
      if (!t || t.length < 25) return false;
      const lower = t.toLowerCase();
      if (lower.includes('copyright') || lower.includes('all rights reserved') || lower.includes('terms of service')) return false;
      if (lower.startsWith('read more') || lower.startsWith('follow us') || lower.startsWith('sign up')) return false;
      return true;
    });

  const leadParagraphs = paragraphs.slice(0, 10);

  const parts = [];
  if (headline) {
    parts.push(headline);
  }
  if (leadParagraphs.length > 0) {
    parts.push(...leadParagraphs);
  }

  const docLang = (
    document.documentElement.lang ||
    document.querySelector('meta[property="og:locale"]')?.content ||
    document.querySelector('meta[name="language"]')?.content ||
    document.querySelector('meta[http-equiv="content-language"]')?.content ||
    ''
  ).toLowerCase().split('-')[0].split('_')[0].trim();

  const extractedText = parts.length > 0 ? parts.join('\n\n') : (container.innerText?.trim() || '');
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
