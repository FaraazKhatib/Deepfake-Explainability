/* ─────────────────────────────────────────────────────────────────
   VeriTrust AI — Chrome Extension Popup Logic
   Calls the same POST /api/v1/text/check-news endpoint as the web app.
   ───────────────────────────────────────────────────────────────── */

// ── DOM refs ────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

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

// ── Tab switching ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Restore saved backend URL
  const stored = await chrome.storage.local.get('apiUrl');
  if (stored.apiUrl) apiUrlInput.value = stored.apiUrl;

  // Tab buttons
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.remove('active'));
      $$('.tab-content').forEach((c) => c.classList.remove('active'));
      tab.classList.add('active');
      $(`#tab-${tab.dataset.tab}`).classList.add('active');
    });
  });

  // Analyze (paste) button
  analyzeBtn.addEventListener('click', async () => {
    const text = inputText.value.trim();
    if (!text) return;
    await runAnalysis(text, analyzeBtn);
  });

  // Scan page button
  scanBtn.addEventListener('click', async () => {
    await scanCurrentPage();
  });

  // Save backend URL on change
  apiUrlInput.addEventListener('change', () => {
    chrome.storage.local.set({ apiUrl: apiUrlInput.value.trim() });
  });
});

// ── Scan the current page ───────────────────────────────────────
async function scanCurrentPage() {
  setLoading(scanBtn, true);
  hideError();
  hideResults();

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.id) {
      throw new Error('Cannot access the current tab.');
    }

    // Guard against chrome:// and other restricted pages
    if (tab.url && (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('about:'))) {
      throw new Error('Cannot scan browser internal pages. Navigate to a news article and try again.');
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageText,
    });

    const pageText = results?.[0]?.result;
    if (!pageText || pageText.trim().length < 20) {
      throw new Error('Could not extract enough text from this page. Try copying and pasting the article text instead.');
    }

    // Show preview of extracted text
    extractedPreview.textContent = pageText.length > 500
      ? pageText.slice(0, 500) + '…'
      : pageText;
    extractedPreview.classList.remove('hidden');

    await runAnalysis(pageText, scanBtn);
  } catch (err) {
    showError(err.message);
    setLoading(scanBtn, false);
  }
}

// ── Extract visible article text from the page ──────────────────
// This function is injected into the page via chrome.scripting.executeScript
function extractPageText() {
  // Try <article> first, then <main>, then fall back to body
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

  // If no semantic container found or it's too short, use body
  if (!container || maxLength < 100) {
    container = document.body;
  }

  // Get all paragraph text from the chosen container
  const paragraphs = container.querySelectorAll('p');
  if (paragraphs.length > 3) {
    const texts = Array.from(paragraphs)
      .map((p) => p.innerText?.trim())
      .filter((t) => t && t.length > 20);
    if (texts.length > 0) {
      return texts.join('\n\n');
    }
  }

  // Fallback: full innerText of container
  return container.innerText?.trim() || '';
}

// ── Call the backend ────────────────────────────────────────────
async function runAnalysis(text, triggerBtn) {
  setLoading(triggerBtn, true);
  hideError();
  hideResults();
  statusBadge.textContent = 'Analyzing…';

  // Truncate very long text to avoid overwhelming the backend
  const MAX_CHARS = 5000;
  const trimmedText = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;

  const apiBase = apiUrlInput.value.trim().replace(/\/+$/, '');

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
    statusBadge.textContent = 'Done';
  } catch (err) {
    if (err.message === 'Failed to fetch') {
      showError(`Cannot reach the backend at ${apiBase}. Make sure your VeriTrust server is running.`);
    } else {
      showError(err.message);
    }
    statusBadge.textContent = 'Error';
  } finally {
    setLoading(triggerBtn, false);
  }
}

// ── Render results ──────────────────────────────────────────────
function renderResults(data) {
  // Verdict banner
  const verdictClass = getVerdictClass(data.verdict);
  verdictBanner.textContent = data.verdict;
  verdictBanner.className = verdictClass;

  // Summary & explanation
  summaryText.textContent = data.summary || '';
  explanationText.textContent = data.explanation || '';

  // Claims
  claimsList.innerHTML = '';
  if (data.claims && data.claims.length > 0) {
    for (const claim of data.claims) {
      claimsList.appendChild(buildClaimCard(claim));
    }
  }

  // Analysis notes (from meta)
  if (data.meta?.notes) {
    const notesArr = Array.isArray(data.meta.notes) ? data.meta.notes : [data.meta.notes];
    analysisNotes.textContent = notesArr.join('\n');
    analysisNotes.classList.remove('hidden');
  } else {
    analysisNotes.classList.add('hidden');
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
      return `<li>${link} <span class="source-stance">— ${escapeHtml(s.stance || '')}</span></li>`;
    }).join('');
    sourcesHtml = `<ul class="source-list">${items}</ul>`;
  }

  card.innerHTML = `
    <p class="claim-text">"${escapeHtml(claim.claim)}"</p>
    <div class="claim-meta">
      <span class="verdict-pill ${verdictClass}">${escapeHtml(claim.verdict)}</span>
      <span class="confidence-label">Evidence: ${escapeHtml(claim.confidence || '')}</span>
    </div>
    <p class="claim-explanation">${escapeHtml(claim.explanation || '')}</p>
    ${sourcesHtml}
  `;

  return card;
}

// ── Helpers ──────────────────────────────────────────────────────
function getVerdictClass(verdict) {
  if (!verdict) return 'unverified';
  const v = verdict.toLowerCase();
  if (v.includes('supported') || v.includes('confirmed')) return 'supported';
  if (v.includes('contradicted') || v.includes('refuted') || v.includes('false')) return 'contradicted';
  if (v.includes('disputed') || v.includes('mixed')) return 'disputed';
  return 'unverified';
}

function setLoading(btn, loading) {
  const label = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.spinner');
  if (loading) {
    btn.disabled = true;
    label.textContent = 'Analyzing…';
    spinner.classList.remove('hidden');
  } else {
    btn.disabled = false;
    label.textContent = btn === analyzeBtn ? 'Analyze Text' : 'Scan This Page';
    spinner.classList.add('hidden');
  }
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.classList.add('hidden');
  errorBanner.textContent = '';
}

function hideResults() {
  resultsSection.classList.add('hidden');
  claimsList.innerHTML = '';
  analysisNotes.classList.add('hidden');
  extractedPreview.classList.add('hidden');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
