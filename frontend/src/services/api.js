// Development requests use Vite's same-origin proxy so localhost/127.0.0.1
// and browser origin policies do not prevent uploads.
const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '' : 'http://localhost:8001');

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/v1/video/health`);
  if (!res.ok) throw new Error('Backend unavailable');
  return res.json();
}

async function uploadFile(endpoint, file, onStage) {
  const formData = new FormData();
  formData.append('file', file);
  onStage?.('uploading');

  const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body: formData });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(error.detail || error.error || error.message || 'Server error');
  }
  return await res.json();
}

export async function analyzeVideo(file, onStage) {
  return await uploadFile('/api/v1/video/analyze', file, onStage);
}

export async function analyzeImage(file, onStage) {
  return await uploadFile('/api/v1/image_audio/analyze?modality=image', file, onStage);
}

export async function analyzeAudio(file, onStage) {
  return await uploadFile('/api/v1/image_audio/analyze?modality=audio', file, onStage);
}

export async function analyzeText(text, onStage, options = {}) {
  onStage?.('analyzing');
  const { single_claim = false, strict_mode = false } = options;
  const res = await fetch(`${API_BASE}/api/v1/text/check-news`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      max_claims: single_claim ? 1 : 3,
      single_claim: Boolean(single_claim),
      strict_mode: Boolean(strict_mode),
      include_evidence: true,
    })
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(error.message || error.detail || 'Server error');
  }
  return await res.json();
}

export async function translateText(text) {
  const res = await fetch(`${API_BASE}/api/v1/text/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(error.message || error.detail || 'Translation failed');
  }
  return await res.json();
}

export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export async function verifyAdminKey(adminKey) {
  const res = await fetch(`${API_BASE}/api/v1/text/feedback/verify-key`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Key': adminKey.trim(),
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Invalid key' }));
    throw new Error(error.detail || error.message || 'Collaborator access denied');
  }
  return await res.json();
}

export async function fetchFeedbackList(adminKey) {
  const res = await fetch(`${API_BASE}/api/v1/text/feedback/list`, {
    headers: {
      'X-Admin-Key': adminKey.trim(),
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to fetch list' }));
    throw new Error(error.detail || error.message || 'Could not load feedback');
  }
  return await res.json();
}

export async function reviewFeedback(adminKey, scanId, action, source = '', reason = '') {
  const res = await fetch(`${API_BASE}/api/v1/text/feedback/review`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Key': adminKey.trim(),
    },
    body: JSON.stringify({
      scan_id: scanId,
      action,
      source: source.trim(),
      reason: reason.trim(),
    }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Review action failed' }));
    throw new Error(error.detail || error.message || 'Action failed');
  }
  return await res.json();
}
