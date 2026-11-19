// Development requests use Vite's same-origin proxy so localhost/127.0.0.1
// and browser origin policies do not prevent uploads.
const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '' : 'http://127.0.0.1:8001');

async function requestJson(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (error) {
    throw new Error(
      `Cannot connect to the analysis backend at ${API_BASE || 'this Vite server'}. ` +
      'Make sure the backend is running on 127.0.0.1:8001.'
    );
  }

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(error.detail || error.error || error.message || 'Server error');
  }
  return await res.json();
}

export async function checkHealth() {
  return await requestJson(`${API_BASE}/api/v1/video/health`);
}

async function uploadFile(endpoint, file, onStage) {
  const formData = new FormData();
  formData.append('file', file);
  onStage?.('uploading');

  return await requestJson(`${API_BASE}${endpoint}`, { method: 'POST', body: formData });
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
  return await requestJson(`${API_BASE}/api/v1/text/check-news`, {
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
}

export async function translateText(text) {
  return await requestJson(`${API_BASE}/api/v1/text/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
}

export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}
