import { useState, useEffect } from 'react';
import {
  Lock,
  Unlock,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  ExternalLink,
  RefreshCw,
  AlertCircle,
  Key,
  Trash2,
  Undo2,
  Globe,
  MessageSquare,
  Check,
  Sparkles
} from 'lucide-react';
import {
  verifyAdminKey,
  fetchFeedbackList,
  reviewFeedback
} from '../services/api';

const DEFAULT_PRESETS = [
  { label: 'Debunked false hoax / satire', reason: 'Fact-checking and wire coverage confirm this claim is a false fabrication or satire.' },
  { label: 'Confirmed by primary newsrooms', reason: 'Confirmed by multiple independent Tier-1 wire and news organizations.' },
  { label: 'Disputed with conflicting statements', reason: 'Conflicting official accounts and evolving reporting leave this dispute unresolved.' },
];

export default function FeedbackReviewPage() {
  const [adminKey, setAdminKey] = useState(() => {
    return localStorage.getItem('vt_admin_key') || sessionStorage.getItem('vt_admin_key') || '';
  });
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [rememberKey, setRememberKey] = useState(true);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(null);
  const [authError, setAuthError] = useState(null);
  const [feedbackItems, setFeedbackItems] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all' | 'pending' | 'approved'
  const [activeApproveScanId, setActiveApproveScanId] = useState(null);
  const [sourceUrl, setSourceUrl] = useState('');
  const [reasonText, setReasonText] = useState('');
  const [actionError, setActionError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Auto-verify stored key on load
  useEffect(() => {
    if (adminKey) {
      handleVerify(adminKey);
    }
  }, []);

  const handleVerify = async (keyToTest) => {
    setLoading(true);
    setAuthError(null);
    try {
      await verifyAdminKey(keyToTest);
      setIsAuthenticated(true);
      setAdminKey(keyToTest);
      if (rememberKey) {
        localStorage.setItem('vt_admin_key', keyToTest);
      } else {
        sessionStorage.setItem('vt_admin_key', keyToTest);
      }
      loadList(keyToTest);
    } catch (err) {
      setIsAuthenticated(false);
      setAuthError(err.message || 'Invalid collaborator key');
    } finally {
      setLoading(false);
    }
  };

  const loadList = async (key) => {
    setLoading(true);
    setActionError(null);
    try {
      const data = await fetchFeedbackList(key || adminKey);
      setFeedbackItems(data.items || []);
    } catch (err) {
      setActionError(err.message || 'Failed to load feedback list');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('vt_admin_key');
    sessionStorage.removeItem('vt_admin_key');
    setAdminKey('');
    setIsAuthenticated(false);
    setFeedbackItems([]);
    setKeyInput('');
  };

  const handleApproveSubmit = async (scanId) => {
    if (!sourceUrl.trim() || !reasonText.trim()) {
      setActionError('Both an Evidence URL (HTTP/HTTPS) and a reason are required for approvals.');
      return;
    }
    setActionLoading(scanId);
    setActionError(null);
    try {
      await reviewFeedback(adminKey, scanId, 'approve', sourceUrl, reasonText);
      setSuccessMessage('Override successfully approved and active!');
      setActiveApproveScanId(null);
      setSourceUrl('');
      setReasonText('');
      await loadList(adminKey);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err) {
      setActionError(err.message || 'Failed to approve feedback');
    } finally {
      setActionLoading(null);
    }
  };

  const handleRevoke = async (scanId) => {
    setActionLoading(scanId);
    setActionError(null);
    try {
      await reviewFeedback(adminKey, scanId, 'revoke');
      setSuccessMessage('Approval revoked. Default automated checking restored.');
      await loadList(adminKey);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err) {
      setActionError(err.message || 'Failed to revoke approval');
    } finally {
      setActionLoading(null);
    }
  };

  const handleDismiss = async (scanId) => {
    if (!window.confirm('Are you sure you want to dismiss and delete this feedback submission?')) {
      return;
    }
    setActionLoading(scanId);
    setActionError(null);
    try {
      await reviewFeedback(adminKey, scanId, 'dismiss');
      setSuccessMessage('Feedback submission dismissed.');
      await loadList(adminKey);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err) {
      setActionError(err.message || 'Failed to dismiss feedback');
    } finally {
      setActionLoading(null);
    }
  };

  const getVerdictBadge = (verdict) => {
    const v = (verdict || '').toLowerCase();
    if (v.includes('supported') || v.includes('true')) {
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200">Supported</span>;
    }
    if (v.includes('contradicted') || v.includes('false')) {
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">Contradicted</span>;
    }
    if (v.includes('disputed')) {
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">Disputed</span>;
    }
    return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200">Unverified</span>;
  };

  const filteredItems = feedbackItems.filter(item => {
    if (filter === 'pending') return !item.is_approved;
    if (filter === 'approved') return item.is_approved;
    return true;
  });

  const pendingCount = feedbackItems.filter(i => !i.is_approved).length;
  const approvedCount = feedbackItems.filter(i => i.is_approved).length;

  // ─── 1. LOCKED VIEW ─────────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <div className="p-8 max-w-xl mx-auto mt-12">
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-200 text-center">
          <div className="w-14 h-14 bg-indigo-50 border border-indigo-100 rounded-2xl flex items-center justify-center mx-auto mb-4 text-indigo-600">
            <Lock size={28} />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Collaborator Portal</h2>
          <p className="text-sm text-gray-600 mb-6">
            Review and override community verdict corrections. Access is restricted to repository collaborators.
          </p>

          {authError && (
            <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-xl text-xs flex items-center gap-2 text-left">
              <AlertCircle size={16} className="shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={(e) => { e.preventDefault(); handleVerify(keyInput); }} className="space-y-4">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-gray-400">
                <Key size={16} />
              </div>
              <input
                type="password"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                placeholder="Enter collaborator secret key (VT_ADMIN_KEY)..."
                className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-black"
                required
              />
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 px-1">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={rememberKey}
                  onChange={(e) => setRememberKey(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500 border-gray-300"
                />
                <span>Remember key on this device</span>
              </label>
              <span className="text-[11px] text-gray-400">Found in backend config.py</span>
            </div>

            <button
              type="submit"
              disabled={loading || !keyInput.trim()}
              className="w-full py-2.5 px-4 bg-gray-900 hover:bg-black text-white font-medium text-sm rounded-xl transition flex items-center justify-center gap-2 disabled:opacity-40"
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  <span>Verifying key…</span>
                </>
              ) : (
                <>
                  <Unlock size={16} />
                  <span>Unlock Review Queue</span>
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // ─── 2. UNLOCKED DASHBOARD ──────────────────────────────────────
  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-2xl font-bold text-gray-900">Verdict Review Queue</h2>
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
              <ShieldCheck size={13} />
              Collaborator Mode
            </span>
          </div>
          <p className="text-sm text-gray-500">
            Audit user corrections and enforce verified overrides across the platform.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => loadList(adminKey)}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition shadow-sm disabled:opacity-50"
            title="Refresh submissions"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-red-600 bg-red-50 border border-red-100 rounded-xl hover:bg-red-100 transition"
            title="Lock collaborator session"
          >
            <Lock size={14} />
            Lock Portal
          </button>
        </div>
      </div>

      {/* Global Alerts */}
      {successMessage && (
        <div className="mb-6 p-4 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-sm flex items-center gap-2 shadow-sm">
          <CheckCircle2 size={18} className="text-emerald-600 shrink-0" />
          <span>{successMessage}</span>
        </div>
      )}

      {actionError && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-800 rounded-xl text-sm flex items-center gap-2 shadow-sm">
          <AlertCircle size={18} className="text-red-600 shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {/* Metrics & Filter Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-gray-500">Total Submissions</span>
            <p className="text-2xl font-bold text-gray-900 mt-0.5">{feedbackItems.length}</p>
          </div>
          <span className="w-10 h-10 rounded-xl bg-gray-100 flex items-center justify-center text-gray-600">
            <MessageSquare size={18} />
          </span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-amber-600">Pending Review</span>
            <p className="text-2xl font-bold text-gray-900 mt-0.5">{pendingCount}</p>
          </div>
          <span className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center text-amber-600">
            <AlertCircle size={18} />
          </span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-gray-200 shadow-sm flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-emerald-600">Active Overrides</span>
            <p className="text-2xl font-bold text-gray-900 mt-0.5">{approvedCount}</p>
          </div>
          <span className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-600">
            <CheckCircle2 size={18} />
          </span>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-200 pb-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            filter === 'all' ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          All ({feedbackItems.length})
        </button>
        <button
          onClick={() => setFilter('pending')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            filter === 'pending' ? 'bg-amber-500 text-white' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Pending ({pendingCount})
        </button>
        <button
          onClick={() => setFilter('approved')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
            filter === 'approved' ? 'bg-emerald-600 text-white' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Approved Overrides ({approvedCount})
        </button>
      </div>

      {/* Queue List */}
      {filteredItems.length === 0 ? (
        <div className="bg-white p-12 rounded-2xl border border-gray-200 text-center">
          <p className="text-gray-500 text-sm">No feedback items match this filter.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredItems.map((item, idx) => {
            const isApproving = activeApproveScanId === item.scan_id;
            const isProcessing = actionLoading === item.scan_id;

            return (
              <div
                key={item.scan_id || idx}
                className={`bg-white rounded-2xl border transition shadow-sm ${
                  item.is_approved ? 'border-emerald-200' : 'border-gray-200'
                } p-5`}
              >
                {/* Card Header */}
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2">
                    {item.is_approved ? (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-bold bg-emerald-100 text-emerald-800">
                        <Check size={12} /> Active Override: {item.approved_verdict?.toUpperCase()}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-bold bg-amber-100 text-amber-800">
                        Pending Collaborator Review
                      </span>
                    )}

                    {item.publisher && (
                      <span className="inline-flex items-center gap-1 text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                        <Globe size={12} /> {item.publisher}
                      </span>
                    )}
                  </div>

                  <span className="text-[11px] text-gray-400 font-mono">
                    ID: {item.scan_id?.slice(0, 8)}...
                  </span>
                </div>

                {/* Checked Claim(s) */}
                <div className="mb-3 bg-gray-50 p-3 rounded-xl border border-gray-100">
                  <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider block mb-1">
                    Checked Claim(s)
                  </span>
                  <ul className="list-disc list-inside text-sm font-medium text-gray-900 space-y-1">
                    {item.claims.map((claim, cIdx) => (
                      <li key={cIdx} className="leading-relaxed">"{claim}"</li>
                    ))}
                  </ul>
                </div>

                {/* Verdict Difference */}
                <div className="flex flex-wrap items-center gap-3 mb-3 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="text-gray-500">AI Original:</span>
                    {getVerdictBadge(item.original_verdict)}
                  </div>
                  <span className="text-gray-400 font-bold">→</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-gray-500">User Suggestion:</span>
                    {getVerdictBadge(item.suggested_verdict)}
                  </div>
                </div>

                {/* User Comment */}
                {item.comment && (
                  <p className="text-xs text-gray-600 mb-3 bg-yellow-50/60 border border-yellow-100 p-2.5 rounded-lg">
                    <span className="font-semibold text-gray-700">User Comment: </span>
                    {item.comment}
                  </p>
                )}

                {/* Active Review Source if already approved */}
                {item.is_approved && (
                  <div className="mb-4 bg-emerald-50/60 border border-emerald-100 p-3 rounded-lg text-xs space-y-1">
                    <p className="text-emerald-900">
                      <span className="font-semibold">Review Reason: </span>
                      {item.review_reason}
                    </p>
                    {item.review_source && (
                      <a
                        href={item.review_source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-emerald-700 hover:underline font-medium"
                      >
                        <ExternalLink size={12} />
                        View Verified Evidence Source
                      </a>
                    )}
                  </div>
                )}

                {/* Action Controls */}
                <div className="border-t border-gray-100 pt-3 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    {!item.is_approved ? (
                      <button
                        onClick={() => {
                          setActiveApproveScanId(isApproving ? null : item.scan_id);
                          setSourceUrl('');
                          setReasonText(`Confirmed correction to ${item.suggested_verdict} via wire reporting.`);
                        }}
                        disabled={isProcessing}
                        className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition shadow-sm"
                      >
                        <CheckCircle2 size={14} />
                        {isApproving ? 'Cancel Approval' : 'Allow / Approve'}
                      </button>
                    ) : (
                      <button
                        onClick={() => handleRevoke(item.scan_id)}
                        disabled={isProcessing}
                        className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition"
                      >
                        <Undo2 size={14} />
                        Revoke Approval
                      </button>
                    )}

                    <button
                      onClick={() => handleDismiss(item.scan_id)}
                      disabled={isProcessing}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold bg-gray-100 hover:bg-red-50 hover:text-red-600 text-gray-700 rounded-lg transition"
                      title="Dismiss feedback submission"
                    >
                      <Trash2 size={14} />
                      Dismiss
                    </button>
                  </div>

                  {item.updated_at && (
                    <span className="text-[11px] text-gray-400">
                      Submitted: {item.updated_at}
                    </span>
                  )}
                </div>

                {/* Inline Approval Drawer */}
                {isApproving && (
                  <div className="mt-4 p-4 bg-gray-50 rounded-xl border border-emerald-200 animate-fadeIn">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-gray-800 flex items-center gap-1">
                        <Sparkles size={14} className="text-emerald-600" />
                        Approve Verdict Override
                      </span>
                      <span className="text-[11px] text-gray-500">
                        Will set verdict to: <strong className="text-emerald-700 uppercase">{item.suggested_verdict}</strong>
                      </span>
                    </div>

                    <div className="space-y-3 mb-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-gray-600 mb-1">
                          Verified Evidence URL (Required)
                        </label>
                        <input
                          type="url"
                          value={sourceUrl}
                          onChange={(e) => setSourceUrl(e.target.value)}
                          placeholder="https://www.reuters.com/... or https://snopes.com/..."
                          className="w-full px-3 py-2 bg-white border border-gray-200 rounded-lg text-xs focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-gray-600 mb-1">
                          Review Justification / Reason (Required)
                        </label>
                        <textarea
                          rows={2}
                          value={reasonText}
                          onChange={(e) => setReasonText(e.target.value)}
                          placeholder="Explain why this source justifies the verdict change..."
                          className="w-full px-3 py-2 bg-white border border-gray-200 rounded-lg text-xs focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                        />
                      </div>

                      {/* Presets */}
                      <div>
                        <span className="text-[11px] font-medium text-gray-500 block mb-1">Quick Presets:</span>
                        <div className="flex flex-wrap gap-1.5">
                          {DEFAULT_PRESETS.map((p, pIdx) => (
                            <button
                              key={pIdx}
                              type="button"
                              onClick={() => setReasonText(p.reason)}
                              className="text-[10.5px] px-2 py-0.5 bg-white hover:bg-emerald-50 hover:text-emerald-700 border border-gray-200 rounded text-gray-600 transition"
                            >
                              + {p.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => setActiveApproveScanId(null)}
                        className="px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-200 rounded-lg transition"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => handleApproveSubmit(item.scan_id)}
                        disabled={isProcessing}
                        className="px-4 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition shadow-sm flex items-center gap-1.5"
                      >
                        {isProcessing ? <RefreshCw size={12} className="animate-spin" /> : <Check size={12} />}
                        Confirm & Enforce Override
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
