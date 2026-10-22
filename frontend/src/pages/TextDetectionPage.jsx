import { useState } from 'react';
import { AlertCircle, FileText, CheckCircle, XCircle } from 'lucide-react';
import { analyzeText } from '../services/api';

export default function TextDetectionPage() {
  const [text, setText] = useState('');
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setStatus('analyzing');
    setError(null);
    try {
      const res = await analyzeText(text);
      setResult(res);
      setStatus('done');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Text Fact Checker</h2>
      <p className="text-gray-600 mb-8">Compare factual claims with available sources. This does not determine whether text was written by AI.</p>

      {error && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 mb-8">
        <textarea
          className="w-full h-40 p-4 bg-gray-50 border border-gray-200 rounded-xl mb-4 focus:ring-2 focus:ring-black focus:outline-none"
          placeholder="Paste text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={status === 'analyzing'}
        />
        <div className="flex justify-end">
          <button
            onClick={handleAnalyze}
            disabled={status === 'analyzing' || !text.trim()}
            className="px-6 py-2 bg-black text-white rounded-lg font-medium disabled:opacity-50"
          >
            {status === 'analyzing' ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>
      </div>

      {status === 'done' && result && (
        <div className="space-y-6 animate-fade-in">
          <h3 className="text-xl font-bold">Results</h3>
          <p className="text-gray-700">{result.summary}</p>
          <p className="text-sm text-gray-600 whitespace-pre-line">{result.explanation}</p>
          {result.claims?.map((claim, idx) => (
            <div key={idx} className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
              <p className="font-medium text-lg mb-4">"{claim.claim}"</p>
              <div className="flex items-center gap-2 mb-4">
                <span className="px-3 py-1 rounded-full text-sm font-medium">
                  {claim.verdict}
                </span>
                <span className="text-sm text-gray-500">Evidence strength: {claim.confidence}</span>
              </div>
              <p className="text-sm text-gray-600 whitespace-pre-line">{claim.explanation}</p>
              <ul className="mt-3 space-y-2 text-sm">
                {[...(claim.supporting_sources || []), ...(claim.refuting_sources || [])].map((source, i) => (
                  <li key={i}>
                    {/^https?:\/\//i.test(source.url || '') ? <a href={source.url} target="_blank" rel="noopener noreferrer" className="underline">{source.title || source.outlet}</a> : <span>{source.title || source.outlet}</span>}
                    <span className="text-gray-500"> — {source.stance}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
