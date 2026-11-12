import { useState } from 'react';
import { AlertCircle, Globe, RefreshCw } from 'lucide-react';
import { analyzeText, translateText as apiTranslate } from '../services/api';

const LANG_MAP = {
  en: 'English',
  es: 'Spanish',
  hi: 'Hindi',
  fr: 'French',
  de: 'German',
  ar: 'Arabic',
  'zh-CN': 'Chinese',
  ru: 'Russian',
  pt: 'Portuguese',
  ja: 'Japanese',
  it: 'Italian',
};

async function translateText(text, targetLang = 'en', sourceLang = 'auto') {
  if (!text || !text.trim()) return { text, lang: 'en' };
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text.slice(0, 3000))}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const detected = data?.[2] || 'en';
    const translated = (data?.[0] || []).map((s) => s[0]).filter(Boolean).join('');
    return { text: translated || text, lang: detected };
  } catch (err) {
    console.warn('Translate error:', err);
    return { text, lang: 'auto' };
  }
}

export default function TextDetectionPage() {
  const [text, setText] = useState('');
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [originalResult, setOriginalResult] = useState(null);
  const [error, setError] = useState(null);
  const [translating, setTranslating] = useState(false);
  const [targetLang, setTargetLang] = useState('en');

  const handleTranslateInput = async () => {
    if (!text.trim()) return;
    setTranslating(true);
    try {
      const data = await apiTranslate(text);
      if (data && data.text) {
        setText(data.text);
      }
    } catch (e) {
      setError('Translation failed: ' + e.message);
    } finally {
      setTranslating(false);
    }
  };

  const handleTranslateResults = async (lang) => {
    setTargetLang(lang);
    if (!result) return;
    if (lang === 'en' && originalResult) {
      setResult(JSON.parse(JSON.stringify(originalResult)));
      return;
    }
    setTranslating(true);
    try {
      const base = originalResult || result;
      const copy = JSON.parse(JSON.stringify(base));

      if (copy.summary) {
        const { text: s } = await translateText(copy.summary, lang, 'en');
        copy.summary = s;
      }
      if (copy.explanation) {
        const { text: e } = await translateText(copy.explanation, lang, 'en');
        copy.explanation = e;
      }
      if (copy.claims) {
        for (const c of copy.claims) {
          if (c.claim) {
            const { text: tc } = await translateText(c.claim, lang, 'en');
            c.claim = tc;
          }
          if (c.explanation) {
            const { text: te } = await translateText(c.explanation, lang, 'en');
            c.explanation = te;
          }
        }
      }
      setResult(copy);
    } catch (e) {
      setError('Could not translate results: ' + e.message);
    } finally {
      setTranslating(false);
    }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setStatus('analyzing');
    setError(null);
    try {
      const res = await analyzeText(text);
      setResult(res);
      setOriginalResult(JSON.parse(JSON.stringify(res)));
      setStatus('done');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Text Fact Checker</h2>
      <p className="text-gray-600 mb-8">
        Compare factual claims in any language against verified sources with multilingual cross-referencing.
      </p>

      {error && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 mb-8">
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-semibold text-gray-700 flex items-center gap-1.5">
            <Globe size={16} className="text-indigo-600" />
            Article / Claim (Multilingual Support)
          </label>
          <button
            onClick={handleTranslateInput}
            disabled={translating || !text.trim()}
            className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md font-medium transition disabled:opacity-40"
          >
            {translating ? 'Translating…' : '🌐 Translate to English'}
          </button>
        </div>

        <textarea
          className="w-full h-40 p-4 bg-gray-50 border border-gray-200 rounded-xl mb-4 focus:ring-2 focus:ring-black focus:outline-none"
          placeholder="Paste text, news headline, or article in any language (English, Spanish, Hindi, French, etc.)..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={status === 'analyzing'}
        />

        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-500">
            {text.length > 0 ? `${text.length} characters` : 'Supports 100+ languages with automated translation'}
          </span>
          <button
            onClick={handleAnalyze}
            disabled={status === 'analyzing' || !text.trim()}
            className="px-6 py-2 bg-black text-white rounded-lg font-medium disabled:opacity-50 transition"
          >
            {status === 'analyzing' ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>
      </div>

      {status === 'done' && result && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex flex-wrap justify-between items-center bg-white p-4 rounded-xl border border-gray-200 gap-3">
            <h3 className="text-xl font-bold">Analysis Results</h3>

            {/* Translate results toolbar */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-gray-600 flex items-center gap-1">
                <Globe size={14} /> Translate results:
              </span>
              <select
                value={targetLang}
                onChange={(e) => handleTranslateResults(e.target.value)}
                disabled={translating}
                className="text-xs p-1.5 bg-gray-50 border border-gray-300 rounded-md"
              >
                <option value="en">English</option>
                <option value="es">Español (Spanish)</option>
                <option value="hi">हिन्दी (Hindi)</option>
                <option value="fr">Français (French)</option>
                <option value="de">Deutsch (German)</option>
                <option value="ar">العربية (Arabic)</option>
                <option value="zh-CN">中文 (Chinese)</option>
                <option value="ru">Русский (Russian)</option>
                <option value="pt">Português (Portuguese)</option>
                <option value="ja">日本語 (Japanese)</option>
                <option value="it">Italiano (Italian)</option>
              </select>
            </div>
          </div>

          {(result.was_translated || result.translation?.was_translated) && (
            <div className="p-3 bg-indigo-50 border border-indigo-100 rounded-xl text-xs text-indigo-800 flex items-center gap-2">
              <Globe size={16} className="text-indigo-600 shrink-0" />
              <span>
                <strong>Multilingual Check:</strong> Original text detected as{' '}
                <strong>{result.translation?.source_language_name || LANG_MAP[result.detected_language] || result.detected_language?.toUpperCase() || 'non-English'}</strong> and
                translated to English for cross-referencing with news and reference sources.
              </span>
            </div>
          )}

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
            <div className="mb-4">
              <span className="inline-block px-3 py-1 rounded-full text-sm font-semibold bg-gray-100 text-gray-800">
                {result.verdict}
              </span>
            </div>
            <p className="text-gray-800 font-medium mb-2">{result.summary}</p>
            <p className="text-sm text-gray-600 whitespace-pre-line">{result.explanation}</p>
          </div>

          {result.claims?.map((claim, idx) => (
            <div key={idx} className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
              <p className="font-medium text-lg mb-4">"{claim.claim}"</p>
              <div className="flex items-center gap-2 mb-4">
                <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100">
                  {claim.verdict}
                </span>
                <span className="text-sm text-gray-500">Evidence strength: {claim.confidence}</span>
              </div>
              <p className="text-sm text-gray-600 whitespace-pre-line">{claim.explanation}</p>
              <ul className="mt-3 space-y-2 text-sm">
                {[...(claim.supporting_sources || []), ...(claim.refuting_sources || [])].map((source, i) => (
                  <li key={i}>
                    {/^https?:\/\//i.test(source.url || '') ? (
                      <a href={source.url} target="_blank" rel="noopener noreferrer" className="underline text-indigo-600">
                        {source.title || source.outlet}
                      </a>
                    ) : (
                      <span>{source.title || source.outlet}</span>
                    )}
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
