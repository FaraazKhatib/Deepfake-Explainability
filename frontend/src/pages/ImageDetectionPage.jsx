import { useState } from 'react';
import { AlertCircle, ImageIcon } from 'lucide-react';
import { analyzeImage } from '../services/api';
import GenericFileUploader from '../components/GenericFileUploader';

export default function ImageDetectionPage() {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async (file) => {
    setStatus('analyzing');
    setError(null);
    try {
      const res = await analyzeImage(file);
      setResult(res);
      setStatus('done');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Image Deepfake Detection</h2>
      <p className="text-gray-600 mb-8">Check an image for signs of AI generation. Small edits and face swaps can be missed.</p>

      {error && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg flex items-start gap-3">
          <AlertCircle size={20} className="shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      {(status === 'idle' || status === 'error') && (
        <GenericFileUploader
          onAnalyze={handleAnalyze}
          isAnalyzing={false}
          acceptString="image/jpeg,image/png,image/webp"
          maxSize={30 * 1024 * 1024}
          icon={ImageIcon}
        />
      )}

      {status === 'analyzing' && (
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-gray-200 border-t-black rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 font-medium">Analyzing Image...</p>
        </div>
      )}

      {status === 'done' && result && (
        <div className="space-y-6 animate-fade-in">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-bold">Analysis Complete</h3>
            <button onClick={() => { setStatus('idle'); setResult(null); }} className="text-sm font-medium text-gray-500 hover:text-black">
              Start Over
            </button>
          </div>
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
            <div className="text-center mb-6">
              <span className="inline-block px-4 py-2 rounded-full text-lg font-bold">
                Verdict: {result.verdict?.replaceAll('_', ' ')}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Model AI score</p>
                <p className="text-lg font-semibold text-gray-900">{result.score_ai < 0.001 ? '<0.1' : (result.score_ai * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Signal strength</p>
                <p className="text-lg font-semibold text-gray-900">{result.confidence.toFixed(1)}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Calibration</p>
                <p className="text-lg font-semibold text-gray-900">{result.calibrated ? 'Ready' : 'Uncalibrated'}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Image Size</p>
                <p className="text-lg font-semibold text-gray-900">{result.image?.width} x {result.image?.height}</p>
              </div>
            </div>

            {result.signals?.length > 0 && (
              <div className="mb-6">
                <h4 className="text-lg font-semibold mb-3">Detector Signals</h4>
                <div className="space-y-2">
                  {result.signals.map((signal) => (
                    <div key={signal.name} className="flex justify-between items-center bg-gray-50 p-3 rounded-lg">
                      <span className="text-sm font-medium text-gray-700">{signal.name}</span>
                      <span className="text-sm font-semibold text-gray-900">{(signal.p_ai * 100).toFixed(1)}% AI</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p className="text-sm text-gray-600 mb-4">This score is a model estimate, not measured accuracy. A low score means few familiar AI patterns were found; it does not prove the image is real. An inconclusive result means there is not enough reliable evidence.</p>
            {result.errors?.length > 0 && <p className="text-sm text-amber-800 mb-4">Some detectors could not complete their checks. Treat this result with caution.</p>}
            {result.notes?.length > 0 && (
              <div>
                <h4 className="text-lg font-semibold mb-3">Analysis Notes</h4>
                <ul className="space-y-2 text-sm text-gray-600 list-disc list-inside">
                  {result.notes.map((note, index) => <li key={index}>{note}</li>)}
                </ul>
              </div>
            )}
            </div>
          </div>
      )}
    </div>
  );
}
