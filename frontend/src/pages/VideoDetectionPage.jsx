import { useState } from 'react';
import { AlertCircle, Video } from 'lucide-react';
import { analyzeVideo } from '../services/api';
import GenericFileUploader from '../components/GenericFileUploader';

export default function VideoDetectionPage() {
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async (file) => {
    setStatus('analyzing');
    setError(null);
    try {
      const res = await analyzeVideo(file);
      setResult(res);
      setStatus('done');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Video Deepfake Detection</h2>
      <p className="text-gray-600 mb-8">Upload a video to detect AI-generated faces or manipulations.</p>

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
          acceptString="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
          maxSize={200 * 1024 * 1024}
          icon={Video}
        />
      )}

      {status === 'analyzing' && (
        <div className="text-center py-12">
          <div className="w-12 h-12 border-4 border-gray-200 border-t-black rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 font-medium">Analyzing Video Frames...</p>
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
                {result.classification === 'UNCERTAIN' ? 'Inconclusive — please review' : result.classification === 'FAKE' ? 'Signs of manipulation' : 'Few signs of manipulation'}
              </span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Signal strength</p>
                <p className="text-lg font-semibold text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Manipulation score</p>
                <p className="text-lg font-semibold text-gray-900">{(result.fake_probability * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Frames</p>
                <p className="text-lg font-semibold text-gray-900">{result.frames_analyzed}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-xl text-center">
                <p className="text-xs text-gray-500 uppercase font-medium mb-1">Faces Found</p>
                <p className="text-lg font-semibold text-gray-900">{result.faces_detected}</p>
              </div>
            </div>

            <p className="text-sm text-gray-600 mb-4">These scores are estimates, not the chance that the verdict is correct. Only sampled frames are checked; the soundtrack is not analyzed.</p>
            {result.warning && <p role="status" className="p-4 mb-4 bg-amber-50 text-amber-900 rounded-lg">{result.warning}</p>}
            {result.disclaimer && <p className="text-sm text-gray-500 mb-4">{result.disclaimer}</p>}
            {result.suspicious_frames && result.suspicious_frames.length > 0 && (
              <div>
                <h4 className="text-lg font-semibold mb-4">Suspicious Frames</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {result.suspicious_frames.map((frame, idx) => (
                    <div key={idx} className="rounded-lg overflow-hidden border border-gray-200">
                      {frame.thumbnail && (
                        <img src={`data:image/jpeg;base64,${frame.thumbnail}`} alt={`Frame ${idx + 1}`} className="w-full h-auto" />
                      )}
                      <div className="p-2 bg-gray-50 text-center text-xs text-gray-600">
                        Time: {frame.timestamp.toFixed(2)}s
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
