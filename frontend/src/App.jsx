import { useState } from 'react';
import { FileText, Image as ImageIcon, Mic, Video, Info, ShieldCheck, Lock } from 'lucide-react';
import TextDetectionPage from './pages/TextDetectionPage';
import ImageDetectionPage from './pages/ImageDetectionPage';
import AudioDetectionPage from './pages/AudioDetectionPage';
import VideoDetectionPage from './pages/VideoDetectionPage';
import FeedbackReviewPage from './pages/FeedbackReviewPage';

const MODES = [
  { id: 'text', label: 'Text Fact Checker', icon: FileText, component: TextDetectionPage },
  { id: 'image', label: 'Image Deepfake', icon: ImageIcon, component: ImageDetectionPage },
  { id: 'audio', label: 'Audio Deepfake', icon: Mic, component: AudioDetectionPage },
  { id: 'video', label: 'Video Deepfake', icon: Video, component: VideoDetectionPage },
  { id: 'feedback', label: 'Review Verdicts', icon: ShieldCheck, component: FeedbackReviewPage, isCollaborator: true }
];

export default function App() {
  const [activeMode, setActiveMode] = useState('text');

  const ActiveComponent = MODES.find(m => m.id === activeMode)?.component || TextDetectionPage;

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-100">
          <h1 className="text-xl font-bold tracking-tight text-gray-900">VeriTrust</h1>
          <p className="text-xs text-gray-500 mt-1">Unified Deepfake Detection</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          {MODES.map((mode, idx) => (
            <div key={mode.id}>
              {mode.isCollaborator && (
                <div className="pt-3 pb-1 border-t border-gray-100 my-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 px-3">
                    Collaborator Tools
                  </span>
                </div>
              )}
              <button
                onClick={() => setActiveMode(mode.id)}
                className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                  activeMode === mode.id
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <mode.icon size={18} />
                  <span>{mode.label}</span>
                </div>
                {mode.isCollaborator && (
                  <Lock size={13} className="text-gray-400" />
                )}
              </button>
            </div>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-100">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <Info size={14} />
            <span>AI results are probabilistic.</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <ActiveComponent />
      </main>
    </div>
  );
}
