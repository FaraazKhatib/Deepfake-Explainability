import { useRef, useState, useCallback } from 'react';
import { Upload, X, File as FileIcon } from 'lucide-react';
import { formatBytes } from '../services/api';

export default function GenericFileUploader({ onAnalyze, isAnalyzing, acceptedTypes, acceptString, maxSize, icon: Icon }) {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((selectedFile) => {
    setError(null);
    if (!selectedFile) return;
    
    // Check extension or mime
    const extMatch = selectedFile.name.match(/\.[0-9a-z]+$/i);
    const ext = extMatch ? extMatch[0].toLowerCase() : '';
    
    // Simple validation (can be more robust based on acceptedTypes)
    if (selectedFile.size > maxSize) {
      setError(`File too large (${formatBytes(selectedFile.size)}). Maximum is ${formatBytes(maxSize)}.`);
      return;
    }

    setFile(selectedFile);
  }, [maxSize]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  }, [handleFile]);

  return (
    <div className="w-full space-y-5">
      {!file ? (
        <div
          className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors cursor-pointer ${
            dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptString}
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFile(e.target.files[0]);
              e.target.value = '';
            }}
          />
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
              <Upload size={24} className="text-gray-600" />
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900">Drop your file here</p>
              <p className="text-sm text-gray-500">or click to browse</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-2xl p-6">
          <div className="flex flex-col items-center gap-4 mb-6">
             <FileIcon size={48} className="text-gray-400" />
             <div className="text-center">
               <p className="font-medium text-gray-900 truncate max-w-xs">{file.name}</p>
               <p className="text-sm text-gray-500">{formatBytes(file.size)}</p>
             </div>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => setFile(null)}
              disabled={isAnalyzing}
              className="flex-1 py-2 px-4 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={() => onAnalyze(file)}
              disabled={isAnalyzing}
              className="flex-1 py-2 px-4 bg-black text-white rounded-lg font-medium hover:bg-gray-900 disabled:opacity-50"
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze File'}
            </button>
          </div>
        </div>
      )}
      {error && (
        <p className="text-red-500 text-sm text-center">{error}</p>
      )}
    </div>
  );
}
