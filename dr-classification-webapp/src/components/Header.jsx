import { Brain, Eye } from 'lucide-react'
import { modelConfig } from '../config/models'

export default function Header({ activeModel, onModelChange }) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">DR Classification</h1>
              <p className="text-xs text-gray-600">AI-Powered Diagnosis</p>
            </div>
          </div>

          {/* Model Selection Tabs */}
          <div className="flex items-center space-x-1 bg-gray-100 p-1 rounded-lg">
               {Object.entries(modelConfig).map(([key, config]) => (
              <button
                key={key}
                onClick={() => onModelChange(key)}
                className={`${
                  activeModel === key ? 'tab-active' : 'tab-inactive'
                } relative`}
              >
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4" />
                  <span>{config.shortName}</span>
                     {config.bestFold != null && (
                       <span className="ml-2 text-xs text-gray-500">(fold {config.bestFold})</span>
                     )}
                </div>
                {activeModel === key && (
                  <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-primary-600 rounded-full"></div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  )
}
