import { Brain, Eye, ArrowLeft } from 'lucide-react'
import { modelConfig } from '../config/models'

export default function Header({ activeModel, onModelChange, onBackToLanding, showBackButton = false }) {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            {showBackButton && (
              <button
                onClick={onBackToLanding}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200 mr-2"
                title="Back to Landing"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
            )}
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg shadow-md">
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
                } relative transition-all duration-200`}
              >
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4" />
                  <span>{config.shortName}</span>
                     {config.bestFold != null && (
                       <span className="ml-2 text-xs text-gray-500">(fold {config.bestFold})</span>
                     )}
                </div>
                {activeModel === key && (
                  <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-blue-600 rounded-full"></div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  )
}
