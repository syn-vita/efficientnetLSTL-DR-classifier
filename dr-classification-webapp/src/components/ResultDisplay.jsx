import { AlertTriangle, CheckCircle, Eye, BarChart3, Activity } from 'lucide-react'
import { drClasses } from '../config/models'

export default function ResultDisplay({ prediction, isLoading, modelName, error }) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="relative">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600 mx-auto mb-4"></div>
              <Activity className="w-6 h-6 text-blue-600 absolute top-3 left-3" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Analyzing Image</h3>
            <p className="text-sm text-gray-600">
              {modelName} is processing your retinal image...
            </p>
            <div className="mt-4 bg-blue-50 rounded-lg p-3">
              <p className="text-xs text-blue-700">
                Deep learning model extracting features and patterns
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            {error ? (
              <>
                <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <AlertTriangle className="w-8 h-8 text-red-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Analysis Error</h3>
                <p className="text-sm text-gray-600 max-w-md mx-auto mb-4">{error}</p>
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-xs text-red-700">
                    Please try uploading a different image or selecting another model
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Eye className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Analysis</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Upload a retinal image to get started with AI-powered analysis.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-blue-700">
                    Supported formats: JPG, PNG, JPEG • Recommended: High resolution fundus images
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  const predictedClass = drClasses[prediction.class]

  const getResultIcon = () => {
    if (prediction.class === 0) {
      return <CheckCircle className="w-8 h-8 text-green-600" />
    }
    return <AlertTriangle className="w-8 h-8 text-yellow-600" />
  }

  const getResultBadgeClass = () => {
    switch (prediction.class) {
      case 0: return 'result-no-dr'
      case 1: return 'result-mild'
      case 2: return 'result-moderate'
      case 3: return 'result-severe'
      case 4: return 'result-proliferative'
      default: return 'result-no-dr'
    }
  }

  const getSeverityColor = () => {
    switch (prediction.class) {
      case 0: return 'bg-green-500'
      case 1: return 'bg-yellow-500'
      case 2: return 'bg-orange-500'
      case 3: return 'bg-red-500'
      case 4: return 'bg-red-600'
      default: return 'bg-green-500'
    }
  }

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <div className="card transform transition-all duration-300 hover:scale-[1.02]">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <span>Classification Result</span>
          </h2>
          <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
            {modelName}
          </span>
        </div>

        <div className="text-center mb-6">
          <div className="flex items-center justify-center mb-4">
            <div className="relative">
              {getResultIcon()}
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-blue-600 rounded-full flex items-center justify-center">
                <div className="w-2 h-2 bg-white rounded-full"></div>
              </div>
            </div>
          </div>
          <div className={`${getResultBadgeClass()} text-lg font-semibold mb-3 transform transition-all duration-300 hover:scale-105`}>
            {predictedClass.name}
          </div>
          <p className="text-sm text-gray-600 mb-4 leading-relaxed">
            {predictedClass.description}
          </p>
        </div>

        {/* Severity Indicator */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-gray-700">Severity Assessment</span>
            <span className="text-sm text-gray-600 bg-white px-2 py-1 rounded">
              Grade {prediction.class}/4
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div 
              className={`h-3 rounded-full ${getSeverityColor()} transition-all duration-1000 ease-out`}
              style={{ width: `${((prediction.class + 1) / 5) * 100}%` }}
            ></div>
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>No DR</span>
            <span>Mild</span>
            <span>Moderate</span>
            <span>Severe</span>
            <span>PDR</span>
          </div>
        </div>

        {/* Recommendations */}
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100">
          <h4 className="text-sm font-medium text-blue-900 mb-2 flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            <span>Clinical Recommendation</span>
          </h4>
          <p className="text-sm text-blue-800 leading-relaxed">
            {predictedClass.recommendation}
          </p>
        </div>
      </div>

      {/* Detailed Probabilities */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Probability Distribution</h3>
        </div>
        
        <div className="space-y-4">
          {drClasses.map((drClass, index) => {
            const probability = (prediction.probabilities[index] * 100).toFixed(1)
            const isHighest = index === prediction.class
            
            return (
              <div key={index} className={`transition-all duration-300 ${isHighest ? 'transform scale-105' : ''}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${isHighest ? 'text-gray-900' : 'text-gray-600'}`}>
                    {drClass.name}
                  </span>
                  <span className={`text-sm font-bold ${isHighest ? 'text-blue-600' : 'text-gray-600'}`}>
                    {probability}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div 
                    className={`h-2.5 rounded-full transition-all duration-1000 ease-out ${
                      isHighest ? 'bg-gradient-to-r from-blue-500 to-indigo-600' : 'bg-gray-400'
                    }`}
                    style={{ width: `${Math.max(probability, 2)}%` }}
                  ></div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <div className="w-6 h-6 bg-yellow-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
            <AlertTriangle className="w-4 h-4 text-yellow-600" />
          </div>
          <div>
            <h4 className="text-sm font-medium text-yellow-900 mb-2">
              Medical Disclaimer
            </h4>
            <p className="text-sm text-yellow-800 leading-relaxed">
              This AI tool is designed for research and educational purposes only. Results should not be used as the sole basis for medical decisions. Always consult with qualified ophthalmologists for proper diagnosis and treatment of diabetic retinopathy.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
