import { AlertTriangle, CheckCircle, Eye, BarChart3 } from 'lucide-react'
import { drClasses } from '../config/models'

export default function ResultDisplay({ prediction, isLoading, modelName, error }) {
  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Analyzing Image</h3>
            <p className="text-sm text-gray-600">
              {modelName} is processing your retinal image...
            </p>
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
                <AlertTriangle className="w-12 h-12 text-yellow-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Analysis Error</h3>
                <p className="text-sm text-gray-600 max-w-md mx-auto">{error}</p>
              </>
            ) : (
              <>
                <Eye className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
                <p className="text-sm text-gray-600">
                  Upload a retinal image to get started with the analysis.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  const predictedClass = drClasses[prediction.class]
  const confidence = (prediction.confidence * 100).toFixed(1)

  const getResultIcon = () => {
    if (prediction.class === 0) {
      return <CheckCircle className="w-6 h-6 text-success-600" />
    }
    return <AlertTriangle className="w-6 h-6 text-warning-600" />
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

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Classification Result</h2>
          <span className="text-sm text-gray-500">by {modelName}</span>
        </div>

        <div className="text-center mb-6">
          <div className="flex items-center justify-center mb-4">
            {getResultIcon()}
          </div>
          <div className={`${getResultBadgeClass()} text-lg font-semibold mb-2`}>
            {predictedClass.name}
          </div>
          <p className="text-sm text-gray-600 mb-4">
            {predictedClass.description}
          </p>
          <div className="text-2xl font-bold text-gray-900">
            {confidence}% confidence
          </div>
        </div>

        {/* Severity Indicator */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Severity Level</span>
            <span className="text-sm text-gray-600">{prediction.class}/4</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                prediction.class === 0 ? 'bg-success-500' :
                prediction.class === 1 ? 'bg-warning-500' :
                prediction.class === 2 ? 'bg-orange-500' :
                prediction.class === 3 ? 'bg-danger-500' : 'bg-red-600'
              }`}
              style={{ width: `${((prediction.class + 1) / 5) * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Recommendations */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-2">
            Clinical Recommendation
          </h4>
          <p className="text-sm text-blue-800">
            {predictedClass.recommendation}
          </p>
        </div>
      </div>

      {/* Detailed Probabilities */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Class Probabilities</h3>
        </div>
        
        <div className="space-y-3">
          {drClasses.map((drClass, index) => {
            const probability = (prediction.probabilities[index] * 100).toFixed(1)
            const isHighest = index === prediction.class
            
            return (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-3 flex-1">
                  <span className={`text-sm font-medium ${isHighest ? 'text-gray-900' : 'text-gray-600'}`}>
                    {drClass.name}
                  </span>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${isHighest ? 'bg-primary-600' : 'bg-gray-400'}`}
                      style={{ width: `${probability}%` }}
                    ></div>
                  </div>
                </div>
                <span className={`text-sm font-medium ml-3 ${isHighest ? 'text-gray-900' : 'text-gray-600'}`}>
                  {probability}%
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-sm font-medium text-yellow-900 mb-1">
              Important Disclaimer
            </h4>
            <p className="text-sm text-yellow-800">
              This tool is for research and educational purposes only. Results should not be used as the sole basis for medical decisions. Always consult with qualified healthcare professionals for proper diagnosis and treatment.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
