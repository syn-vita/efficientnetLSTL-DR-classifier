import { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import ResultDisplay from './components/ResultDisplay'
import LandingPage from './components/LandingPage'
import { modelConfig } from './config/models'

function App() {
  const [currentView, setCurrentView] = useState('landing') // 'landing' or 'testing'
  const [activeModel, setActiveModel] = useState('baseline')
  const [uploadedImage, setUploadedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastError, setLastError] = useState(null)

  const handleStartTesting = () => {
    setCurrentView('testing')
  }

  const handleBackToLanding = () => {
    setCurrentView('landing')
    // Reset all state when going back to landing
    setUploadedImage(null)
    setPrediction(null)
    setIsLoading(false)
    setLastError(null)
  }

  const handleImageUpload = async (imageFile) => {
    // Handle clear action or empty input
    if (!imageFile) {
      setUploadedImage(null)
      setPrediction(null)
      setIsLoading(false)
      setLastError(null)
      return
    }

    setUploadedImage(imageFile)
    setIsLoading(true)
    setPrediction(null)
    setLastError(null)

    try {
      const cfg = modelConfig[activeModel]
      if (cfg.modelPath && cfg.modelPath.endsWith('.onnx')) {
        const { runOnnx } = await import('./utils/inference')
        const arrBuf = await imageFile.arrayBuffer()
        const blob = new Blob([arrBuf])
        // Prefer ImageBitmap for performance, but fall back to HTMLImageElement if unavailable
        let imgSource
        try {
          imgSource = await createImageBitmap(blob)
        } catch {
          imgSource = await new Promise((resolve, reject) => {
            const img = new Image()
            img.onload = () => resolve(img)
            img.onerror = reject
            img.src = URL.createObjectURL(blob)
          })
        }
        const result = await runOnnx(cfg.modelPath, imgSource)
        const probs = result.probabilities
        const top = result.classIndex
        setPrediction({
          class: top,
          confidence: probs[top],
          probabilities: probs
        })
      } else {
        // Fallback mock prediction
        await new Promise(resolve => setTimeout(resolve, 1200))
        const mockResult = {
          class: Math.floor(Math.random() * 5),
          confidence: 0.85 + Math.random() * 0.14,
          probabilities: [
            Math.random() * 0.3,
            Math.random() * 0.2,
            Math.random() * 0.25,
            Math.random() * 0.15,
            Math.random() * 0.1
          ]
        }
        setPrediction(mockResult)
      }
    } catch (error) {
      console.error('Prediction error:', error)
      setLastError(error?.message || 'Prediction failed. Please try a different image or model.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearResults = () => {
    setUploadedImage(null)
    setPrediction(null)
  }

  // Show landing page if currentView is 'landing'
  if (currentView === 'landing') {
    return <LandingPage onStartTesting={handleStartTesting} />
  }

  // Show testing interface
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        activeModel={activeModel}
        onModelChange={setActiveModel}
        onBackToLanding={handleBackToLanding}
        showBackButton={true}
      />
      
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload and Controls */}
          <div className="space-y-6">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Upload Retinal Image
              </h2>
              <p className="text-sm text-gray-600 mb-6">
                Upload a retinal fundus image for diabetic retinopathy analysis using the {modelConfig[activeModel].name} model.
              </p>
              
              <ImageUpload 
                onImageUpload={handleImageUpload}
                onClear={handleClearResults}
                uploadedImage={uploadedImage}
                isLoading={isLoading}
              />
              
              {uploadedImage && (
                <div className="mt-4 flex gap-3">
                  <button
                    onClick={handleClearResults}
                    className="btn-secondary"
                    disabled={isLoading}
                  >
                    Clear Results
                  </button>
                  <button
                    onClick={() => handleImageUpload(uploadedImage)}
                    className="btn-primary"
                    disabled={isLoading}
                  >
                    {isLoading ? 'Analyzing...' : 'Re-analyze'}
                  </button>
                </div>
              )}
            </div>

            {/* Model Information */}
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                Current Model: {modelConfig[activeModel].name}
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                {modelConfig[activeModel].description}
              </p>
              {/* <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Architecture:</span>
                  <span className="font-medium">{modelConfig[activeModel].architecture}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Parameters:</span>
                  <span className="font-medium">{modelConfig[activeModel].parameters}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-medium">{modelConfig[activeModel].accuracy}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">FLOPs (224x224):</span>
                  <span className="font-medium">{modelConfig[activeModel].gflops?.toFixed ? modelConfig[activeModel].gflops.toFixed(3) : modelConfig[activeModel].gflops} GFLOPs</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Accuracy shown is balanced accuracy on the validation set at the best epoch for the selected fold.
              </p> */}
            </div>
          </div>

          {/* Right Column - Results */}
          <div>
            <ResultDisplay 
              prediction={prediction}
              isLoading={isLoading}
              modelName={modelConfig[activeModel].name}
              error={lastError}
            />
          </div>
        </div>
      </main>
      
      <footer className="border-t border-gray-200 bg-white mt-16">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <p className="text-center text-sm text-gray-600">
            DR Classification Tool • For Research and Educational Use Only
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
