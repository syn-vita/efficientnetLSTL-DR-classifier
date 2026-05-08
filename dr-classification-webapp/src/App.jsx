import { useState } from 'react'
import { Brain, Microscope, ShieldCheck } from 'lucide-react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import ResultDisplay from './components/ResultDisplay'
import LandingPage from './components/LandingPage'
import { appConfig, modelConfig } from './config/models'

function App() {
  const [currentView, setCurrentView] = useState('landing')
  const [activeModel, setActiveModel] = useState('lstl')
  const [uploadedImage, setUploadedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [lastError, setLastError] = useState(null)

  const modelEntries = Object.entries(modelConfig)
  const activeModelConfig = modelConfig[activeModel]

  const handleStartTesting = () => {
    setCurrentView('testing')
  }

  const handleBackToLanding = () => {
    setCurrentView('landing')
    setUploadedImage(null)
    setPrediction(null)
    setIsLoading(false)
    setLastError(null)
  }

  const handleImageUpload = async (imageFile) => {
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
        await new Promise((resolve) => setTimeout(resolve, 1200))
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

  if (currentView === 'landing') {
    return <LandingPage onStartTesting={handleStartTesting} />
  }

  return (
    <div className="app-shell">
      <Header onBackToLanding={handleBackToLanding} showBackButton />

      <main className="mx-auto max-w-7xl px-4 py-8 lg:py-10">
        <section className="surface-card mb-6 overflow-hidden p-0">
          <div className="grid gap-0 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="border-b border-slate-200/80 px-6 py-6 lg:border-b-0 lg:border-r">
              <span className="eyebrow">Testing workspace</span>
              <h2 className="mt-4 text-2xl font-semibold tracking-tight text-slate-950 md:text-3xl">
                Model comparison and retinal image analysis
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-600">
                Select a model, upload a retinal fundus image, and review the resulting
                classification output and severity assessment.
              </p>

              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <div className="rounded-3xl border border-teal-100 bg-teal-50/70 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-teal-700 shadow-sm">
                    <Brain className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm font-medium text-slate-500">Primary model</p>
                  <p className="mt-2 font-semibold text-slate-950">EfficientNet-B0 + LSTL</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/80 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
                    <Microscope className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm font-medium text-slate-500">Use context</p>
                  <p className="mt-2 font-semibold text-slate-950">Research and educational analysis</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/80 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm font-medium text-slate-500">Workflow</p>
                  <p className="mt-2 font-semibold text-slate-950">Upload, analyze, then compare</p>
                </div>
              </div>
            </div>

            <div className="bg-slate-50/80 px-6 py-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                Available variants
              </p>
              <div className="mt-5 space-y-3">
                {modelEntries.map(([key, config]) => {
                  const isActive = activeModel === key

                  return (
                    <button
                      key={key}
                      onClick={() => onModelChangeHelper(key, setActiveModel)}
                      className={`w-full rounded-[26px] border px-5 py-4 text-left transition-all duration-200 ${
                        isActive
                          ? 'border-2 border-teal-500 bg-teal-50/70 shadow-[0_0_0_4px_rgba(16,152,173,0.12)]'
                          : 'border-slate-200 bg-white/75 hover:border-slate-300 hover:bg-white'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-lg font-semibold text-slate-950">{config.shortName}</span>
                            <span
                              className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
                                config.badge === 'Recommended'
                                  ? 'bg-teal-100 text-teal-800'
                                  : 'bg-slate-100 text-slate-600'
                              }`}
                            >
                              {config.badge}
                            </span>
                          </div>
                          <p className="mt-2 text-sm leading-6 text-slate-600">{config.description}</p>
                        </div>
                        <span className="shrink-0 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500">
                          fold {config.bestFold}
                        </span>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="space-y-6">
            <div className="surface-card">
              <div className="mb-6">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                  Upload workflow
                </p>
                <h2 className="mt-2 text-2xl font-semibold text-slate-950">Upload retinal fundus image</h2>
                <p className="mt-3 text-sm leading-7 text-slate-600">
                  Upload a retinal fundus image for diabetic retinopathy analysis using the{' '}
                  {activeModelConfig.name} model.
                </p>
              </div>

              <ImageUpload
                onImageUpload={handleImageUpload}
                onClear={handleClearResults}
                uploadedImage={uploadedImage}
                isLoading={isLoading}
              />

              {uploadedImage && (
                <div className="mt-5 flex flex-wrap gap-3">
                  <button onClick={handleClearResults} className="btn-secondary" disabled={isLoading}>
                    Clear Results
                  </button>
                  <button onClick={() => handleImageUpload(uploadedImage)} className="btn-primary" disabled={isLoading}>
                    {isLoading ? 'Analyzing...' : 'Re-analyze'}
                  </button>
                </div>
              )}
            </div>

            <div className="surface-card">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                Active model details
              </p>
              <h3 className="mt-2 text-2xl font-semibold text-slate-950">{activeModelConfig.name}</h3>
              <p className="mt-3 text-sm leading-7 text-slate-600">{activeModelConfig.description}</p>

              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <div className="rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Architecture</p>
                  <p className="mt-2 font-semibold text-slate-950">{activeModelConfig.architecture}</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Best fold</p>
                  <p className="mt-2 font-semibold text-slate-950">Fold {activeModelConfig.bestFold}</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Parameters</p>
                  <p className="mt-2 font-semibold text-slate-950">{activeModelConfig.parameters}</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Balanced accuracy</p>
                  <p className="mt-2 font-semibold text-slate-950">{activeModelConfig.accuracy}</p>
                </div>
              </div>
            </div>
          </div>

          <div>
            <ResultDisplay
              prediction={prediction}
              isLoading={isLoading}
              modelName={activeModelConfig.name}
              error={lastError}
            />
          </div>
        </section>
      </main>

      <footer className="mt-16 border-t border-slate-200/80 bg-white/70 px-4 py-6 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl">
          <p className="text-center text-sm text-slate-600">
            {appConfig.compactTitle} • For research and educational use only
          </p>
        </div>
      </footer>
    </div>
  )
}

function onModelChangeHelper(modelKey, setActiveModel) {
  setActiveModel(modelKey)
}

export default App
