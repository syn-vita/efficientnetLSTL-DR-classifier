import { useEffect, useRef, useState } from 'react'
import { Activity, FileImage, Upload, X } from 'lucide-react'

export default function ImageUpload({ onImageUpload, onClear, uploadedImage, isLoading }) {
  const [dragOver, setDragOver] = useState(false)
  const [fundusSession, setFundusSession] = useState(null)
  const [modelLoading, setModelLoading] = useState(true)
  const [checkingFundus, setCheckingFundus] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function loadModel() {
      try {
        const session = await ort.InferenceSession.create('/models/fundus_classifier_efficientnet_b3.onnx')
        if (!cancelled) {
          setFundusSession(session)
        }
      } catch (err) {
        console.error('Failed to load fundus ONNX model:', err)
      } finally {
        if (!cancelled) setModelLoading(false)
      }
    }

    loadModel()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!uploadedImage) {
      setPreviewUrl(null)
      return undefined
    }

    const objectUrl = URL.createObjectURL(uploadedImage)
    setPreviewUrl(objectUrl)

    return () => {
      URL.revokeObjectURL(objectUrl)
    }
  }, [uploadedImage])

  const preprocessToTensor = async (file, size = 300) => {
    const url = URL.createObjectURL(file)

    try {
      const img = await new Promise((resolve, reject) => {
        const image = new Image()
        image.onload = () => resolve(image)
        image.onerror = reject
        image.src = url
      })

      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')
      const scale = Math.max(size / img.width, size / img.height)
      const x = (size - img.width * scale) / 2
      const y = (size - img.height * scale) / 2
      ctx.drawImage(img, x, y, img.width * scale, img.height * scale)

      const { data } = ctx.getImageData(0, 0, size, size)
      const floatData = new Float32Array(3 * size * size)
      const mean = [0.485, 0.456, 0.406]
      const std = [0.229, 0.224, 0.225]

      let idx = 0
      for (let i = 0; i < size * size; i++) {
        const r = data[idx] / 255
        const g = data[idx + 1] / 255
        const b = data[idx + 2] / 255
        floatData[i] = (r - mean[0]) / std[0]
        floatData[i + size * size] = (g - mean[1]) / std[1]
        floatData[i + 2 * size * size] = (b - mean[2]) / std[2]
        idx += 4
      }

      return new ort.Tensor('float32', floatData, [1, 3, size, size])
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x))

  const isFundusImage = async (file) => {
    if (!fundusSession) {
      throw new Error('Fundus model is not loaded yet')
    }

    const inputTensor = await preprocessToTensor(file, 300)
    const feeds = { input: inputTensor }
    const results = await fundusSession.run(feeds)
    const logit = results.logit.data[0]
    const pNonFundus = sigmoid(logit)
    return pNonFundus < 0.5
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleFileSelect = async (file) => {
    if (!file || !file.type?.startsWith('image/')) {
      alert('Please select a valid image file.')
      return
    }

    if (modelLoading) {
      alert('Model is loading, please wait a moment and try again...')
      return
    }

    if (!fundusSession) {
      alert('Fundus check is unavailable right now. Please refresh the page and try again.')
      return
    }

    try {
      setCheckingFundus(true)
      const ok = await isFundusImage(file)
      if (!ok) {
        alert('The uploaded image does not appear to be a retinal fundus image. Please upload a valid fundus image.')
        return
      }
      onImageUpload(file)
    } catch (err) {
      console.error('Fundus check failed:', err)
      alert('Unable to verify the image right now. Please try again.')
    } finally {
      setCheckingFundus(false)
    }
  }

  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleClick = () => {
    if (!isLoading && !modelLoading && !checkingFundus) {
      fileInputRef.current?.click()
    }
  }

  const isBusy = isLoading || modelLoading || checkingFundus

  return (
    <div className="space-y-4">
      {!uploadedImage ? (
        <div className="upload-panel-shell">
          <div
            className={`upload-panel ${dragOver ? 'upload-panel-active' : ''} ${isBusy ? 'upload-panel-busy' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={handleClick}
          >
            <div className="upload-panel__icon">
              <Upload className="h-8 w-8" />
            </div>
            <h3 className="upload-panel__title">{dragOver ? 'Drop your image here' : 'Upload retinal fundus image'}</h3>
            <p className="upload-panel__copy">
              {modelLoading
                ? 'Preparing the fundus-image checker before analysis.'
                : 'Drag and drop a retinal image here, or click to browse from your device.'}
            </p>
            <p className="upload-panel__hint">Supports JPG, PNG, JPEG • Max size 10MB</p>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
              disabled={isBusy}
            />
          </div>
        </div>
      ) : (
        <div className="upload-preview-card">
          <div className="upload-preview-meta">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700">
                <FileImage className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-950">{uploadedImage.name}</p>
                <p className="mt-1 text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                  {(uploadedImage.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            {!isLoading && (
              <button
                onClick={() => onClear?.()}
                className="flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-500 transition-colors hover:text-slate-900"
                aria-label="Clear uploaded image"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="relative overflow-hidden bg-slate-100">
            <img
              src={previewUrl}
              alt="Uploaded retinal image"
              className="h-72 w-full object-cover md:h-80"
            />
            {(isLoading || checkingFundus) && (
              <div className="upload-overlay">
                <div className="upload-overlay-card">
                  <Activity className="h-5 w-5 animate-spin text-cyan-700" />
                  <span>{checkingFundus ? 'Verifying fundus image...' : 'Analyzing image...'}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
