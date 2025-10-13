import { useState, useRef, useEffect } from 'react'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

export default function ImageUpload({ onImageUpload, onClear, uploadedImage, isLoading }) {
  const [dragOver, setDragOver] = useState(false)
  const [fundusSession, setFundusSession] = useState(null)
  const [modelLoading, setModelLoading] = useState(true)
  const [checkingFundus, setCheckingFundus] = useState(false)
  const fileInputRef = useRef(null)

  // Load the fundus classifier ONNX model once from public/models
  useEffect(() => {
    let cancelled = false
    async function loadModel() {
      try {
        // 'ort' is provided by index.html script tag at /ort/ort.min.js
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

  // Preprocess image to 300x300 NCHW Float32 with ImageNet normalization
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
      // Cover fit to preserve aspect ratio, center-crop style
      const scale = Math.max(size / img.width, size / img.height)
      const x = (size - img.width * scale) / 2
      const y = (size - img.height * scale) / 2
      ctx.drawImage(img, x, y, img.width * scale, img.height * scale)

      const { data } = ctx.getImageData(0, 0, size, size)
      const floatData = new Float32Array(3 * size * size)
      const mean = [0.485, 0.456, 0.406]
      const std = [0.229, 0.224, 0.225]
      // Convert HWC RGBA -> CHW, normalize
      let idx = 0
      for (let i = 0; i < size * size; i++) {
        const r = data[idx] / 255
        const g = data[idx + 1] / 255
        const b = data[idx + 2] / 255
        // ignore alpha channel
        floatData[i] = (r - mean[0]) / std[0]                    // R channel
        floatData[i + size * size] = (g - mean[1]) / std[1]      // G channel
        floatData[i + 2 * size * size] = (b - mean[2]) / std[2]  // B channel
        idx += 4
      }

      const tensor = new ort.Tensor('float32', floatData, [1, 3, size, size])
      return tensor
    } finally {
      URL.revokeObjectURL(url)
    }
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x))

  // Returns true if image is fundus, false otherwise
  const isFundusImage = async (file) => {
    if (!fundusSession) {
      throw new Error('Fundus model is not loaded yet')
    }
    const inputTensor = await preprocessToTensor(file, 300)
    const feeds = { input: inputTensor } // matches export input name
    const results = await fundusSession.run(feeds)
    const logit = results.logit.data[0]
    const p = sigmoid(logit)
    // Note: training labels used 1 = non-fundus, 0 = fundus, so p approximates P(non-fundus)
    const pNonFundus = p
    const threshold = 0.5
    return pNonFundus < threshold
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

  const getImagePreviewUrl = () => {
    if (uploadedImage) {
      return URL.createObjectURL(uploadedImage)
    }
    return null
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      {!uploadedImage ? (
        <div
          className={`upload-area ${dragOver ? 'dragover' : ''} ${(isLoading || modelLoading || checkingFundus) ? 'opacity-50 cursor-not-allowed' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={handleClick}
        >
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {dragOver ? 'Drop your image here' : 'Upload retinal image'}
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            {modelLoading ? 'Loading fundus checker...' : 'Drag and drop your fundus image here, or click to browse'}
          </p>
          <p className="text-xs text-gray-500">
            Supports: JPG, PNG, JPEG • Max size: 10MB
          </p>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="hidden"
            disabled={isLoading || modelLoading || checkingFundus}
          />
        </div>
      ) : (
        /* Image Preview */
        <div className="relative">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                <ImageIcon className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    {uploadedImage.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(uploadedImage.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
        {!isLoading && (
                <button
          onClick={() => onClear?.()}
                  className="p-1 hover:bg-gray-100 rounded"
                >
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              )}
            </div>
            
            <div className="relative overflow-hidden rounded-lg bg-gray-100">
              <img
                src={getImagePreviewUrl()}
                alt="Uploaded retinal image"
                className="w-full h-64 object-cover"
              />
              {(isLoading || checkingFundus) && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                  <div className="bg-white rounded-lg p-4 flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
                    <span className="text-sm font-medium text-gray-900">{checkingFundus ? 'Verifying fundus image...' : 'Analyzing image...'}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
