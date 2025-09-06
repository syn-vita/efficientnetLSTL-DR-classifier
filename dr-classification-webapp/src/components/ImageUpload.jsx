import { useState, useRef } from 'react'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

export default function ImageUpload({ onImageUpload, onClear, uploadedImage, isLoading }) {
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

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

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file)
    } else {
      alert('Please select a valid image file.')
    }
  }

  const handleFileInput = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleClick = () => {
    if (!isLoading) {
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
          className={`upload-area ${dragOver ? 'dragover' : ''} ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
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
            Drag and drop your fundus image here, or click to browse
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
            disabled={isLoading}
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
              {isLoading && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                  <div className="bg-white rounded-lg p-4 flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
                    <span className="text-sm font-medium text-gray-900">Analyzing image...</span>
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
