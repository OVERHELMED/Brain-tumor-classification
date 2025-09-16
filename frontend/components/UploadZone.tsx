'use client'

import { useCallback, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, FileImage, Zap, AlertCircle, CheckCircle } from 'lucide-react'

interface UploadZoneProps {
  onFileSelect: (file: File) => void
  selectedFile: File | null
  isAnalyzing: boolean
  onAnalyze: () => void
  disabled?: boolean
}

export default function UploadZone({ 
  onFileSelect, 
  selectedFile, 
  isAnalyzing, 
  onAnalyze,
  disabled = false
}: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [])

  const handleFileSelect = useCallback((file: File) => {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp']
    if (!allowedTypes.includes(file.type)) {
      alert('Invalid file type. Please upload a valid image file.')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size too large. Please upload images smaller than 10MB.')
      return
    }

    onFileSelect(file)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }, [onFileSelect])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  return (
    <div className="space-y-6">
      
      {/* Upload Area */}
      <motion.div
        className={`
          relative glass rounded-2xl p-8 border-2 transition-all duration-300 cursor-pointer
          ${isDragOver 
            ? 'border-stark-electric glow-electric bg-stark-electric/10' 
            : selectedFile 
              ? 'border-stark-neon glow-neon bg-stark-neon/5'
              : 'border-stark-cyan/30 hover:border-stark-cyan hover:glow-cyan'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
        whileHover={{ scale: 1.02, y: -5 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        
        {/* Scan line effect */}
        <div className="scan-line absolute inset-0 rounded-2xl overflow-hidden opacity-30" />
        
        {/* Content */}
        <div className="relative z-10">
          <AnimatePresence mode="wait">
            {!selectedFile ? (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="text-center"
              >
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                  className="mb-6"
                >
                  <Upload className="w-16 h-16 text-stark-cyan mx-auto mb-4" />
                </motion.div>
                
                <h3 className="text-2xl font-bold text-stark-cyan mb-2">
                  NEURAL SCAN INTERFACE
                </h3>
                <p className="text-gray-300 mb-4 font-mono">
                  Drag and drop MRI image or click to browse
                </p>
                <p className="text-sm text-gray-500 font-mono">
                  Supported: JPG, PNG, BMP, TIFF, WEBP • Max 10MB
                </p>
                
                {/* Floating elements */}
                <div className="absolute top-4 right-4">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                    className="w-8 h-8 border-2 border-stark-cyan/30 rounded-full"
                  />
                </div>
                <div className="absolute bottom-4 left-4">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 3, repeat: Infinity }}
                    className="w-6 h-6 bg-stark-neon/20 rounded-full"
                  />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="text-center"
              >
                <motion.div
                  className="relative mb-6"
                  whileHover={{ scale: 1.05 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  {imagePreview && (
                    <div className="relative inline-block">
                      <img
                        src={imagePreview}
                        alt="MRI Preview"
                        className="w-48 h-48 object-cover rounded-lg border-2 border-stark-cyan shadow-lg"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-stark-cyan/20 to-transparent rounded-lg" />
                      <motion.div
                        className="absolute top-2 right-2 w-4 h-4 bg-stark-neon rounded-full"
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                    </div>
                  )}
                </motion.div>
                
                <div className="flex items-center justify-center space-x-3 mb-4">
                  <CheckCircle className="w-6 h-6 text-stark-neon" />
                  <span className="text-stark-neon font-bold">FILE LOADED</span>
                </div>
                
                <p className="text-gray-300 font-mono mb-2">{selectedFile.name}</p>
                <p className="text-sm text-gray-500 font-mono">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Hidden file input */}
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />
      </motion.div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <motion.button
          onClick={() => document.getElementById('file-input')?.click()}
          className="btn-stark flex items-center justify-center space-x-3 px-8 py-4 text-lg"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          disabled={isAnalyzing}
        >
          <FileImage className="w-5 h-5" />
          <span>SELECT IMAGE</span>
        </motion.button>

        <motion.button
          onClick={onAnalyze}
          disabled={!selectedFile || isAnalyzing || disabled}
          className={`
            flex items-center justify-center space-x-3 px-8 py-4 text-lg font-bold rounded-lg
            transition-all duration-300 relative overflow-hidden
            ${selectedFile && !isAnalyzing && !disabled
              ? 'bg-gradient-to-r from-stark-neon to-stark-electric text-stark-black hover:shadow-lg hover:shadow-stark-neon/50 hover:scale-105'
              : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }
          `}
          whileHover={selectedFile && !isAnalyzing && !disabled ? { scale: 1.05 } : {}}
          whileTap={selectedFile && !isAnalyzing && !disabled ? { scale: 0.95 } : {}}
        >
          <motion.div
            animate={isAnalyzing ? { rotate: 360 } : {}}
            transition={isAnalyzing ? { duration: 1, repeat: Infinity, ease: "linear" } : {}}
          >
            <Zap className="w-5 h-5" />
          </motion.div>
          <span>
            {disabled ? 'NEURAL NET OFFLINE' : isAnalyzing ? 'ANALYZING...' : 'ANALYZE BRAIN'}
          </span>
          
          {/* Button glow effect */}
          {selectedFile && !isAnalyzing && !disabled && (
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-stark-neon/20 to-stark-electric/20 rounded-lg"
              animate={{ opacity: [0, 0.5, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}
        </motion.button>
      </div>

      {/* File Requirements */}
      <motion.div
        className="glass rounded-lg p-4 border border-stark-cyan/20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.6 }}
      >
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-stark-amber mt-0.5" />
          <div>
            <h4 className="text-stark-amber font-bold mb-2 font-mono">SYSTEM REQUIREMENTS</h4>
            <ul className="text-sm text-gray-300 space-y-1 font-mono">
              <li>• Brain MRI images (T1-weighted preferred)</li>
              <li>• Supported formats: JPG, PNG, BMP, TIFF, WEBP</li>
              <li>• Optimal resolution: 224x224 pixels or higher</li>
              <li>• Maximum file size: 10MB</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
