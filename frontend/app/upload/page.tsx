'use client'

import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { 
  ArrowLeft,
  Wifi,
  WifiOff
} from 'lucide-react'

import Header from '@/components/Header'
import UploadZone from '@/components/UploadZone'
import LoadingAnimation from '@/components/LoadingAnimation'
import { brainMRIAPI, type PredictionResult } from '@/lib/api'

export default function UploadPage() {
  const router = useRouter()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [modelReady, setModelReady] = useState(false)

  // Check connection and model status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const connected = await brainMRIAPI.checkConnection()
        setIsConnected(connected)
        
        if (connected) {
          const health = await brainMRIAPI.healthCheck()
          setModelReady(health.brain_mri_model_loaded && health.medical_validator_loaded)
        }
      } catch (error) {
        console.error('Status check failed:', error)
        setIsConnected(false)
        setModelReady(false)
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile || !isConnected || !modelReady) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const result = await brainMRIAPI.predictImage(selectedFile)
      
      if (result.success) {
        // Store results in sessionStorage and navigate to results page
        sessionStorage.setItem('analysisResults', JSON.stringify(result))
        sessionStorage.setItem('uploadedFileName', selectedFile.name)
        router.push('/results')
      } else {
        setError(result.error || 'Analysis failed. Please try again.')
      }
    } catch (err) {
      setError('Connection failed. Please ensure the backend is running.')
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile, isConnected, modelReady, router])

  return (
    <div className="min-h-screen bg-stark-black text-white relative overflow-hidden">
      <Header />
      
      {/* Back to Home Button */}
      <motion.button
        onClick={() => router.push('/')}
        className="fixed top-20 left-6 z-50 glass rounded-lg px-4 py-2 flex items-center space-x-2 border-stark-cyan/30 hover:border-stark-cyan transition-colors"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <ArrowLeft className="w-4 h-4 text-stark-cyan" />
        <span className="text-stark-cyan text-sm font-mono">HOME</span>
      </motion.button>
      
      {/* Connection Status */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed top-20 right-6 z-50"
      >
        <div className={`glass rounded-lg px-4 py-2 flex items-center space-x-2 ${
          isConnected && modelReady 
            ? 'border-green-500/50 bg-green-500/10' 
            : 'border-red-500/50 bg-red-500/10'
        }`}>
          {isConnected && modelReady ? (
            <>
              <Wifi className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm font-mono">NEURAL NET ONLINE</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-red-400" />
              <span className="text-red-400 text-sm font-mono">
                {!isConnected ? 'BACKEND OFFLINE' : 'MODEL LOADING...'}
              </span>
            </>
          )}
        </div>
      </motion.div>
      
      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 pt-32">
        <div className="max-w-4xl mx-auto">
          
          {/* Page Title */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <motion.h1 
              className="text-4xl md:text-5xl font-bold mb-4 holographic"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.8, type: "spring" }}
            >
              NEURAL SCAN INTERFACE
            </motion.h1>
            <motion.p 
              className="text-xl text-stark-cyan font-mono"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.6 }}
            >
              Upload Brain MRI for AI-Powered Tumor Classification
            </motion.p>
            <motion.div 
              className="w-32 h-1 bg-gradient-to-r from-stark-cyan to-stark-electric mx-auto mt-4 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: 128 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            />
          </motion.div>

          {/* Upload Zone */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            <UploadZone
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              isAnalyzing={isAnalyzing}
              onAnalyze={handleAnalyze}
              disabled={!isConnected || !modelReady}
            />
          </motion.div>

          {/* Loading Animation */}
          <AnimatePresence>
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.5 }}
                className="mt-8"
              >
                <LoadingAnimation />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error Display */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 50 }}
                className="mt-8 glass rounded-lg p-6 border-stark-red bg-red-500/10"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 text-stark-red">⚠️</div>
                  <p className="text-stark-red font-medium">{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}
