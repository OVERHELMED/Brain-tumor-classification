'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { 
  ArrowLeft,
  Upload,
  Home,
  RotateCcw
} from 'lucide-react'

import Header from '@/components/Header'
import ResultsPanel from '@/components/ResultsPanel'
import SystemStats from '@/components/SystemStats'
import { type PredictionResult } from '@/lib/api'

export default function ResultsPage() {
  const router = useRouter()
  const [results, setResults] = useState<PredictionResult | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get results from sessionStorage
    const storedResults = sessionStorage.getItem('analysisResults')
    const storedFileName = sessionStorage.getItem('uploadedFileName')
    
    if (storedResults) {
      setResults(JSON.parse(storedResults))
      setFileName(storedFileName || 'Unknown File')
    } else {
      // No results found, redirect to upload
      router.push('/upload')
    }
    
    setLoading(false)
  }, [router])

  const handleNewAnalysis = () => {
    // Clear stored results and go to upload
    sessionStorage.removeItem('analysisResults')
    sessionStorage.removeItem('uploadedFileName')
    router.push('/upload')
  }

  const handleBackToUpload = () => {
    router.push('/upload')
  }

  const handleBackToHome = () => {
    router.push('/')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-stark-black text-white flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-8 h-8 border-2 border-stark-cyan border-t-transparent rounded-full"
        />
      </div>
    )
  }

  if (!results) {
    return (
      <div className="min-h-screen bg-stark-black text-white flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-stark-red mb-4">No Results Found</h2>
          <button
            onClick={() => router.push('/upload')}
            className="btn-stark"
          >
            Go to Upload
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-stark-black text-white relative overflow-hidden">
      <Header />
      
      {/* Navigation Buttons */}
      <div className="fixed top-20 left-6 z-50 flex space-x-3">
        <motion.button
          onClick={handleBackToHome}
          className="glass rounded-lg px-4 py-2 flex items-center space-x-2 border-stark-cyan/30 hover:border-stark-cyan transition-colors"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Home className="w-4 h-4 text-stark-cyan" />
          <span className="text-stark-cyan text-sm font-mono">HOME</span>
        </motion.button>
        
        <motion.button
          onClick={handleBackToUpload}
          className="glass rounded-lg px-4 py-2 flex items-center space-x-2 border-stark-electric/30 hover:border-stark-electric transition-colors"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <ArrowLeft className="w-4 h-4 text-stark-electric" />
          <span className="text-stark-electric text-sm font-mono">UPLOAD</span>
        </motion.button>
      </div>
      
      {/* New Analysis Button */}
      <motion.button
        onClick={handleNewAnalysis}
        className="fixed top-20 right-6 z-50 glass rounded-lg px-4 py-2 flex items-center space-x-2 border-stark-neon/30 hover:border-stark-neon transition-colors"
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <RotateCcw className="w-4 h-4 text-stark-neon" />
        <span className="text-stark-neon text-sm font-mono">NEW ANALYSIS</span>
      </motion.button>
      
      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 pt-32">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 max-w-7xl mx-auto">
          
          {/* Left Panel - Results */}
          <div className="xl:col-span-2 space-y-6">
            
            {/* Page Title */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-8"
            >
              <motion.h1 
                className="text-4xl md:text-5xl font-bold mb-4 holographic"
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.8, type: "spring" }}
              >
                ANALYSIS COMPLETE
              </motion.h1>
              <motion.p 
                className="text-lg text-stark-cyan font-mono"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                Neural Classification Results for: <span className="text-stark-electric">{fileName}</span>
              </motion.p>
              <motion.div 
                className="w-32 h-1 bg-gradient-to-r from-stark-cyan to-stark-electric mx-auto mt-4 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: 128 }}
                transition={{ delay: 0.6, duration: 0.8 }}
              />
            </motion.div>

            {/* Results Panel */}
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, type: "spring" }}
            >
              <ResultsPanel results={results} />
            </motion.div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="flex flex-col sm:flex-row gap-4 justify-center mt-8"
            >
              <motion.button
                onClick={handleNewAnalysis}
                className="btn-stark flex items-center justify-center space-x-3 px-8 py-4 text-lg"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Upload className="w-5 h-5" />
                <span>ANALYZE NEW IMAGE</span>
              </motion.button>

              <motion.button
                onClick={handleBackToUpload}
                className="flex items-center justify-center space-x-3 px-8 py-4 text-lg font-bold rounded-lg
                  bg-stark-navy/50 text-stark-cyan border border-stark-cyan/30 
                  hover:bg-stark-cyan/10 hover:border-stark-cyan transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ArrowLeft className="w-5 h-5" />
                <span>BACK TO UPLOAD</span>
              </motion.button>
            </motion.div>
          </div>

          {/* Right Panel - System Stats */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
            >
              <SystemStats />
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  )
}
