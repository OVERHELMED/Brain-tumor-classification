'use client'

import { motion } from 'framer-motion'
import { 
  Brain, 
  AlertTriangle, 
  Shield, 
  Activity, 
  Clock,
  TrendingUp,
  Eye,
  Zap,
  Cpu
} from 'lucide-react'
import * as Progress from '@radix-ui/react-progress'
import { PredictionResult, getRiskColor, getRiskIcon, formatInferenceTime } from '@/lib/api'

interface ResultsPanelProps {
  results: PredictionResult
}

export default function ResultsPanel({ results }: ResultsPanelProps) {
  if (!results.success) {
    const isValidationError = results.error?.includes('valid brain MRI image')
    
    return (
      <motion.div
        className="glass rounded-2xl p-8 border border-red-500/30 bg-red-500/10"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, type: "spring" }}
      >
        <div className="text-center">
          <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-red-400 mb-4">
            {isValidationError ? 'INVALID IMAGE TYPE' : 'ANALYSIS FAILED'}
          </h2>
          <p className="text-gray-300 mb-4">{results.error}</p>
          
          {isValidationError && (
            <div className="glass rounded-lg p-4 border border-stark-amber/30 bg-stark-amber/10 mt-4">
              <h4 className="text-stark-amber font-bold mb-2 font-mono">UPLOAD REQUIREMENTS:</h4>
              <ul className="text-sm text-gray-300 space-y-1 font-mono text-left">
                <li>• Brain MRI scans (T1-weighted preferred)</li>
                <li>• Medical imaging formats (DICOM, JPG, PNG)</li>
                <li>• Grayscale or medical imaging appearance</li>
                <li>• Clear brain tissue visibility</li>
              </ul>
            </div>
          )}
        </div>
      </motion.div>
    )
  }

  const { prediction, confidence, probabilities, inference_time_ms, description, risk_level } = results

  // Class display names and colors
  const classInfo = {
    GLIOMA: { name: 'GLIOMA', color: 'text-red-400', bgColor: 'bg-red-500/20', description: 'Aggressive glial cell tumor' },
    MENINGIOMA: { name: 'MENINGIOMA', color: 'text-orange-400', bgColor: 'bg-orange-500/20', description: 'Meninges tumor' },
    PITUITARY: { name: 'PITUITARY', color: 'text-purple-400', bgColor: 'bg-purple-500/20', description: 'Pituitary gland tumor' },
    NOTUMOR: { name: 'NO TUMOR', color: 'text-green-400', bgColor: 'bg-green-500/20', description: 'Normal brain tissue' },
  }

  const currentClass = classInfo[prediction as keyof typeof classInfo]
  const riskColor = getRiskColor(risk_level || 'LOW')
  const riskIcon = getRiskIcon(risk_level || 'LOW')

  return (
    <motion.div
      className="glass rounded-2xl p-8 border border-stark-cyan/30"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, type: "spring" }}
    >
      
      {/* Header */}
      <motion.div
        className="text-center mb-8"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        <h2 className="text-3xl font-bold holographic mb-2">
          NEURAL ANALYSIS COMPLETE
        </h2>
        <div className="w-24 h-1 bg-gradient-to-r from-stark-cyan to-stark-electric mx-auto rounded-full" />
      </motion.div>

      {/* Main Result */}
      <motion.div
        className={`${currentClass?.bgColor} rounded-xl p-6 border border-current/20 mb-6`}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            <motion.div
              className={`w-16 h-16 rounded-full ${currentClass?.bgColor} border-2 border-current flex items-center justify-center`}
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <Brain className={`w-8 h-8 ${currentClass?.color}`} />
            </motion.div>
            
            <div>
              <h3 className={`text-2xl font-bold ${currentClass?.color} font-mono`}>
                {currentClass?.name}
              </h3>
              <p className="text-gray-300 text-sm font-mono">
                {currentClass?.description}
              </p>
            </div>
          </div>

          {/* Confidence Badge */}
          <motion.div
            className="text-right"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6, duration: 0.4 }}
          >
            <div className="text-3xl font-bold text-stark-electric font-mono">
              {confidence?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 font-mono">CONFIDENCE</div>
          </motion.div>
        </div>

        {/* Confidence Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm font-mono mb-3">
            <span className="text-gray-400">NEURAL CERTAINTY</span>
            <span className={`${riskColor} font-bold`}>
              {risk_level} RISK {riskIcon}
            </span>
          </div>
          
          <div className="relative">
            <div className="w-full bg-stark-navy/50 rounded-full h-3 overflow-hidden">
              <motion.div
                className={`h-full bg-gradient-to-r ${
                  (confidence || 0) > 80 
                    ? 'from-stark-neon to-stark-electric' 
                    : (confidence || 0) > 60 
                      ? 'from-stark-amber to-stark-cyan'
                      : 'from-stark-red to-stark-amber'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ delay: 0.3, duration: 0.8, ease: "easeOut" }}
              />
            </div>
            
            {/* Confidence percentage below the bar */}
            <div className="text-center mt-2">
              <span className="text-xs font-mono text-stark-electric">
                {confidence?.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Class Probabilities */}
      <motion.div
        className="mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.6 }}
      >
        <h4 className="text-lg font-bold text-stark-cyan mb-4 font-mono flex items-center">
          <TrendingUp className="w-5 h-5 mr-2" />
          CLASS PROBABILITY MATRIX
        </h4>
        
        <div className="space-y-4">
          {probabilities && Object.entries(probabilities).map(([className, probability], index) => {
            const classStyle = classInfo[className as keyof typeof classInfo]
            
            return (
              <motion.div
                key={className}
                className="space-y-2"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + index * 0.1, duration: 0.4 }}
              >
                <div className="flex justify-between items-center">
                  <span className="text-sm font-mono text-gray-300 uppercase">
                    {classStyle?.name}
                  </span>
                  <span className="text-xs font-mono text-stark-electric">
                    {probability.toFixed(1)}%
                  </span>
                </div>
                
                <div className="w-full bg-stark-navy/50 rounded-full h-2 overflow-hidden">
                  <motion.div
                    className={`h-full ${
                      className === prediction 
                        ? 'bg-gradient-to-r from-stark-cyan to-stark-electric'
                        : 'bg-gray-600/50'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${probability}%` }}
                    transition={{ 
                      delay: 0.6 + index * 0.1, 
                      duration: 0.6, 
                      ease: "easeOut" 
                    }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      </motion.div>

      {/* System Metrics */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1, duration: 0.6 }}
      >
        
        {/* Risk Assessment */}
        <div className="glass rounded-lg p-4 border border-stark-cyan/20">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className="w-5 h-5 text-stark-neon" />
            <span className="text-sm font-mono text-gray-300">RISK LEVEL</span>
          </div>
          <div className={`text-xl font-bold ${riskColor} font-mono`}>
            {risk_level} {riskIcon}
          </div>
        </div>

        {/* Inference Time */}
        <div className="glass rounded-lg p-4 border border-stark-cyan/20">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="w-5 h-5 text-stark-electric" />
            <span className="text-sm font-mono text-gray-300">INFERENCE</span>
          </div>
          <div className="text-xl font-bold text-stark-electric font-mono">
            {formatInferenceTime(inference_time_ms || 0)}
          </div>
        </div>

        {/* Model Info */}
        <div className="glass rounded-lg p-4 border border-stark-cyan/20">
          <div className="flex items-center space-x-2 mb-2">
            <Cpu className="w-5 h-5 text-stark-neon" />
            <span className="text-sm font-mono text-gray-300">MODEL</span>
          </div>
          <div className="text-xl font-bold text-stark-neon font-mono">
            MNetV2
          </div>
        </div>
      </motion.div>

      {/* Medical Disclaimer */}
      <motion.div
        className="mt-6 p-4 bg-stark-amber/10 border border-stark-amber/30 rounded-lg"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2, duration: 0.6 }}
      >
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-stark-amber mt-0.5" />
          <div>
            <h5 className="text-stark-amber font-bold mb-1 font-mono">MEDICAL DISCLAIMER</h5>
            <p className="text-xs text-gray-300 font-mono leading-relaxed">
              This AI system is for research purposes only. Results should not be used for 
              medical diagnosis without consultation with qualified healthcare professionals.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
