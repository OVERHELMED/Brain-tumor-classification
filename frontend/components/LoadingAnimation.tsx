'use client'

import { motion } from 'framer-motion'
import { Brain, Cpu, Zap, Activity, FileImage, CheckCircle } from 'lucide-react'

export default function LoadingAnimation() {
  const processingSteps = [
    { icon: FileImage, label: 'IMAGE PROCESSING', delay: 0 },
    { icon: Brain, label: 'NEURAL EXTRACTION', delay: 0.5 },
    { icon: Cpu, label: 'FEATURE ANALYSIS', delay: 1.0 },
    { icon: Zap, label: 'CLASSIFICATION', delay: 1.5 },
  ]

  return (
    <div className="glass rounded-2xl p-8 border border-stark-cyan/30">
      
      {/* Main Loading Animation */}
      <div className="text-center mb-8">
        <motion.div
          className="relative mx-auto mb-6"
          style={{ width: 120, height: 120 }}
        >
          {/* Outer ring */}
          <motion.div
            className="absolute inset-0 border-4 border-stark-cyan/30 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Middle ring */}
          <motion.div
            className="absolute inset-2 border-4 border-stark-electric/50 rounded-full"
            animate={{ rotate: -360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Inner ring */}
          <motion.div
            className="absolute inset-4 border-4 border-stark-neon/70 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Center brain icon */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Brain className="w-8 h-8 text-stark-cyan" />
          </motion.div>
          
          {/* Pulsing dots */}
          {Array.from({ length: 8 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-stark-electric rounded-full"
              style={{
                top: '50%',
                left: '50%',
                transformOrigin: '0 0',
              }}
              animate={{
                rotate: i * 45,
                scale: [0, 1, 0],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                delay: i * 0.2,
              }}
            />
          ))}
        </motion.div>

        <h3 className="text-2xl font-bold text-stark-cyan mb-2 holographic">
          NEURAL ANALYSIS IN PROGRESS
        </h3>
        
        <p className="text-gray-300 font-mono">
          Processing brain imaging data...
        </p>
      </div>

      {/* Processing Steps */}
      <div className="space-y-4">
        {processingSteps.map((step, index) => (
          <motion.div
            key={step.label}
            className="flex items-center space-x-4 p-4 glass rounded-lg border border-stark-cyan/20"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: step.delay, duration: 0.6 }}
          >
            <div className="w-10 h-10 rounded-full bg-stark-cyan/20 flex items-center justify-center">
              <step.icon className="w-5 h-5 text-stark-cyan" />
            </div>
            
            <div className="flex-1">
              <h4 className="font-bold text-stark-electric font-mono">{step.label}</h4>
              <div className="w-full bg-stark-navy/50 rounded-full h-2 mt-2 overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-stark-cyan to-stark-electric"
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ 
                    delay: step.delay + 0.3, 
                    duration: 1.2,
                    ease: "easeOut"
                  }}
                />
              </div>
            </div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: step.delay + 1.5 }}
              className="text-stark-neon"
            >
              <CheckCircle className="w-5 h-5" />
            </motion.div>
          </motion.div>
        ))}
      </div>

      {/* System Status */}
      <motion.div
        className="mt-6 p-4 bg-stark-navy/30 rounded-lg border border-stark-cyan/20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 0.6 }}
      >
        <div className="flex items-center justify-between text-sm font-mono">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-stark-neon" />
            <span className="text-gray-300">SYSTEM STATUS:</span>
            <span className="text-stark-neon font-bold">OPERATIONAL</span>
          </div>
          
          <div className="flex items-center space-x-4 text-xs">
            <span className="text-gray-400">CPU: 45.7 IPS</span>
            <span className="text-gray-400">MEM: 8.52 MB</span>
            <span className="text-gray-400">LAT: ~22ms</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
