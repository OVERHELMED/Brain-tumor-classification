'use client'

import { motion } from 'framer-motion'
import { Brain, Zap, Activity, Shield, Eye } from 'lucide-react'

export default function Header() {
  const systemStatus = [
    { label: 'NEURAL NET', status: 'ONLINE', color: 'text-stark-neon' },
    { label: 'CLASSIFIER', status: 'ACTIVE', color: 'text-stark-cyan' },
    { label: 'SECURITY', status: 'SECURE', color: 'text-stark-electric' },
  ]

  return (
    <header className="border-b border-stark-cyan/20 bg-stark-navy/50 backdrop-blur-sm">
      <div className="container mx-auto px-6 py-4">
        <div className="flex flex-col lg:flex-row justify-between items-center">
          
          {/* Logo and Title */}
          <motion.div 
            className="flex items-center space-x-4 mb-4 lg:mb-0"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="relative">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="w-12 h-12 rounded-full border-2 border-stark-cyan glow-cyan"
              >
                <Brain className="w-8 h-8 text-stark-cyan absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
              </motion.div>
              <motion.div
                className="absolute -top-1 -right-1 w-3 h-3 bg-stark-neon rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
            
            <div>
              <h1 className="text-2xl font-bold holographic">
                BRAIN AI NEURAL NET
              </h1>
              <p className="text-sm text-stark-cyan font-mono">
                Classification System v2.0
              </p>
            </div>
          </motion.div>

          {/* System Status Indicators */}
          <motion.div 
            className="flex flex-wrap justify-center lg:justify-end space-x-6"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {systemStatus.map((status, index) => (
              <motion.div
                key={status.label}
                className="flex items-center space-x-2"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1, duration: 0.4 }}
              >
                <motion.div
                  className={`w-2 h-2 rounded-full ${status.color === 'text-stark-neon' ? 'bg-stark-neon' : status.color === 'text-stark-cyan' ? 'bg-stark-cyan' : 'bg-stark-electric'}`}
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <span className="text-xs font-mono text-gray-300">
                  {status.label}
                </span>
                <span className={`text-xs font-mono font-bold ${status.color}`}>
                  {status.status}
                </span>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Secondary Navigation */}
        <motion.div 
          className="flex flex-wrap justify-center lg:justify-start space-x-8 mt-4 pt-4 border-t border-stark-cyan/10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.6 }}
        >
          {[
            { icon: Zap, label: 'REAL-TIME', value: '21.9ms' },
            { icon: Activity, label: 'ACCURACY', value: '96.2%' },
            { icon: Shield, label: 'SECURITY', value: 'HIPAA' },
            { icon: Eye, label: 'EXPLAINABLE', value: 'GRAD-CAM' },
          ].map((item, index) => (
            <motion.div
              key={item.label}
              className="flex items-center space-x-2 text-sm"
              whileHover={{ scale: 1.05, y: -2 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
            >
              <item.icon className="w-4 h-4 text-stark-cyan" />
              <span className="text-gray-400 font-mono">{item.label}:</span>
              <span className="text-stark-electric font-mono font-bold">{item.value}</span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </header>
  )
}
