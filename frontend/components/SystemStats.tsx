'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Cpu, 
  HardDrive, 
  Zap, 
  Activity, 
  Brain,
  BarChart3,
  Shield,
  Clock,
  Database,
  Eye
} from 'lucide-react'

export default function SystemStats() {
  const [mounted, setMounted] = useState(false)
  const [liveMetrics, setLiveMetrics] = useState({
    cpu: 25,
    memory: 35,
    gpu: 42
  })

  // Prevent hydration mismatch by only rendering dynamic content after mount
  useEffect(() => {
    setMounted(true)
    
    // Update live metrics periodically
    const interval = setInterval(() => {
      setLiveMetrics({
        cpu: Math.floor(Math.random() * 30 + 20),
        memory: Math.floor(Math.random() * 40 + 30),
        gpu: Math.floor(Math.random() * 60 + 20)
      })
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const systemMetrics = [
    { 
      label: 'MODEL ACCURACY', 
      value: '96.2%', 
      icon: BarChart3, 
      color: 'stark-neon',
      description: 'Internal validation performance'
    },
    { 
      label: 'PARAMETERS', 
      value: '2.22M', 
      icon: Cpu, 
      color: 'stark-cyan',
      description: 'Total model parameters'
    },
    { 
      label: 'INFERENCE TIME', 
      value: '21.9ms', 
      icon: Zap, 
      color: 'stark-electric',
      description: 'Average prediction latency'
    },
    { 
      label: 'MODEL SIZE', 
      value: '8.52MB', 
      icon: HardDrive, 
      color: 'stark-amber',
      description: 'Total model footprint'
    },
    { 
      label: 'THROUGHPUT', 
      value: '45.7 IPS', 
      icon: Activity, 
      color: 'stark-purple',
      description: 'Images per second'
    },
    { 
      label: 'EXTERNAL ACC', 
      value: '77.9%', 
      icon: Shield, 
      color: 'stark-red',
      description: 'External validation (adapted)'
    },
  ]

  const trainingStats = [
    { label: 'TRAINING SAMPLES', value: '14,046' },
    { label: 'EXTERNAL SAMPLES', value: '394' },
    { label: 'FEATURE DIMENSION', value: '1,280' },
    { label: 'CLASSES', value: '4' },
  ]

  return (
    <div className="space-y-6">
      
      {/* System Status Header */}
      <motion.div
        className="glass rounded-xl p-6 border border-stark-cyan/30"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="flex items-center space-x-3 mb-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          >
            <Brain className="w-6 h-6 text-stark-cyan" />
          </motion.div>
          <h3 className="text-xl font-bold holographic font-mono">
            SYSTEM STATUS
          </h3>
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-mono text-gray-300">NEURAL NETWORK</span>
            <span className="text-sm font-mono text-stark-neon font-bold">ONLINE</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm font-mono text-gray-300">CLASSIFICATION</span>
            <span className="text-sm font-mono text-stark-electric font-bold">READY</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm font-mono text-gray-300">SECURITY</span>
            <span className="text-sm font-mono text-stark-neon font-bold">SECURE</span>
          </div>
        </div>
      </motion.div>

      {/* Performance Metrics */}
      <motion.div
        className="glass rounded-xl p-6 border border-stark-cyan/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        <h3 className="text-lg font-bold text-stark-cyan mb-4 font-mono flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          PERFORMANCE MATRIX
        </h3>
        
        <div className="space-y-4">
          {systemMetrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              className="group"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + index * 0.1, duration: 0.5 }}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <metric.icon className={`w-4 h-4 text-${metric.color}`} />
                  <span className="text-xs font-mono text-gray-400">
                    {metric.label}
                  </span>
                </div>
                <span className={`text-sm font-bold text-${metric.color} font-mono`}>
                  {metric.value}
                </span>
              </div>
              
              {/* Animated progress bar */}
              <div className="w-full bg-stark-navy/50 rounded-full h-2 overflow-hidden">
                <motion.div
                  className={`h-full bg-${metric.color} progress-glow`}
                  initial={{ width: 0 }}
                  animate={{ width: '85%' }}
                  transition={{ 
                    delay: 0.5 + index * 0.1, 
                    duration: 1.2, 
                    ease: "easeOut" 
                  }}
                />
              </div>
              
              {/* Description on hover */}
              <motion.p
                className="text-xs text-gray-500 mt-1 font-mono opacity-0 group-hover:opacity-100 transition-opacity"
                initial={{ height: 0 }}
                whileHover={{ height: 'auto' }}
              >
                {metric.description}
              </motion.p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Training Data Stats */}
      <motion.div
        className="glass rounded-xl p-6 border border-stark-cyan/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        <h3 className="text-lg font-bold text-stark-cyan mb-4 font-mono flex items-center">
          <Database className="w-5 h-5 mr-2" />
          TRAINING MATRIX
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          {trainingStats.map((stat, index) => (
            <motion.div
              key={stat.label}
              className="text-center p-3 bg-stark-navy/30 rounded-lg border border-stark-cyan/10"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + index * 0.1, duration: 0.4 }}
              whileHover={{ scale: 1.05, y: -2 }}
            >
              <div className="text-lg font-bold text-stark-electric font-mono">
                {stat.value}
              </div>
              <div className="text-xs text-gray-400 font-mono">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Live System Monitor */}
      <motion.div
        className="glass rounded-xl p-6 border border-stark-cyan/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.6 }}
      >
        <h3 className="text-lg font-bold text-stark-cyan mb-4 font-mono flex items-center">
          <Eye className="w-5 h-5 mr-2" />
          LIVE MONITOR
        </h3>
        
        <div className="space-y-3">
          {mounted && [
            { label: 'CPU USAGE', value: liveMetrics.cpu },
            { label: 'MEMORY', value: liveMetrics.memory },
            { label: 'GPU UTIL', value: liveMetrics.gpu },
          ].map((metric, index) => (
            <motion.div
              key={metric.label}
              className="flex items-center justify-between"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 + index * 0.1, duration: 0.5 }}
            >
              <span className="text-xs font-mono text-gray-400">{metric.label}</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-stark-navy/50 rounded-full h-2">
                  <motion.div
                    className="h-full bg-gradient-to-r from-stark-cyan to-stark-electric rounded-full"
                    animate={{ width: `${metric.value}%` }}
                    transition={{ duration: 2, repeat: Infinity, repeatType: "reverse" }}
                  />
                </div>
                <span className="text-xs font-mono text-stark-electric w-8">
                  {metric.value}%
                </span>
              </div>
            </motion.div>
          ))}
          
          {!mounted && (
            <div className="space-y-3">
              {[1, 2, 3].map((_, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-xs font-mono text-gray-400">LOADING...</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-stark-navy/50 rounded-full h-2">
                      <div className="h-full bg-stark-cyan/30 rounded-full w-0" />
                    </div>
                    <span className="text-xs font-mono text-gray-500 w-8">--</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}
