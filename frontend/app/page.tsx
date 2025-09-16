'use client'

import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { 
  Brain, 
  Upload, 
  Zap, 
  Shield, 
  Activity,
  BarChart3,
  ArrowRight,
  Cpu,
  Eye,
  Database
} from 'lucide-react'

import Header from '@/components/Header'

export default function LandingPage() {
  const router = useRouter()

  const features = [
    {
      icon: Brain,
      title: "Neural Classification",
      description: "Advanced MobileNetV2 architecture with 96.2% accuracy",
      color: "stark-cyan"
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description: "Lightning-fast inference in just 21.9ms",
      color: "stark-electric"
    },
    {
      icon: Shield,
      title: "Medical Grade",
      description: "Calibrated predictions with confidence assessment",
      color: "stark-neon"
    },
    {
      icon: BarChart3,
      title: "External Validation",
      description: "Tested on independent datasets for reliability",
      color: "stark-amber"
    }
  ]

  const stats = [
    { label: "Model Accuracy", value: "96.2%", icon: BarChart3 },
    { label: "Parameters", value: "2.22M", icon: Cpu },
    { label: "Inference Time", value: "21.9ms", icon: Zap },
    { label: "Training Samples", value: "14,046", icon: Database }
  ]

  return (
    <div className="min-h-screen bg-stark-black text-white relative overflow-hidden">
      <Header />
      
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center">
        <div className="container mx-auto px-6 py-20 text-center">
          
          {/* Main Title */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-12"
          >
            <motion.h1 
              className="text-6xl md:text-8xl font-bold mb-6 holographic"
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ duration: 1, type: "spring" }}
            >
              NEURAL INTERFACE
            </motion.h1>
            
            <motion.p 
              className="text-2xl md:text-3xl text-stark-cyan font-mono mb-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.8 }}
            >
              AI-Powered Brain MRI Tumor Classification
            </motion.p>
            
            <motion.div 
              className="w-48 h-1 bg-gradient-to-r from-stark-cyan to-stark-electric mx-auto rounded-full"
              initial={{ width: 0 }}
              animate={{ width: 192 }}
              transition={{ delay: 0.6, duration: 1 }}
            />
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="mb-16"
          >
            <motion.button
              onClick={() => router.push('/upload')}
              className="group relative overflow-hidden bg-gradient-to-r from-stark-neon to-stark-electric 
                text-stark-black font-bold text-xl px-12 py-6 rounded-lg
                hover:shadow-lg hover:shadow-stark-neon/50 transition-all duration-300"
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
            >
              <span className="flex items-center space-x-3">
                <Upload className="w-6 h-6" />
                <span>START NEURAL ANALYSIS</span>
                <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
              </span>
              
              {/* Button glow effect */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-stark-neon/20 to-stark-electric/20 rounded-lg"
                animate={{ opacity: [0, 0.5, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </motion.button>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.8 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16"
          >
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                className="glass rounded-xl p-6 border border-stark-cyan/30 group hover:border-stark-cyan transition-all duration-300"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 + index * 0.1, duration: 0.6 }}
                whileHover={{ y: -10, scale: 1.02 }}
              >
                <motion.div
                  className={`w-16 h-16 rounded-full bg-${feature.color}/20 flex items-center justify-center mb-4 mx-auto`}
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                >
                  <feature.icon className={`w-8 h-8 text-${feature.color}`} />
                </motion.div>
                
                <h3 className={`text-lg font-bold text-${feature.color} mb-2 font-mono`}>
                  {feature.title}
                </h3>
                <p className="text-gray-300 text-sm">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>

          {/* Stats Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.4, duration: 0.8 }}
            className="glass rounded-2xl p-8 border border-stark-cyan/30 max-w-4xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-stark-cyan mb-8 font-mono flex items-center justify-center">
              <Activity className="w-8 h-8 mr-3" />
              SYSTEM PERFORMANCE
            </h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  className="text-center"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.6 + index * 0.1, duration: 0.5 }}
                  whileHover={{ scale: 1.1 }}
                >
                  <stat.icon className="w-8 h-8 text-stark-electric mx-auto mb-3" />
                  <div className="text-3xl font-bold text-stark-electric font-mono mb-1">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-400 font-mono uppercase">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-stark-cyan/20 bg-stark-navy/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <Brain className="w-6 h-6 text-stark-cyan" />
              <span className="text-lg text-gray-300 font-mono">
                Neural Interface Medical AI System
              </span>
            </div>
            <div className="flex items-center space-x-6 text-sm text-gray-400">
              <span className="flex items-center space-x-2">
                <Shield className="w-4 h-4" />
                <span>Research Grade</span>
              </span>
              <span className="flex items-center space-x-2">
                <Eye className="w-4 h-4" />
                <span>Explainable AI</span>
              </span>
              <span className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span>Validated</span>
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}