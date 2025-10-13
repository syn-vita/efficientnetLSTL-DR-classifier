import { useState } from 'react'
import { Eye, Brain, Zap, Shield, Award, ChevronRight, ArrowRight, Users, FileImage, BarChart3 } from 'lucide-react'
import SeverityCarousel from './SeverityCarousel'

export default function LandingPage({ onStartTesting }) {
  const [currentFeature, setCurrentFeature] = useState(0)

  const features = [
    {
      icon: Brain,
      title: "Advanced AI Models",
      description: "Three state-of-the-art deep learning architectures: EfficientNet-B0 baseline, CBAM-enhanced, and LSTL-optimized models."
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description: "Instant diabetic retinopathy classification with confidence scores and detailed probability distributions."
    },
    {
      icon: Shield,
      title: "Medical Grade Accuracy",
      description: "Models trained on clinical datasets with validation accuracy exceeding 85% for reliable diagnostic support."
    },
    {
      icon: Award,
      title: "Research-Backed",
      description: "Implementation based on peer-reviewed research and best practices in medical AI applications."
    }
  ]

  const stats = [
    { label: "Model Accuracy", value: "85%+", icon: BarChart3 },
    { label: "Processing Speed", value: "<2s", icon: Zap },
    { label: "DR Classes", value: "5", icon: FileImage },
    { label: "Model Variants", value: "3", icon: Brain }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600/5 to-indigo-600/5"></div>
        <div className="relative max-w-7xl mx-auto px-4 py-20">
          <div className="text-center space-y-8">
            {/* Logo and Title */}
            <div className="flex justify-center">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                    <Eye className="w-8 h-8 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                </div>
                <div className="text-left">
                  <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                    DR Classification
                  </h1>
                  <p className="text-lg text-gray-600 font-medium">AI-Powered Diabetic Retinopathy Analysis</p>
                </div>
              </div>
            </div>

            {/* Subtitle */}
            <div className="max-w-3xl mx-auto space-y-4">
              <p className="text-xl md:text-2xl text-gray-700 leading-relaxed">
                Advanced deep learning models for automated diabetic retinopathy classification
              </p>
              <p className="text-lg text-gray-600">
                Leveraging EfficientNet architectures with attention mechanisms for precise medical image analysis
              </p>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-6">
              <button 
                onClick={onStartTesting}
                className="group bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 flex items-center space-x-3"
              >
                <span>Start Testing Models</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              <button 
                onClick={() => document.querySelector('.severity-carousel')?.scrollIntoView({ behavior: 'smooth' })}
                className="px-8 py-4 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold text-lg hover:border-blue-500 hover:text-blue-600 transition-all duration-300 flex items-center space-x-3"
              >
                <span>Learn More</span>
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white border-y border-gray-100">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center group">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300">
                  <stat.icon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="text-2xl font-bold text-gray-900 mb-1">{stat.value}</div>
                <div className="text-sm text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Cutting-Edge AI Technology
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our research implements and compares multiple deep learning architectures for diabetic retinopathy classification
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
            {features.map((feature, index) => (
              <div 
                key={index}
                className={`group p-8 rounded-2xl border-2 transition-all duration-300 cursor-pointer ${
                  currentFeature === index 
                    ? 'border-blue-500 bg-blue-50/50 shadow-lg' 
                    : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
                }`}
                onMouseEnter={() => setCurrentFeature(index)}
              >
                <div className="flex items-start space-x-4">
                  <div className={`p-3 rounded-xl transition-all duration-300 ${
                    currentFeature === index 
                      ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white' 
                      : 'bg-gray-100 text-gray-600 group-hover:bg-blue-100 group-hover:text-blue-600'
                  }`}>
                    <feature.icon className="w-6 h-6" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                    <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Severity Levels Carousel */}
      <SeverityCarousel />

      {/* Model Comparison Section */}
      <section className="py-20 bg-gradient-to-br from-gray-50 to-blue-50/30">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Model Architecture Comparison
            </h2>
            <p className="text-xl text-gray-600">
              Three distinct implementations for diabetic retinopathy classification
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                name: "EfficientNet-B0 Baseline",
                description: "Standard EfficientNet-B0 architecture serving as our baseline model for comparison.",
                accuracy: "83.2%",
                params: "5.3M",
                color: "from-gray-600 to-gray-700"
              },
              {
                name: "CBAM Enhanced",
                description: "EfficientNet-B0 with Convolutional Block Attention Module for improved feature selection.",
                accuracy: "85.7%",
                params: "5.4M",
                color: "from-blue-600 to-indigo-600"
              },
              {
                name: "LSTL Optimized",
                description: "EfficientNet-B0 with the Lightweight Spatial Transformer Layer for computational-efficiency.",
                accuracy: "84.9%",
                params: "5.2M",
                color: "from-emerald-600 to-green-600"
              }
            ].map((model, index) => (
              <div key={index} className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2">
                <div className={`w-12 h-12 bg-gradient-to-br ${model.color} rounded-xl flex items-center justify-center mb-6`}>
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">{model.name}</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">{model.description}</p>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Accuracy</span>
                    <span className="font-semibold text-gray-900">{model.accuracy}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Parameters</span>
                    <span className="font-semibold text-gray-900">{model.params}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Impact Section */}
      <section className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4">
          <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-3xl p-12 text-white text-center">
            <Users className="w-16 h-16 mx-auto mb-6 text-blue-200" />
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Social Impact & Applications
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto leading-relaxed">
              This research aims to contribute to the field of deep learning and computer-aided medical image analysis, 
              potentially improving early detection of diabetic retinopathy in clinical settings.
            </p>
            <button 
              onClick={onStartTesting}
              className="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-50 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              Explore the Models
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="font-semibold">DR Classification Tool</div>
                <div className="text-sm text-gray-400">For Research & Educational Use Only</div>
              </div>
            </div>
            <div className="text-sm text-gray-400 text-center md:text-right">
              <p>Enhancing EfficientNet-B0 Using A Local-Global Transformer Layer For Diabetic Retinopathy Classification</p>
              <p className="mt-1">Alano • Bajit • Labuyo • Lugue • Vega • 2025</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}