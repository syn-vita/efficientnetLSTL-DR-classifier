import { useState } from 'react'
import { Globe, Eye, Brain, Zap, Shield, Award, ChevronRight, ArrowRight, Users, FileImage, BarChart3, EyeOff, HeartPulse, AlertTriangle, CalendarCheck } from 'lucide-react'
import SeverityCarousel from './SeverityCarousel'

export default function LandingPage({ onStartTesting }) {
  const [currentFeature, setCurrentFeature] = useState(0)

  const features = [
    {
      icon: Globe,
      title: "A Growing Global Concern",
      description: "DR is one of the the top global causes of vision loss, and its prevalence continues to rise with increasing diabetes cases. Without timely screening, many individuals remain undiagnosed until irreversible vision loss occurs."
    },
    {
      icon: EyeOff,
      title: "How Vision Is Affected",
      description: "Damage to the retinal blood vessels causes them to leak, swell, or close off entirely. This leads to blurred or distorted vision, and in advanced stages, severe retinal scarring or detachment."
    },
    {
      icon: BarChart3,
      title: "Stages of Progression",
      description: "DR progresses gradually, from mild non-proliferative changes to severe proliferative stages, characterized by abnormal vessel growth. Each stage requires specific clinical attention and management."
    },
    {
      icon: HeartPulse,
      title: "Why Early Detection Saves Sight",
      description: "Early diagnosis and intervention can reduce the risk of blindness. Regular retinal screening plays a vital role in preventing avoidable vision loss among diabetic patients."
    }
  ]

  const stats = [
    { label: "People Affected Worldwide", value: "100M+", icon: Globe },
    { label: "Diabetic Patients Develop DR", value: "1 in 3", icon: Eye },
    { label: "Preventable with Early Detection", value: "90%", icon: AlertTriangle },
    { label: "Eye Exams Recommended for Diabetics", value: "Annual", icon: CalendarCheck }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600/5 to-indigo-600/5"></div>
        <div className="relative max-w-7xl mx-auto px-4 py-20">
          <div className="text-center space-y-8">
            {/* Logo, Title, and Details */}
            <div className="text-center mb-10">
              {/* Icon Container*/}
            <div className="relative mx-auto mb-6 w-fit">
              {/* Eye Icon */}
              <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg relative">
                <Eye className="w-8 h-8 text-white" />
                {/* Small Circle Badge */}
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center ring-2 ring-white">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
              </div>
            </div>

              {/* Thesis Title */}
              <h1 className="text-2xl md:text-3xl lg:text-4xl font-semibold leading-snug text-gray-900 max-w-4xl mx-auto">
                Enhancing EfficientNet-B0 Using a Local-Global Transformer Layer for Diabetic Retinopathy Classification
              </h1>

              {/* Formal subtitles */}
              <div className="mt-5">
                <p className="text-lg md:text-xl text-gray-700 font-medium">
                  A Thesis Presented to the Department of Computer Science
                </p>
                <p className="text-sm md:text-base text-gray-500 mt-1 pb-5">
                  In Partial Fulfillment of the Requirements for the Degree of<br />
                  Bachelor of Science in Computer Science
                </p>
              </div>
            </div>

            {/* Authors */}
            <div className="text-center text-gray-600 mb-8 pb-5">
              <p className="text-xs font-semibold uppercase tracking-wider mb-1">By</p>
              <p className="text-sm md:text-base font-medium">
                Ruzel Alano, Giancarlo Bajit, Angel Labuyo, Kevin Lugue, and Francesca Vega
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
              Why Diabetic Retinopathy Matters
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              DR is one of the most common complications of diabetes and is one of the leading causes of preventable blindness.
              Understanding its nature highlights why early detection and monitoring is important.
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
              Three implementations of the DR Classification Tool, as per the Thesis' Statement of the Problem.
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
                description: "EfficientNet-B0 with Convolutional Block Attention Module for improved feature extraction.",
                accuracy: "85.7%",
                params: "5.4M",
                color: "from-blue-600 to-indigo-600"
              },
              {
                name: "LSTL Optimized",
                description: "EfficientNet-B0 with the Local-Global Spatially-Aware Transformer Layer for improved feature extraction and computational efficiency.",
                accuracy: "84.9%",
                params: "5.2M",
                color: "from-emerald-600 to-green-600"
              }
            ].map((model, index) => (
              <div key={index} className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2">
                <div className={`w-12 h-12 bg-gradient-to-br ${model.color} rounded-xl flex items-center justify-center mb-6 mx-auto`}>
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4 text-center">{model.name}</h3>
                <p className="text-gray-600 mb-6 leading-relaxed text-center">{model.description}</p>
                {/* <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Accuracy</span>
                    <span className="font-semibold text-gray-900">{model.accuracy}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">Parameters</span>
                    <span className="font-semibold text-gray-900">{model.params}</span>
                  </div>
                </div> */}
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