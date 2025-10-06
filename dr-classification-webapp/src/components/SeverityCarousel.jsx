import { useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight, Eye, AlertTriangle, CheckCircle, RotateCcw } from 'lucide-react'

const severityLevels = [
  {
    id: 0,
    name: "No Diabetic Retinopathy",
    shortName: "No DR",
    description: "Normal retinal appearance with no signs of diabetic retinopathy. Blood vessels appear healthy with no microaneurysms, hemorrhages, or other pathological changes.",
    clinicalNote: "No diabetic retinopathy detected. Continue regular monitoring and maintain good glycemic control.",
    severity: "Normal",
    color: "green",
    icon: CheckCircle,
    prevalence: "~60%",
    riskLevel: "Low",
    features: [
      "Clear, well-defined blood vessels",
      "No microaneurysms or hemorrhages",
      "Normal optic disc appearance",
      "Healthy macula region"
    ],
    recommendation: "Annual eye examination recommended for diabetic patients",
    // Using a placeholder image - in real implementation, you'd use actual retinal images
    image: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23f0fdf4'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23dcfce7' stroke='%2316a34a' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%2316a34a'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23166534'%3ENo DR - Normal Retina%3C/text%3E%3C/svg%3E",
    realImage: "/sample-images/grade0_sample.png"
  },
  {
    id: 1,
    name: "Mild Non-proliferative DR",
    shortName: "Mild NPDR",
    description: "Early signs of diabetic retinopathy with microaneurysms present. These small bulges in blood vessel walls are the earliest detectable changes in diabetic retinopathy.",
    clinicalNote: "Mild diabetic retinopathy detected. Increased monitoring recommended with focus on glycemic control.",
    severity: "Mild",
    color: "yellow",
    icon: AlertTriangle,
    prevalence: "~25%",
    riskLevel: "Low-Moderate",
    features: [
      "Microaneurysms (small red dots)",
      "Possible small hemorrhages",
      "Blood vessels still largely intact",
      "No significant vision impact"
    ],
    recommendation: "Follow-up examination in 6-12 months, optimize diabetes management",
    image: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fffbeb'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fef3c7' stroke='%23d97706' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23d97706'/%3E%3Ccircle cx='170' cy='130' r='3' fill='%23dc2626'/%3E%3Ccircle cx='230' cy='140' r='2' fill='%23dc2626'/%3E%3Ccircle cx='190' cy='180' r='2' fill='%23dc2626'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%2392400e'%3EMild NPDR%3C/text%3E%3C/svg%3E",
    realImage: "/sample-images/grade1_sample.png"
  },
  {
    id: 2,
    name: "Moderate Non-proliferative DR",
    shortName: "Moderate NPDR",
    description: "Progressive diabetic retinopathy with more widespread microaneurysms, hemorrhages, and possible hard exudates. Blood vessel blockages may begin to occur.",
    clinicalNote: "Moderate diabetic retinopathy with increased vascular changes. More frequent monitoring required.",
    severity: "Moderate",
    color: "orange",
    icon: AlertTriangle,
    prevalence: "~10%",
    riskLevel: "Moderate",
    features: [
      "Multiple microaneurysms",
      "Retinal hemorrhages",
      "Hard exudates (lipid deposits)",
      "Early signs of blood vessel blockage"
    ],
    recommendation: "Follow-up every 3-6 months, consider referral to retinal specialist",
    image: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fff7ed'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fed7aa' stroke='%23ea580c' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23ea580c'/%3E%3Ccircle cx='160' cy='120' r='4' fill='%23dc2626'/%3E%3Ccircle cx='240' cy='130' r='3' fill='%23dc2626'/%3E%3Ccircle cx='180' cy='180' r='3' fill='%23dc2626'/%3E%3Ccircle cx='220' cy='170' r='2' fill='%23dc2626'/%3E%3Crect x='170' y='160' width='6' height='3' fill='%23fbbf24'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23c2410c'%3EModerate NPDR%3C/text%3E%3C/svg%3E",
    realImage: "/sample-images/grade2_sample.png"
  },
  {
    id: 3,
    name: "Severe Non-proliferative DR",
    shortName: "Severe NPDR",
    description: "Advanced non-proliferative diabetic retinopathy with extensive hemorrhages, microaneurysms, and significant blood vessel blockages. High risk of progression to proliferative DR.",
    clinicalNote: "Severe diabetic retinopathy with extensive vascular changes. Urgent ophthalmologic referral recommended.",
    severity: "Severe",
    color: "red",
    icon: AlertTriangle,
    prevalence: "~3%",
    riskLevel: "High",
    features: [
      "Extensive hemorrhages and microaneurysms",
      "Significant cotton wool spots",
      "Venous beading",
      "Intraretinal microvascular abnormalities"
    ],
    recommendation: "Immediate referral to retinal specialist, follow-up every 2-4 months",
    image: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fef2f2'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fecaca' stroke='%23dc2626' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23dc2626'/%3E%3Ccircle cx='150' cy='110' r='5' fill='%23991b1b'/%3E%3Ccircle cx='250' cy='120' r='4' fill='%23991b1b'/%3E%3Ccircle cx='170' cy='190' r='4' fill='%23991b1b'/%3E%3Ccircle cx='230' cy='180' r='3' fill='%23991b1b'/%3E%3Crect x='160' y='140' width='8' height='4' fill='%23fbbf24'/%3E%3Crect x='220' y='160' width='6' height='3' fill='%23fbbf24'/%3E%3Cpath d='M180 120 Q190 130 200 120' stroke='%23991b1b' stroke-width='2' fill='none'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23991b1b'%3ESevere NPDR%3C/text%3E%3C/svg%3E",
    realImage: "/sample-images/grade3_sample.png"
  },
  {
    id: 4,
    name: "Proliferative Diabetic Retinopathy",
    shortName: "PDR",
    description: "Most advanced stage with new blood vessel growth (neovascularization) and potential for severe vision loss. May include retinal detachment and vitreous hemorrhage.",
    clinicalNote: "Proliferative diabetic retinopathy detected. Immediate treatment required to prevent vision loss.",
    severity: "Critical",
    color: "red",
    icon: AlertTriangle,
    prevalence: "~2%",
    riskLevel: "Critical",
    features: [
      "Neovascularization (new blood vessels)",
      "Vitreous hemorrhage risk",
      "Retinal detachment potential",
      "Severe vision threatening complications"
    ],
    recommendation: "Immediate ophthalmologic intervention, likely requiring laser treatment or surgery",
    image: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23450a0a'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23991b1b' stroke='%23dc2626' stroke-width='3'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23dc2626'/%3E%3Ccircle cx='140' cy='100' r='6' fill='%23450a0a'/%3E%3Ccircle cx='260' cy='110' r='5' fill='%23450a0a'/%3E%3Ccircle cx='160' cy='200' r='5' fill='%23450a0a'/%3E%3Ccircle cx='240' cy='190' r='4' fill='%23450a0a'/%3E%3Cpath d='M150 130 Q170 140 190 130 Q210 120 230 130' stroke='%23dc2626' stroke-width='3' fill='none'/%3E%3Cpath d='M170 170 Q190 180 210 170 Q230 160 250 170' stroke='%23dc2626' stroke-width='3' fill='none'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23dc2626'%3EPDR - Critical%3C/text%3E%3C/svg%3E",
    realImage: "/sample-images/grade4_sample.png"
  }
]

export default function SeverityCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(true)
  const [flippedCards, setFlippedCards] = useState({})

  const toggleFlip = (levelId) => {
    setFlippedCards(prev => ({
      ...prev,
      [levelId]: !prev[levelId]
    }))
  }

  // Auto-advance carousel
  useEffect(() => {
    if (!isAutoPlaying) return

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % severityLevels.length)
    }, 5000) // 5 seconds per slide

    return () => clearInterval(interval)
  }, [isAutoPlaying])

  // Reset flip state when carousel changes
  useEffect(() => {
    setFlippedCards({})
  }, [currentIndex])

  const goToSlide = (index) => {
    setCurrentIndex(index)
    setIsAutoPlaying(false)
    // Resume auto-play after 10 seconds
    setTimeout(() => setIsAutoPlaying(true), 10000)
  }

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev - 1 + severityLevels.length) % severityLevels.length)
    setIsAutoPlaying(false)
    setTimeout(() => setIsAutoPlaying(true), 10000)
  }

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % severityLevels.length)
    setIsAutoPlaying(false)
    setTimeout(() => setIsAutoPlaying(true), 10000)
  }

  const currentLevel = severityLevels[currentIndex]
  const IconComponent = currentLevel.icon

  const getSeverityColorClasses = (color) => {
    const colors = {
      green: {
        bg: 'from-green-500 to-emerald-600',
        text: 'text-green-700',
        border: 'border-green-200',
        bgLight: 'bg-green-50'
      },
      yellow: {
        bg: 'from-yellow-500 to-amber-600',
        text: 'text-yellow-700',
        border: 'border-yellow-200',
        bgLight: 'bg-yellow-50'
      },
      orange: {
        bg: 'from-orange-500 to-red-500',
        text: 'text-orange-700',
        border: 'border-orange-200',
        bgLight: 'bg-orange-50'
      },
      red: {
        bg: 'from-red-500 to-red-700',
        text: 'text-red-700',
        border: 'border-red-200',
        bgLight: 'bg-red-50'
      }
    }
    return colors[color] || colors.green
  }

  const colorClasses = getSeverityColorClasses(currentLevel.color)

  return (
    <section className="severity-carousel py-20 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Diabetic Retinopathy Severity Levels
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-6">
            Understanding the progression of diabetic retinopathy through AI-powered classification
          </p>
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Grade 0: No DR</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span>Grade 1: Mild</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
              <span>Grade 2: Moderate</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span>Grade 3: Severe</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-red-700 rounded-full"></div>
              <span>Grade 4: PDR</span>
            </div>
          </div>
        </div>

        <div className="relative bg-white rounded-3xl shadow-2xl overflow-hidden">
          {/* Main Carousel Content */}
          <div className="grid lg:grid-cols-2 gap-0 min-h-[600px]">
            {/* Image Section - Flip Card */}
            <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center p-8">
              <div className="relative w-full max-w-md">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-100/20 to-indigo-100/20 rounded-2xl transform rotate-3"></div>
                
                {/* Flip Card Container */}
                <div 
                  key={currentLevel.id}
                  className="relative w-full h-80 cursor-pointer perspective-1000"
                  onClick={() => toggleFlip(currentLevel.id)}
                >
                  <div className={`flip-card-inner w-full h-full transition-transform duration-700 preserve-3d ${flippedCards[currentLevel.id] ? 'rotate-y-180' : ''}`}>
                    {/* Front - Infographic */}
                    <div className="flip-card-face flip-card-front absolute inset-0 w-full h-full backface-hidden">
                      <img 
                        src={currentLevel.image} 
                        alt={`${currentLevel.name} - Infographic`}
                        className="w-full h-full object-cover rounded-2xl shadow-xl border-4 border-white"
                      />
                      <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm flex items-center space-x-2">
                        <RotateCcw className="w-4 h-4" />
                        <span>Click to see real image</span>
                      </div>
                    </div>
                    
                    {/* Back - Real Medical Image */}
                    <div className="flip-card-face flip-card-back absolute inset-0 w-full h-full backface-hidden rotate-y-180">
                      <img 
                        src={currentLevel.realImage} 
                        alt={`${currentLevel.name} - Real Medical Case`}
                        className="w-full h-full object-cover rounded-2xl shadow-xl border-4 border-white"
                      />
                      <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm flex items-center space-x-2">
                        <RotateCcw className="w-4 h-4" />
                        <span>Click to see infographic</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="absolute top-4 left-4">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${colorClasses.bgLight} ${colorClasses.text} border ${colorClasses.border} shadow-md backdrop-blur-sm`}>
                    Grade {currentLevel.id}/4
                  </div>
                </div>
                <div className="absolute bottom-4 right-4">
                  <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses.bg} rounded-full flex items-center justify-center shadow-lg backdrop-blur-sm border-2 border-white`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                </div>
                {/* Floating elements for visual appeal */}
                <div className="absolute -top-2 -right-2 w-4 h-4 bg-blue-500 rounded-full opacity-70 animate-pulse"></div>
                <div className="absolute -bottom-2 -left-2 w-3 h-3 bg-indigo-500 rounded-full opacity-60 animate-pulse delay-75"></div>
              </div>
            </div>

            {/* Content Section */}
            <div className="p-8 lg:p-12 flex flex-col justify-center">
              <div className="space-y-6">
                {/* Header */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold ${colorClasses.bgLight} ${colorClasses.text} border ${colorClasses.border}`}>
                      <div className={`w-2 h-2 bg-gradient-to-r ${colorClasses.bg} rounded-full mr-2`}></div>
                      {currentLevel.severity}
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-500 mb-1">Severity Progress</div>
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full bg-gradient-to-r ${colorClasses.bg} transition-all duration-1000`}
                          style={{ width: `${((currentLevel.id + 1) / 5) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <h3 className="text-2xl lg:text-3xl font-bold text-gray-900 mb-2">
                    {currentLevel.name}
                  </h3>
                  <p className="text-lg text-gray-600 leading-relaxed">
                    {currentLevel.description}
                  </p>
                </div>

                {/* Features */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Key Features:</h4>
                  <ul className="space-y-2">
                    {currentLevel.features.map((feature, index) => (
                      <li key={index} className="flex items-start space-x-3">
                        <div className={`w-2 h-2 bg-gradient-to-r ${colorClasses.bg} rounded-full mt-2 flex-shrink-0`}></div>
                        <span className="text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Statistics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className="text-lg font-bold text-gray-900">{currentLevel.prevalence}</div>
                    <div className="text-xs text-gray-600">Prevalence</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded-lg">
                    <div className={`text-lg font-bold ${colorClasses.text}`}>{currentLevel.riskLevel}</div>
                    <div className="text-xs text-gray-600">Risk Level</div>
                  </div>
                </div>

                {/* Clinical Note */}
                <div className={`p-4 rounded-xl ${colorClasses.bgLight} border ${colorClasses.border}`}>
                  <h4 className={`font-semibold ${colorClasses.text} mb-2`}>Clinical Recommendation:</h4>
                  <p className={`text-sm ${colorClasses.text}`}>
                    {currentLevel.recommendation}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="absolute top-1/2 left-4 transform -translate-y-1/2">
            <button
              onClick={goToPrevious}
              className="w-12 h-12 bg-white/90 hover:bg-white rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110"
            >
              <ChevronLeft className="w-6 h-6 text-gray-700" />
            </button>
          </div>
          <div className="absolute top-1/2 right-4 transform -translate-y-1/2">
            <button
              onClick={goToNext}
              className="w-12 h-12 bg-white/90 hover:bg-white rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110"
            >
              <ChevronRight className="w-6 h-6 text-gray-700" />
            </button>
          </div>

          {/* Slide Indicators */}
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2">
            <div className="flex space-x-3">
              {severityLevels.map((_, index) => (
                <button
                  key={index}
                  onClick={() => goToSlide(index)}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    index === currentIndex 
                      ? `bg-gradient-to-r ${colorClasses.bg} scale-125` 
                      : 'bg-gray-300 hover:bg-gray-400'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Auto-play indicator */}
          <div className="absolute top-4 right-4">
            <div className={`w-2 h-2 rounded-full ${isAutoPlaying ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
          </div>
        </div>

        {/* Quick Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-8">
          {severityLevels.map((level, index) => {
            const isActive = index === currentIndex
            const colors = getSeverityColorClasses(level.color)
            const Icon = level.icon
            
            return (
              <button
                key={level.id}
                onClick={() => goToSlide(index)}
                className={`p-4 rounded-xl transition-all duration-300 ${
                  isActive 
                    ? `${colors.bgLight} border-2 ${colors.border} scale-105 shadow-lg` 
                    : 'bg-white border border-gray-200 hover:border-gray-300 hover:shadow-md'
                }`}
              >
                <div className="text-center">
                  <div className={`w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center ${
                    isActive ? `bg-gradient-to-br ${colors.bg}` : 'bg-gray-100'
                  }`}>
                    <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-gray-600'}`} />
                  </div>
                  <div className={`text-xs font-medium ${isActive ? colors.text : 'text-gray-600'}`}>
                    {level.shortName}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Grade {level.id}
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Educational Information */}
        <div className="mt-12 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 border border-blue-100">
          <div className="max-w-4xl mx-auto">
            <h3 className="text-xl font-bold text-gray-900 mb-4 text-center">
              Understanding Diabetic Retinopathy Classification
            </h3>
            <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-700">
              <div>
                <h4 className="font-semibold text-blue-900 mb-2">AI-Powered Detection</h4>
                <p>Our deep learning models analyze retinal images to identify subtle patterns and features that indicate different stages of diabetic retinopathy progression.</p>
              </div>
              <div>
                <h4 className="font-semibold text-blue-900 mb-2">Clinical Significance</h4>
                <p>Early detection and classification are crucial for preventing vision loss. Each grade requires different monitoring intervals and treatment approaches.</p>
              </div>
              <div>
                <h4 className="font-semibold text-blue-900 mb-2">Research Impact</h4>
                <p>This classification system helps standardize diagnosis and enables large-scale screening programs to identify at-risk patients efficiently.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}