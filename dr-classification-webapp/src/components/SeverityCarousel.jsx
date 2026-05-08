import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle, ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'

const severityLevels = [
  {
    id: 0,
    name: 'No Diabetic Retinopathy',
    shortName: 'No DR',
    description:
      'Normal retinal appearance with no signs of diabetic retinopathy. Blood vessels appear healthy with no microaneurysms, hemorrhages, or other pathological changes.',
    severity: 'Normal',
    color: 'green',
    icon: CheckCircle,
    prevalence: '~60%',
    riskLevel: 'Low',
    features: [
      'Clear, well-defined blood vessels',
      'No microaneurysms or hemorrhages',
      'Normal optic disc appearance',
      'Healthy macula region'
    ],
    recommendation: 'Annual eye examination recommended for diabetic patients.',
    image:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23f0fdf4'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23dcfce7' stroke='%2316a34a' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%2316a34a'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23166534'%3ENo DR - Normal Retina%3C/text%3E%3C/svg%3E",
    realImage: '/sample-images/grade0_sample.png'
  },
  {
    id: 1,
    name: 'Mild Non-proliferative DR',
    shortName: 'Mild NPDR',
    description:
      'Early signs of diabetic retinopathy with microaneurysms present. These small bulges in blood vessel walls are the earliest detectable changes in diabetic retinopathy.',
    severity: 'Mild',
    color: 'yellow',
    icon: AlertTriangle,
    prevalence: '~25%',
    riskLevel: 'Low-Moderate',
    features: [
      'Microaneurysms (small red dots)',
      'Possible small hemorrhages',
      'Blood vessels still largely intact',
      'No significant vision impact'
    ],
    recommendation: 'Follow-up examination in 6-12 months, with stronger diabetes management support.',
    image:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fffbeb'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fef3c7' stroke='%23d97706' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23d97706'/%3E%3Ccircle cx='170' cy='130' r='3' fill='%23dc2626'/%3E%3Ccircle cx='230' cy='140' r='2' fill='%23dc2626'/%3E%3Ccircle cx='190' cy='180' r='2' fill='%23dc2626'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%2392400e'%3EMild NPDR%3C/text%3E%3C/svg%3E",
    realImage: '/sample-images/grade1_sample.png'
  },
  {
    id: 2,
    name: 'Moderate Non-proliferative DR',
    shortName: 'Moderate NPDR',
    description:
      'Progressive diabetic retinopathy with more widespread microaneurysms, hemorrhages, and possible hard exudates. Blood vessel blockages may begin to occur.',
    severity: 'Moderate',
    color: 'orange',
    icon: AlertTriangle,
    prevalence: '~10%',
    riskLevel: 'Moderate',
    features: [
      'Multiple microaneurysms',
      'Retinal hemorrhages',
      'Hard exudates (lipid deposits)',
      'Early signs of blood vessel blockage'
    ],
    recommendation: 'Follow-up every 3-6 months and consider referral to a retinal specialist.',
    image:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fff7ed'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fed7aa' stroke='%23ea580c' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23ea580c'/%3E%3Ccircle cx='160' cy='120' r='4' fill='%23dc2626'/%3E%3Ccircle cx='240' cy='130' r='3' fill='%23dc2626'/%3E%3Ccircle cx='180' cy='180' r='3' fill='%23dc2626'/%3E%3Ccircle cx='220' cy='170' r='2' fill='%23dc2626'/%3E%3Crect x='170' y='160' width='6' height='3' fill='%23fbbf24'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23c2410c'%3EModerate NPDR%3C/text%3E%3C/svg%3E",
    realImage: '/sample-images/grade2_sample.png'
  },
  {
    id: 3,
    name: 'Severe Non-proliferative DR',
    shortName: 'Severe NPDR',
    description:
      'Advanced non-proliferative diabetic retinopathy with extensive hemorrhages, microaneurysms, and significant blood vessel blockages. High risk of progression to proliferative DR.',
    severity: 'Severe',
    color: 'red',
    icon: AlertTriangle,
    prevalence: '~3%',
    riskLevel: 'High',
    features: [
      'Extensive hemorrhages and microaneurysms',
      'Significant cotton wool spots',
      'Venous beading',
      'Intraretinal microvascular abnormalities'
    ],
    recommendation: 'Immediate referral to a retinal specialist, with close follow-up in 2-4 months.',
    image:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23fef2f2'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23fecaca' stroke='%23dc2626' stroke-width='2'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23dc2626'/%3E%3Ccircle cx='150' cy='110' r='5' fill='%23991b1b'/%3E%3Ccircle cx='250' cy='120' r='4' fill='%23991b1b'/%3E%3Ccircle cx='170' cy='190' r='4' fill='%23991b1b'/%3E%3Ccircle cx='230' cy='180' r='3' fill='%23991b1b'/%3E%3Crect x='160' y='140' width='8' height='4' fill='%23fbbf24'/%3E%3Crect x='220' y='160' width='6' height='3' fill='%23fbbf24'/%3E%3Cpath d='M180 120 Q190 130 200 120' stroke='%23991b1b' stroke-width='2' fill='none'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23991b1b'%3ESevere NPDR%3C/text%3E%3C/svg%3E",
    realImage: '/sample-images/grade3_sample.png'
  },
  {
    id: 4,
    name: 'Proliferative Diabetic Retinopathy',
    shortName: 'PDR',
    description:
      'Most advanced stage with new blood vessel growth (neovascularization) and potential for severe vision loss. May include retinal detachment and vitreous hemorrhage.',
    severity: 'Critical',
    color: 'red',
    icon: AlertTriangle,
    prevalence: '~2%',
    riskLevel: 'Critical',
    features: [
      'Neovascularization (new blood vessels)',
      'Vitreous hemorrhage risk',
      'Retinal detachment potential',
      'Severe vision threatening complications'
    ],
    recommendation: 'Immediate ophthalmologic intervention, often requiring laser treatment or surgery.',
    image:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23450a0a'/%3E%3Ccircle cx='200' cy='150' r='80' fill='%23991b1b' stroke='%23dc2626' stroke-width='3'/%3E%3Ccircle cx='200' cy='150' r='20' fill='%23dc2626'/%3E%3Ccircle cx='140' cy='100' r='6' fill='%23450a0a'/%3E%3Ccircle cx='260' cy='110' r='5' fill='%23450a0a'/%3E%3Ccircle cx='160' cy='200' r='5' fill='%23450a0a'/%3E%3Ccircle cx='240' cy='190' r='4' fill='%23450a0a'/%3E%3Cpath d='M150 130 Q170 140 190 130 Q210 120 230 130' stroke='%23dc2626' stroke-width='3' fill='none'/%3E%3Cpath d='M170 170 Q190 180 210 170 Q230 160 250 170' stroke='%23dc2626' stroke-width='3' fill='none'/%3E%3Ctext x='200' y='250' text-anchor='middle' font-family='Arial' font-size='14' fill='%23dc2626'%3EPDR - Critical%3C/text%3E%3C/svg%3E",
    realImage: '/sample-images/grade4_sample.png'
  }
]

export default function SeverityCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(true)
  const [showRealImages, setShowRealImages] = useState(false)

  useEffect(() => {
    if (!isAutoPlaying) return

    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % severityLevels.length)
    }, 5000)

    return () => clearInterval(interval)
  }, [isAutoPlaying])

  const pauseAutoplay = () => {
    setIsAutoPlaying(false)
    window.setTimeout(() => setIsAutoPlaying(true), 10000)
  }

  const goToSlide = (index) => {
    setCurrentIndex(index)
    pauseAutoplay()
  }

  const goToPrevious = () => {
    setCurrentIndex((prev) => (prev - 1 + severityLevels.length) % severityLevels.length)
    pauseAutoplay()
  }

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % severityLevels.length)
    pauseAutoplay()
  }

  const toggleImageMode = () => {
    setShowRealImages((prev) => !prev)
  }

  const currentLevel = severityLevels[currentIndex]
  const IconComponent = currentLevel.icon

  const getSeverityColorClasses = (color) => {
    const colors = {
      green: {
        bg: 'from-emerald-500 to-teal-600',
        text: 'text-emerald-700',
        border: 'border-emerald-200',
        bgLight: 'bg-emerald-50'
      },
      yellow: {
        bg: 'from-amber-400 to-orange-500',
        text: 'text-amber-700',
        border: 'border-amber-200',
        bgLight: 'bg-amber-50'
      },
      orange: {
        bg: 'from-orange-500 to-red-500',
        text: 'text-orange-700',
        border: 'border-orange-200',
        bgLight: 'bg-orange-50'
      },
      red: {
        bg: 'from-rose-500 to-red-700',
        text: 'text-rose-700',
        border: 'border-rose-200',
        bgLight: 'bg-rose-50'
      }
    }

    return colors[color] || colors.green
  }

  const colorClasses = getSeverityColorClasses(currentLevel.color)

  return (
    <section className="severity-carousel border-y border-slate-200/70 bg-white/70 py-20 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4">
        <div className="mb-12 max-w-3xl">
          <span className="eyebrow">Severity reference</span>
          <h2 className="section-title mt-4">Diabetic retinopathy grading overview</h2>
          <p className="section-copy mt-4">
            Reference guide to the retinal findings and severity levels commonly used in diabetic
            retinopathy grading.
          </p>
        </div>

        <div className="surface-card overflow-hidden p-0">
          <div className="grid gap-0 lg:grid-cols-[0.92fr_1.08fr]">
            <div className="border-b border-slate-200/80 bg-slate-50/80 p-6 lg:border-b-0 lg:border-r">
              <div className="mb-5 flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                    Image view
                  </p>
                  <p className="mt-1 text-sm text-slate-500">
                    Switch between educational illustrations and sample medical images.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={toggleImageMode}
                  className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold transition-colors ${
                    showRealImages
                      ? 'border-cyan-200 bg-cyan-50 text-cyan-800'
                      : 'border-slate-200 bg-white text-slate-600'
                  }`}
                >
                  <RotateCcw className="h-4 w-4" />
                  {showRealImages ? 'Real image mode' : 'Illustration mode'}
                </button>
              </div>

              <div className="relative mx-auto max-w-md">
                <div className="absolute inset-0 rounded-[30px] bg-gradient-to-br from-cyan-100/50 to-teal-100/40 blur-2xl" />
                <div
                  key={currentLevel.id}
                  className="relative h-80 w-full cursor-pointer perspective-1000"
                  onClick={toggleImageMode}
                >
                  <div className={`flip-card-inner h-full w-full transition-transform duration-700 preserve-3d ${showRealImages ? 'rotate-y-180' : ''}`}>
                    <div className="flip-card-face absolute inset-0 backface-hidden">
                      <img
                        src={currentLevel.image}
                        alt={`${currentLevel.name} infographic`}
                        className="h-full w-full rounded-[30px] border-4 border-white object-cover shadow-xl"
                      />
                    </div>
                    <div className="flip-card-face flip-card-back absolute inset-0 backface-hidden rotate-y-180">
                      <img
                        src={currentLevel.realImage}
                        alt={`${currentLevel.name} sample medical image`}
                        className="h-full w-full rounded-[30px] border-4 border-white object-cover shadow-xl"
                      />
                    </div>
                  </div>
                </div>

                <div className="absolute left-4 top-4 rounded-full border border-white/70 bg-white/90 px-3 py-1 text-sm font-semibold text-slate-700 shadow-sm">
                  Grade {currentLevel.id}
                </div>
                <div className="absolute bottom-4 left-4 rounded-full bg-black/70 px-3 py-1 text-sm text-white">
                  Click image to switch view
                </div>
                <div className="absolute bottom-4 right-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/80 bg-white/90 shadow-sm">
                  <IconComponent className={`h-6 w-6 ${colorClasses.text}`} />
                </div>
              </div>
            </div>

            <div className="p-6 lg:p-8">
              <div className="flex flex-col gap-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <div className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold ${colorClasses.bgLight} ${colorClasses.text} ${colorClasses.border}`}>
                      <span className={`h-2.5 w-2.5 rounded-full bg-gradient-to-r ${colorClasses.bg}`} />
                      {currentLevel.severity}
                    </div>
                    <h3 className="mt-4 text-3xl font-semibold tracking-tight text-slate-950">
                      {currentLevel.name}
                    </h3>
                    <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
                      {currentLevel.description}
                    </p>
                  </div>

                  <div className="rounded-3xl border border-slate-200 bg-slate-50 px-4 py-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                      Auto-rotate
                    </p>
                    <div className="mt-3 flex items-center gap-3">
                      <span className={`h-2.5 w-2.5 rounded-full ${isAutoPlaying ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                      <span className="text-sm font-medium text-slate-600">
                        {isAutoPlaying ? 'Running' : 'Paused'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-3xl border border-slate-200 bg-slate-50/70 p-5">
                    <p className="text-sm font-medium text-slate-500">Prevalence</p>
                    <p className="mt-2 text-2xl font-semibold text-slate-950">{currentLevel.prevalence}</p>
                  </div>
                  <div className="rounded-3xl border border-slate-200 bg-slate-50/70 p-5">
                    <p className="text-sm font-medium text-slate-500">Risk level</p>
                    <p className={`mt-2 text-2xl font-semibold ${colorClasses.text}`}>{currentLevel.riskLevel}</p>
                  </div>
                </div>

                <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6">
                  <div className="mb-4 flex items-center justify-between gap-4">
                    <h4 className="text-lg font-semibold text-slate-950">Key features</h4>
                    <div className="w-24 rounded-full bg-slate-200/80">
                      <div
                        className={`h-2 rounded-full bg-gradient-to-r ${colorClasses.bg}`}
                        style={{ width: `${((currentLevel.id + 1) / 5) * 100}%` }}
                      />
                    </div>
                  </div>

                  <ul className="grid gap-3 md:grid-cols-2">
                    {currentLevel.features.map((feature) => (
                      <li key={feature} className="flex items-start gap-3 rounded-2xl bg-slate-50 px-4 py-3">
                        <span className={`mt-2 h-2 w-2 shrink-0 rounded-full bg-gradient-to-r ${colorClasses.bg}`} />
                        <span className="text-sm leading-6 text-slate-600">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className={`rounded-[28px] border p-6 ${colorClasses.border} ${colorClasses.bgLight}`}>
                  <p className={`text-sm font-semibold uppercase tracking-[0.18em] ${colorClasses.text}`}>
                    Clinical recommendation
                  </p>
                  <p className={`mt-3 text-base leading-7 ${colorClasses.text}`}>{currentLevel.recommendation}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="border-t border-slate-200/80 bg-slate-50/80 px-4 py-5 md:px-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={goToPrevious}
                  className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition-transform hover:-translate-y-0.5"
                >
                  <ChevronLeft className="h-5 w-5" />
                </button>
                <button
                  type="button"
                  onClick={goToNext}
                  className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-sm transition-transform hover:-translate-y-0.5"
                >
                  <ChevronRight className="h-5 w-5" />
                </button>
              </div>

              <div className="grid flex-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
                {severityLevels.map((level, index) => {
                  const colors = getSeverityColorClasses(level.color)
                  const isActive = index === currentIndex
                  const Icon = level.icon

                  return (
                    <button
                      key={level.id}
                      type="button"
                      onClick={() => goToSlide(index)}
                      className={`rounded-2xl border px-4 py-4 text-left transition-all duration-200 ${
                        isActive
                          ? `${colors.bgLight} ${colors.border} shadow-sm`
                          : 'border-slate-200 bg-white hover:border-slate-300'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`flex h-9 w-9 items-center justify-center rounded-2xl ${
                            isActive ? `bg-gradient-to-br ${colors.bg} text-white` : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                        </div>
                        <div>
                          <p className={`text-sm font-semibold ${isActive ? colors.text : 'text-slate-800'}`}>
                            {level.shortName}
                          </p>
                          <p className="text-xs text-slate-500">Grade {level.id}</p>
                        </div>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 grid gap-5 md:grid-cols-3">
          <div className="surface-card">
            <h3 className="text-lg font-semibold text-slate-950">Schedule regular examinations</h3>
            <p className="mt-3 text-sm leading-7 text-slate-600">
              People with diabetes benefit from annual comprehensive eye examinations, which can surface
              retinal changes before symptoms become obvious.
            </p>
          </div>
          <div className="surface-card">
            <h3 className="text-lg font-semibold text-slate-950">Maintain systemic control</h3>
            <p className="mt-3 text-sm leading-7 text-slate-600">
              Blood glucose, cholesterol, and blood pressure management remain central to limiting
              disease progression and preserving retinal health.
            </p>
          </div>
          <div className="surface-card">
            <h3 className="text-lg font-semibold text-slate-950">Escalate promptly when needed</h3>
            <p className="mt-3 text-sm leading-7 text-slate-600">
              When symptoms or advanced findings appear, timely referral and intervention are critical to
              protecting long-term vision.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
