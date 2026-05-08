import { useState } from 'react'
import {
  Activity,
  ArrowRight,
  BarChart3,
  Brain,
  CalendarCheck,
  ChevronRight,
  Eye,
  EyeOff,
  Globe,
  HeartPulse,
  ShieldCheck,
  Sparkles,
  Users
} from 'lucide-react'
import SeverityCarousel from './SeverityCarousel'

const thesisTitle = 'Thesis: Integrating LSTL with EfficientNetB0 for Diabetic Retinopathy Classification'

const overviewStats = [
  { label: 'People affected worldwide', value: '100M+', icon: Globe },
  { label: 'Diabetic patients developing DR', value: '1 in 3', icon: Eye },
  { label: 'Preventable vision loss with early detection', value: '90%', icon: ShieldCheck },
  { label: 'Recommended screening cadence', value: 'Annual', icon: CalendarCheck }
]

const features = [
  {
    icon: Globe,
    title: 'A growing global burden',
    description:
      'Diabetic retinopathy remains one of the leading causes of preventable vision loss, with many cases still detected only after damage has advanced.'
  },
  {
    icon: EyeOff,
    title: 'Vision changes can progress quietly',
    description:
      'Retinal vessels may leak, close off, or form abnormal growth long before symptoms feel urgent, making routine screening clinically important.'
  },
  {
    icon: HeartPulse,
    title: 'Early identification changes outcomes',
    description:
      'Timely detection supports referral, monitoring, and intervention before more severe retinal complications threaten long-term sight.'
  }
]

const models = [
  {
    name: 'EfficientNet-B0 + LSTL',
    label: 'Recommended Thesis Variant',
    description:
      'Integrates a Local-Global Spatially-Aware Transformer Layer while preserving the EfficientNet-B0 backbone.',
    accent: 'from-teal-600 to-cyan-500'
  },
  {
    name: 'EfficientNet-B0 Baseline',
    label: 'Reference Model',
    description:
      'The baseline architecture serves as the direct reference point for comparing the impact of the thesis enhancements.',
    accent: 'from-slate-600 to-slate-500'
  },
  {
    name: 'EfficientNet-B0 + CBAM',
    label: 'Attention-Based Comparator',
    description:
      'The CBAM variant provides an alternative enhancement path centered on attention-guided feature refinement.',
    accent: 'from-sky-600 to-cyan-500'
  }
]

const academicContext = [
  'A thesis presented to the Department of Computer Science',
  'In partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Science',
  'Researchers: Ruzel Alano, Giancarlo Bajit, Angel Labuyo, Kevin Lugue, and Francesca Vega'
]

export default function LandingPage({ onStartTesting }) {
  const [currentFeature, setCurrentFeature] = useState(0)

  const scrollToSeverity = () => {
    document.querySelector('.severity-carousel')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <section className="relative overflow-hidden px-4 pb-16 pt-8 md:pt-12">
        <div className="hero-glow absolute inset-x-4 top-0 h-[34rem] rounded-[40px] border border-white/40 muted-grid md:inset-x-8" />
        <div className="relative mx-auto max-w-7xl">
          <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
            <div className="space-y-8">
              <div className="space-y-5">
                <span className="eyebrow">Research-use screening interface</span>
                <h1 className="max-w-5xl text-4xl font-semibold leading-[1.05] tracking-tight text-slate-950 md:text-5xl lg:text-6xl">
                  {thesisTitle}
                </h1>
                <p className="max-w-3xl text-lg leading-8 text-slate-600 md:text-xl">
                  Web-based retinal fundus image classification across baseline, CBAM, and
                  LSTL-enhanced EfficientNet-B0 variants for research and educational use.
                </p>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row">
                <button onClick={onStartTesting} className="btn-primary px-6 py-3.5 text-base">
                  Start Testing
                  <ArrowRight className="h-4 w-4" />
                </button>
                <button onClick={scrollToSeverity} className="btn-secondary px-6 py-3.5 text-base">
                  View Severity Guide
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <div className="stat-card">
                  <p className="text-sm font-medium text-slate-500">Primary focus</p>
                  <p className="mt-2 text-lg font-semibold text-slate-950">Clinical readability</p>
                </div>
                <div className="stat-card">
                  <p className="text-sm font-medium text-slate-500">Featured variant</p>
                  <p className="mt-2 text-lg font-semibold text-slate-950">EfficientNet-B0 + LSTL</p>
                </div>
                <div className="stat-card">
                  <p className="text-sm font-medium text-slate-500">Use context</p>
                  <p className="mt-2 text-lg font-semibold text-slate-950">Research and education</p>
                </div>
              </div>
            </div>

            <div className="surface-card p-0 overflow-hidden">
              <div className="border-b border-slate-200/80 px-6 py-5">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-600 to-cyan-500 text-white shadow-sm">
                    <Eye className="h-6 w-6" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
                      Featured analysis
                    </p>
                    <h2 className="mt-1 text-xl font-semibold text-slate-950">
                      Retinal image analysis
                    </h2>
                  </div>
                </div>
              </div>

              <div className="space-y-6 px-6 py-6">
                <div className="rounded-[28px] border border-teal-100 bg-gradient-to-br from-teal-50 via-white to-cyan-50 p-5">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700">
                        Recommended model
                      </p>
                      <h3 className="mt-2 text-2xl font-semibold text-slate-950">EfficientNet-B0 + LSTL</h3>
                    </div>
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-teal-700 shadow-sm">
                      <Sparkles className="h-5 w-5" />
                    </div>
                  </div>
                  <p className="mt-4 leading-7 text-slate-600">
                    LSTL is presented as the recommended model, while Baseline and CBAM remain
                    available for comparison.
                  </p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-3xl border border-slate-200 bg-white/90 p-5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700">
                        <Brain className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-950">Three model variants</p>
                        <p className="text-sm text-slate-500">Baseline, CBAM, and LSTL</p>
                      </div>
                    </div>
                  </div>
                  <div className="rounded-3xl border border-slate-200 bg-white/90 p-5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
                        <Activity className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-950">Guided testing flow</p>
                        <p className="text-sm text-slate-500">Upload, analyze, compare</p>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-sm leading-7 text-slate-500">
                  Designed for research and educational interpretation, not production clinical use.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="px-4 py-6">
        <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-2 xl:grid-cols-4">
          {overviewStats.map((stat) => (
            <div key={stat.label} className="stat-card">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm leading-6 text-slate-500">{stat.label}</p>
                  <p className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">{stat.value}</p>
                </div>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700">
                  <stat.icon className="h-5 w-5" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="px-4 py-16">
        <div className="mx-auto max-w-7xl">
          <div className="mb-12 max-w-3xl">
            <span className="eyebrow">Why it matters</span>
            <h2 className="section-title mt-4">Diabetic retinopathy screening benefits from timely, readable decision support</h2>
            <p className="section-copy mt-4">
              Early recognition and clear grading support timely follow-up, referral, and ongoing
              monitoring for people at risk of diabetic retinopathy.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className={`feature-card ${currentFeature === index ? 'feature-card-active' : ''}`}
                onMouseEnter={() => setCurrentFeature(index)}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ${
                      currentFeature === index ? 'bg-gradient-to-br from-teal-600 to-cyan-500 text-white' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    <feature.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-slate-950">{feature.title}</h3>
                    <p className="mt-3 leading-7 text-slate-600">{feature.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <SeverityCarousel />

      <section className="px-4 py-20">
        <div className="mx-auto max-w-7xl">
          <div className="mb-12 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <span className="eyebrow">Model overview</span>
              <h2 className="section-title mt-4">Three model variants for diabetic retinopathy classification</h2>
              <p className="section-copy mt-4">
                Baseline, CBAM, and LSTL variants are available for comparison, with LSTL presented
                as the recommended primary model.
              </p>
            </div>
            <button onClick={onStartTesting} className="btn-primary w-fit">
              Open testing workspace
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="model-spotlight">
              <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                <div className="max-w-2xl">
                  <span className="eyebrow">Recommended thesis variant</span>
                  <h3 className="mt-4 text-3xl font-semibold tracking-tight text-slate-950">
                    {models[0].name}
                  </h3>
                  <p className="mt-4 text-lg leading-8 text-slate-600">{models[0].description}</p>
                </div>
                <div className="flex h-16 w-16 items-center justify-center rounded-[22px] bg-gradient-to-br from-teal-600 to-cyan-500 text-white shadow-sm">
                  <Brain className="h-7 w-7" />
                </div>
              </div>

              <div className="mt-8 grid gap-4 md:grid-cols-3">
                <div className="rounded-3xl border border-teal-100 bg-white/80 p-5">
                  <p className="text-sm font-medium text-slate-500">Role</p>
                  <p className="mt-2 font-semibold text-slate-950">Primary model in the UI</p>
                </div>
                <div className="rounded-3xl border border-teal-100 bg-white/80 p-5">
                  <p className="text-sm font-medium text-slate-500">Objective</p>
                  <p className="mt-2 font-semibold text-slate-950">Improved feature representation</p>
                </div>
                <div className="rounded-3xl border border-teal-100 bg-white/80 p-5">
                  <p className="text-sm font-medium text-slate-500">Context</p>
                  <p className="mt-2 font-semibold text-slate-950">Research and educational evaluation</p>
                </div>
              </div>
            </div>

            <div className="grid gap-6">
              {models.slice(1).map((model) => (
                <div key={model.name} className="model-secondary">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{model.label}</p>
                      <h3 className="mt-2 text-xl font-semibold text-slate-950">{model.name}</h3>
                    </div>
                    <div className={`flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br ${model.accent} text-white`}>
                      <BarChart3 className="h-5 w-5" />
                    </div>
                  </div>
                  <p className="mt-4 leading-7 text-slate-600">{model.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="px-4 pb-20">
        <div className="mx-auto max-w-7xl">
          <div className="surface-card">
            <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
              <div>
                <span className="eyebrow">Academic context</span>
                <h2 className="section-title mt-4">Academic and thesis context</h2>
                <p className="section-copy mt-4">
                  University, authorship, and degree context are included here as supporting information
                  for the project and its research setting.
                </p>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-3xl border border-slate-200 bg-white/80 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
                    <Users className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm leading-7 text-slate-600">{academicContext[2]}</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/80 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
                    <Brain className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm leading-7 text-slate-600">{academicContext[0]}</p>
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white/80 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <p className="mt-4 text-sm leading-7 text-slate-600">{academicContext[1]}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-slate-200/80 bg-white/70 px-4 py-10 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-600 to-cyan-500 text-white">
              <Eye className="h-5 w-5" />
            </div>
            <div>
              <p className="font-semibold text-slate-950">{thesisTitle}</p>
              <p className="text-sm text-slate-500">For research and educational use only.</p>
            </div>
          </div>
          <p className="max-w-xl text-sm leading-6 text-slate-500 md:text-right">
            Supports comparison of baseline, CBAM, and LSTL-enhanced diabetic retinopathy
            classification models.
          </p>
        </div>
      </footer>
    </div>
  )
}
