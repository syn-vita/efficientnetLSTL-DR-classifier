import { Activity, AlertTriangle, BarChart3, CheckCircle, Eye, ShieldAlert } from 'lucide-react'
import { drClasses } from '../config/models'

export default function ResultDisplay({ prediction, isLoading, modelName, error }) {
  if (isLoading) {
    return (
      <div className="diagnostic-card">
        <div className="flex items-center justify-center py-14">
          <div className="text-center">
            <div className="relative mx-auto mb-5 h-14 w-14">
              <div className="h-14 w-14 animate-spin rounded-full border-4 border-cyan-100 border-t-cyan-600" />
              <Activity className="absolute left-4 top-4 h-6 w-6 text-cyan-700" />
            </div>
            <h3 className="text-xl font-semibold text-slate-950">Analyzing image</h3>
            <p className="mt-3 text-sm leading-7 text-slate-600">
              {modelName} is processing the retinal image and preparing a classification result.
            </p>
            <div className="mt-5 inline-flex rounded-full border border-cyan-100 bg-cyan-50 px-4 py-2 text-sm font-medium text-cyan-800">
              Deep learning model extracting retinal features
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="diagnostic-card">
        <div className="flex items-center justify-center py-14">
          <div className="max-w-md text-center">
            {error ? (
              <>
                <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-[22px] bg-rose-50 text-rose-700">
                  <AlertTriangle className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-semibold text-slate-950">Analysis error</h3>
                <p className="mt-3 text-sm leading-7 text-slate-600">{error}</p>
                <div className="mt-5 rounded-[24px] border border-rose-200 bg-rose-50/80 p-4 text-sm leading-7 text-rose-800">
                  Please try a different image or switch to another available model variant.
                </div>
              </>
            ) : (
              <>
                <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-[22px] bg-slate-100 text-slate-500">
                  <Eye className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-semibold text-slate-950">Ready for analysis</h3>
                <p className="mt-3 text-sm leading-7 text-slate-600">
                  Upload a retinal fundus image to generate classification output, severity context, and a clinical recommendation.
                </p>
                <div className="mt-5 rounded-[24px] border border-cyan-100 bg-cyan-50/80 p-4 text-sm leading-7 text-cyan-900">
                  Supported formats: JPG, PNG, JPEG. High-resolution fundus images are recommended for clearer interpretation.
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  const predictedClass = drClasses[prediction.class]
  const probabilityLabels = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']

  const getResultIcon = () => {
    if (prediction.class === 0) {
      return <CheckCircle className="h-9 w-9 text-emerald-600" />
    }
    return <AlertTriangle className="h-9 w-9 text-amber-600" />
  }

  const getResultBadgeClass = () => {
    switch (prediction.class) {
      case 0:
        return 'result-no-dr'
      case 1:
        return 'result-mild'
      case 2:
        return 'result-moderate'
      case 3:
        return 'result-severe'
      case 4:
        return 'result-proliferative'
      default:
        return 'result-no-dr'
    }
  }

  const getSeverityColor = () => {
    switch (prediction.class) {
      case 0:
        return 'bg-emerald-500'
      case 1:
        return 'bg-amber-500'
      case 2:
        return 'bg-orange-500'
      case 3:
        return 'bg-rose-500'
      case 4:
        return 'bg-red-700'
      default:
        return 'bg-emerald-500'
    }
  }

  return (
    <div className="space-y-6">
      <section className="diagnostic-card">
        <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">Classification result</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">{predictedClass.name}</h2>
          </div>
          <span className="inline-flex w-fit rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-600">
            {modelName}
          </span>
        </div>

        <div className="diagnostic-summary">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="flex items-start gap-4">
              <div className="mt-1 flex h-14 w-14 items-center justify-center rounded-[22px] bg-white shadow-sm">
                {getResultIcon()}
              </div>
              <div>
                <div className={`${getResultBadgeClass()} text-sm`}>{predictedClass.severity}</div>
                <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600">{predictedClass.description}</p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:w-[20rem] lg:grid-cols-1">
              <div className="rounded-3xl border border-slate-200 bg-white/90 px-4 py-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Predicted grade</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">Grade {prediction.class}/4</p>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white/90 px-4 py-4">
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Top confidence</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">{(prediction.confidence * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-3xl border border-slate-200 bg-white/90 p-4">
            <div className="mb-3 flex items-center justify-between gap-4">
              <span className="text-sm font-semibold text-slate-700">Severity assessment</span>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                DR scale
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-slate-200">
              <div
                className={`h-3 rounded-full ${getSeverityColor()} transition-all duration-1000 ease-out`}
                style={{ width: `${((prediction.class + 1) / 5) * 100}%` }}
              />
            </div>
            <div className="mt-3 grid grid-cols-5 gap-2 text-center text-[11px] font-medium uppercase tracking-[0.12em] text-slate-500">
              {probabilityLabels.map((label) => (
                <span key={label}>{label}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="recommendation-card mt-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white text-teal-700 shadow-sm">
              <ShieldAlert className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">Clinical recommendation</p>
              <h3 className="mt-1 text-lg font-semibold text-slate-950">Suggested next step</h3>
            </div>
          </div>
          <p className="mt-4 text-sm leading-7 text-slate-700">{predictedClass.recommendation}</p>
        </div>
      </section>

      <section className="diagnostic-card">
        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
            <BarChart3 className="h-5 w-5" />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Model output</p>
            <h3 className="mt-1 text-xl font-semibold text-slate-950">Probability distribution</h3>
          </div>
        </div>

        <div className="space-y-4">
          {drClasses.map((drClass, index) => {
            const probability = Number((prediction.probabilities[index] * 100).toFixed(1))
            const isHighest = index === prediction.class

            return (
              <div
                key={drClass.id}
                className={`rounded-3xl border px-4 py-4 transition-all duration-200 ${
                  isHighest ? 'border-cyan-200 bg-cyan-50/70 shadow-sm' : 'border-slate-200 bg-white/80'
                }`}
              >
                <div className="mb-3 flex items-center justify-between gap-4">
                  <span className={`text-sm font-semibold ${isHighest ? 'text-slate-950' : 'text-slate-700'}`}>
                    {drClass.name}
                  </span>
                  <span className={`text-sm font-bold ${isHighest ? 'text-cyan-700' : 'text-slate-600'}`}>
                    {probability.toFixed(1)}%
                  </span>
                </div>
                <div className="probability-bar-track">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-1000 ease-out ${
                      isHighest ? 'bg-gradient-to-r from-teal-600 to-cyan-500' : 'bg-slate-400'
                    }`}
                    style={{ width: `${Math.max(probability, 2)}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <section className="disclaimer-card">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-white/70 text-amber-700">
            <AlertTriangle className="h-4 w-4" />
          </div>
          <div>
            <h4 className="text-sm font-semibold uppercase tracking-[0.16em]">Medical disclaimer</h4>
            <p className="mt-2">
              This interface is designed for research and educational use only. Outputs should not be used
              as the sole basis for diagnosis or treatment decisions.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
