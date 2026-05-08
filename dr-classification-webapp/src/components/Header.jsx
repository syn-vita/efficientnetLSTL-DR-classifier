import { ArrowLeft, Eye } from 'lucide-react'
import { appConfig } from '../config/models'

export default function Header({ onBackToLanding, showBackButton = false }) {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4">
        <div className="flex min-w-0 items-start gap-3">
          {showBackButton && (
            <button
              onClick={onBackToLanding}
              className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-white/80 text-slate-600 transition-colors hover:bg-white hover:text-slate-900"
              title="Back to landing"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
          )}

          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-600 to-cyan-500 text-white shadow-sm">
            <Eye className="h-5 w-5" />
          </div>

          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">
              {appConfig.subtitle}
            </p>
            <h1 className="mt-1 line-clamp-2 text-sm font-semibold text-slate-950 md:text-base">
              {appConfig.compactTitle}
            </h1>
          </div>
        </div>

        <p className="hidden max-w-sm text-right text-sm leading-6 text-slate-500 lg:block">
          Guided retinal image testing with LSTL surfaced first and the comparison models kept available.
        </p>
      </div>
    </header>
  )
}
