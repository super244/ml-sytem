import type { ReactNode } from 'react';

type StatePanelProps = {
  eyebrow: string;
  title: string;
  description: string;
  tone?: 'default' | 'error' | 'loading';
  action?: ReactNode;
};

export function StatePanel({
  eyebrow,
  title,
  description,
  tone = 'default',
  action,
}: StatePanelProps) {
  const baseClasses =
    'flex flex-col gap-4 p-8 rounded-3xl border backdrop-blur-xl shadow-sm transition-all duration-300';

  let toneClasses = 'bg-white/80 border-gray-200/60';
  let eyebrowClasses = 'text-emerald-700 bg-emerald-50 border-emerald-100/50';

  if (tone === 'error') {
    toneClasses = 'bg-red-50/50 border-red-200 shadow-red-500/5';
    eyebrowClasses = 'text-red-700 bg-red-100 border-red-200';
  } else if (tone === 'loading') {
    toneClasses = 'bg-gray-50/50 border-gray-200 border-dashed animate-pulse';
    eyebrowClasses = 'text-gray-600 bg-gray-100 border-gray-200';
  }

  return (
    <section className={`${baseClasses} ${toneClasses}`}>
      <div className="flex flex-col items-start gap-3">
        <div
          className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold tracking-wider uppercase border ${eyebrowClasses}`}
        >
          {eyebrow}
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-gray-900 leading-tight">{title}</h2>
        <p className="text-base text-gray-600 leading-relaxed max-w-2xl">{description}</p>
      </div>
      {action ? <div className="mt-4 pt-4 border-t border-gray-100/50 flex">{action}</div> : null}
    </section>
  );
}
