import type { Candidate } from '@/lib/api';

import { MathBlock } from '@/components/math-block';
import { MetricBadge } from '@/components/panels/metric-badge';

type CandidateInspectorProps = {
  candidates: Candidate[];
  selectedIndex: number;
  onSelect: (index: number) => void;
};

export function CandidateInspector({
  candidates,
  selectedIndex,
  onSelect,
}: CandidateInspectorProps) {
  const selectedCandidate = candidates[selectedIndex] ?? candidates[0];
  if (!selectedCandidate) {
    return (
      <section className="panel detail-panel">
        <div className="section-title">Candidate Inspector</div>
        <p className="hero-copy">
          Run a query to inspect multi-sample candidates and verifier signals.
        </p>
      </section>
    );
  }

  return (
    <section className="panel detail-panel">
      <div className="section-title">Candidate Inspector</div>
      <div className="message-meta compact">
        <span>Selected candidate</span>
        <span className="status-pill">{selectedCandidate.final_answer ?? 'no final answer'}</span>
      </div>
      <div className="candidate-tabs">
        {candidates.map((candidate, index) => (
          <button
            key={`${candidate.final_answer ?? 'candidate'}-${index}`}
            className={`candidate-tab${index === selectedIndex ? ' active' : ''}`}
            type="button"
            onClick={() => onSelect(index)}
          >
            Candidate {index + 1}
          </button>
        ))}
      </div>
      <div className="badge-row">
        <MetricBadge label="Vote" value={`${selectedCandidate.vote_count ?? 0}`} />
        <MetricBadge
          label="Score"
          value={(selectedCandidate.score ?? 0).toFixed(2)}
          tone="accent"
        />
        <MetricBadge
          label="Step"
          value={
            typeof selectedCandidate.verification?.step_correctness === 'number'
              ? `${(selectedCandidate.verification.step_correctness * 100).toFixed(0)}%`
              : 'n/a'
          }
          tone="secondary"
        />
      </div>
      {selectedCandidate.calculator_trace?.length ? (
        <div className="preview-block subtle">
          <strong>Calculator trace</strong>
          <p>
            {selectedCandidate.calculator_trace
              .map((item) => `${item.expression} = ${item.result}`)
              .join(' | ')}
          </p>
        </div>
      ) : null}
      <MathBlock content={selectedCandidate.display_text} />
    </section>
  );
}
