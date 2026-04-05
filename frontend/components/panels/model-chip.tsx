import type { ModelInfo } from '@/lib/api';
import { formatModelAvailabilityDetail, formatModelSummary, formatModelTitle } from '@/lib/model-metadata';

type ModelChipProps = {
  model: ModelInfo;
};

export function ModelChip({ model }: ModelChipProps) {
  const summary = formatModelSummary(model);
  const availabilityState = model.availability_context?.state ?? (model.available ? 'available' : 'missing');
  const availabilityLabel = availabilityState === 'available' ? 'ready' : availabilityState;
  return (
    <div className="meta-tile model-chip">
      <div className="model-chip-header">
        <strong>{formatModelTitle(model)}</strong>
        <span className={`availability-pill${model.available ? ' ready' : ''}`}>
          {availabilityLabel}
        </span>
      </div>
      <span>{model.description ?? model.base_model}</span>
      {summary.length ? <span>{summary.join(' · ')}</span> : null}
      {model.tags?.length ? <span>{model.tags.join(' · ')}</span> : null}
      <span>{formatModelAvailabilityDetail(model)}</span>
    </div>
  );
}
