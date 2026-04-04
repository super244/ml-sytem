import type { ModelInfo } from '@/lib/api';

type ModelChipProps = {
  model: ModelInfo;
};

export function ModelChip({ model }: ModelChipProps) {
  return (
    <div className="meta-tile model-chip">
      <div className="model-chip-header">
        <strong>{model.label ?? model.name}</strong>
        <span className={`availability-pill${model.available ? ' ready' : ''}`}>
          {model.available ? 'ready' : 'missing'}
        </span>
      </div>
      <span>{model.description ?? model.base_model}</span>
      {model.tags?.length ? <span>{model.tags.join(' · ')}</span> : null}
    </div>
  );
}
