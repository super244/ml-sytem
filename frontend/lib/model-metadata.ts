import type { ModelInfo } from '@/lib/api';
import { formatParameterSize } from '@/lib/formatting';

export function formatModelTitle(model: Pick<ModelInfo, 'label' | 'name' | 'parameter_size_label' | 'parameter_size_b'>): string {
  const title = model.label ?? model.name;
  const size = model.parameter_size_label ?? formatParameterSize(model.parameter_size_b);
  return size && size !== 'n/a' ? `${title} · ${size}` : title;
}

export function formatModelSummary(
  model: Pick<ModelInfo, 'quantization' | 'tier' | 'scale_tags'>,
): string[] {
  const parts: string[] = [];
  const seen = new Set<string>();
  for (const value of [
    model.quantization,
    model.tier,
    ...(model.scale_tags?.slice(0, 3) ?? []),
  ]) {
    if (!value || !value.trim() || seen.has(value)) {
      continue;
    }
    seen.add(value);
    parts.push(value);
  }
  return parts;
}

export function formatModelAvailabilityDetail(model: Pick<ModelInfo, 'available' | 'availability_context'>): string {
  return model.availability_context?.detail ?? (model.available ? 'ready to serve' : 'missing from registry');
}
