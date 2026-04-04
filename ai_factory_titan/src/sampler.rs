use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
        }
    }
}

pub fn sample_token(logits: &[f32], recent_tokens: &[u32], config: &SamplerConfig) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }

    let mut scored: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    for (index, value) in &mut scored {
        if recent_tokens.iter().any(|token| *token as usize == *index) {
            *value /= config.repetition_penalty.max(1.0);
        }
        if config.temperature > 0.0 {
            *value /= config.temperature;
        }
    }

    scored.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = config.top_k.max(1).min(scored.len());
    scored.truncate(top_k);

    let max_logit = scored.first()?.1;
    let mut weighted = Vec::with_capacity(scored.len());
    let mut total = 0.0f32;
    for (index, value) in scored {
        let weight = (value - max_logit).exp();
        total += weight;
        weighted.push((index, value, weight));
    }

    let mut cumulative = 0.0f32;
    let mut nucleus = Vec::new();
    for (index, value, weight) in weighted {
        let prob = if total > 0.0 { weight / total } else { 0.0 };
        cumulative += prob;
        nucleus.push((index, value));
        if cumulative >= config.top_p.clamp(0.0, 1.0) {
            break;
        }
    }

    nucleus
        .into_iter()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}
