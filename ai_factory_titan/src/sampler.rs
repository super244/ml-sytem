//! Token sampler — temperature, top-k, top-p, min-p, typical-p, repetition penalty.
//!
//! v0.2 additions beyond v0.1:
//! - `min_p` filter: removes tokens whose probability < min_p * p_max (fast, no sort).
//! - `typical_p` filter: entropy-based "locally typical" sampling.
//! - Seed-based deterministic sampling via a fast Lehmer LCG.
//! - `SamplerConfig::greedy()` convenience constructor.

use serde::Serialize;

// ─── Config ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
pub struct SamplerConfig {
    /// Softmax temperature (1.0 = neutral).
    pub temperature: f32,
    /// Keep only the top-k logits (0 = disabled).
    pub top_k: usize,
    /// Nucleus (cumulative probability threshold).
    pub top_p: f32,
    /// Minimum probability relative to top token (0 = disabled).
    pub min_p: f32,
    /// Locally typical sampling threshold (0 = disabled).
    pub typical_p: f32,
    /// Penalise recently generated tokens (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Optional fixed seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature:        0.8,
            top_k:              40,
            top_p:              0.95,
            min_p:              0.0,
            typical_p:          0.0,
            repetition_penalty: 1.1,
            seed:               None,
        }
    }
}

impl SamplerConfig {
    /// Deterministic argmax — no randomness.
    pub fn greedy() -> Self {
        Self {
            temperature:        0.0,
            top_k:              1,
            top_p:              1.0,
            min_p:              0.0,
            typical_p:          0.0,
            repetition_penalty: 1.0,
            seed:               None,
        }
    }
}

// ─── Sampler ──────────────────────────────────────────────────────────────────

/// Sample the next token from `logits`.
///
/// Pipeline (applied in order):
/// 1. Repetition penalty on tokens in `recent_tokens`.
/// 2. Temperature scaling.
/// 3. Top-k truncation.
/// 4. Softmax and probability computation.
/// 5. Min-p filter.
/// 6. Typical-p filter.
/// 7. Top-p (nucleus) truncation.
/// 8. Weighted draw (or argmax if temperature == 0).
pub fn sample_token(
    logits: &[f32],
    recent_tokens: &[u32],
    config: &SamplerConfig,
) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }

    // 1. Apply repetition penalty.
    let mut scores: Vec<f32> = logits.to_vec();
    for token in recent_tokens {
        let idx = *token as usize;
        if idx < scores.len() {
            scores[idx] /= config.repetition_penalty.max(1.0);
        }
    }

    // 2. Greedy fast path.
    if config.temperature <= 0.0 || config.top_k == 1 {
        return scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
    }

    // 3. Temperature scaling.
    for s in &mut scores {
        *s /= config.temperature;
    }

    // 4. Top-k truncation.
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = config.top_k.max(1).min(indexed.len());
    indexed.truncate(top_k);

    // 5. Softmax.
    let max_l = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| (i, (l - max_l).exp()))
        .collect();
    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        return Some(probs[0].0);
    }
    for (_, p) in &mut probs {
        *p /= total;
    }

    // 6. Min-p filter.
    if config.min_p > 0.0 {
        let p_max = probs[0].1;
        let threshold = config.min_p * p_max;
        probs.retain(|(_, p)| *p >= threshold);
        if probs.is_empty() {
            return Some(indexed[0].0);
        }
    }

    // 7. Typical-p filter.
    if config.typical_p > 0.0 && config.typical_p < 1.0 {
        let entropy: f32 = probs.iter().map(|(_, p)| -p * p.ln()).sum();
        let mut with_shift: Vec<(usize, f32, f32)> = probs
            .iter()
            .map(|&(i, p)| (i, p, (-p.ln() - entropy).abs()))
            .collect();
        with_shift.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        probs = with_shift
            .into_iter()
            .take_while(|(_, p, _)| {
                if cum >= config.typical_p {
                    return false;
                }
                cum += p;
                true
            })
            .map(|(i, p, _)| (i, p))
            .collect();
        if probs.is_empty() {
            return Some(indexed[0].0);
        }
    }

    // 8. Top-p nucleus.
    if config.top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut nucleus: Vec<(usize, f32)> = probs
            .iter()
            .take_while(|(_, p)| {
                if cum >= config.top_p {
                    return false;
                }
                cum += p;
                true
            })
            .cloned()
            .collect();
        if !nucleus.is_empty() {
            // Always include the last token that pushed us past threshold.
            if nucleus.len() < probs.len() {
                nucleus.push(probs[nucleus.len()]);
            }
            probs = nucleus;
        }
    }

    // 9. Weighted draw.
    weighted_draw(&probs, config.seed)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Draw from a discrete distribution using a fast Lehmer LCG.
fn weighted_draw(probs: &[(usize, f32)], seed: Option<u64>) -> Option<usize> {
    if probs.is_empty() {
        return None;
    }
    if probs.len() == 1 {
        return Some(probs[0].0);
    }

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let r = lcg_uniform(seed) * total;

    let mut cum = 0.0f32;
    for &(i, p) in probs {
        cum += p;
        if cum >= r {
            return Some(i);
        }
    }
    Some(probs.last().unwrap().0)
}

/// Lehmer MCG — reproducible pseudo-random f32 in [0, 1).
fn lcg_uniform(seed: Option<u64>) -> f32 {
    // Mix the seed with a timestamp to avoid constant folding.
    let state = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(42)
    });
    let next = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (next >> 33) as f32 / (1u64 << 31) as f32
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1f32, 0.9, 0.5, 0.3];
        let pick = sample_token(&logits, &[], &SamplerConfig::greedy()).unwrap();
        assert_eq!(pick, 1);
    }

    #[test]
    fn repetition_penalty_suppresses_recent_token() {
        let config = SamplerConfig {
            temperature:        0.7,
            top_k:              1, // greedy path: always picks post-penalty argmax
            top_p:              0.9,
            min_p:              0.0,
            typical_p:          0.0,
            repetition_penalty: 2.5,
            seed:               None,
        };
        // Token 1 is highest logit (1.4) but penalised → 0.56; token 2 (1.1) now wins.
        let pick = sample_token(&[0.2, 1.4, 1.1, 0.7], &[1], &config).unwrap();
        assert_eq!(pick, 2);
    }

    #[test]
    fn min_p_filters_low_probability_tokens() {
        let config = SamplerConfig {
            temperature:        1.0,
            top_k:              100,
            top_p:              1.0,
            min_p:              0.5, // very aggressive — everything below 50% of top gets cut
            typical_p:          0.0,
            repetition_penalty: 1.0,
            seed:               Some(42),
        };
        // logits dominated by token 0; min_p should remove tokens 1–3
        let pick = sample_token(&[10.0f32, 0.1, 0.1, 0.1], &[], &config).unwrap();
        assert_eq!(pick, 0, "min_p should leave only dominant token");
    }

    #[test]
    fn empty_logits_returns_none() {
        assert!(sample_token(&[], &[], &SamplerConfig::default()).is_none());
    }
}
