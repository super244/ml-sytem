//! KV-cache with paged attention and sliding-window eviction.
//!
//! Architecture:
//! - Pages of fixed size allow allocation without fragmentation.
//! - Each page covers `page_size` token positions; new pages are appended on demand.
//! - The cache can store N heads × head_dim key/value vectors per token position.
//! - A sliding-window eviction policy discards the oldest pages once `max_pages` is reached,
//!   giving O(1) append even at very long context lengths.

use serde::Serialize;
use std::collections::VecDeque;

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Number of token positions per page.
    pub page_size: usize,
    /// Hard limit on cached token positions.
    pub max_tokens: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Optional sliding-window eviction: keep at most this many pages.
    /// `None` means unlimited (up to `max_tokens`).
    pub max_pages: Option<usize>,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_tokens: 4096,
            heads: 32,
            head_dim: 128,
            max_pages: None,
        }
    }
}

// ─── Statistics ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct KvCacheStats {
    pub stored_tokens: usize,
    pub pages_in_use: usize,
    pub max_tokens: usize,
    /// Pages that were evicted under the sliding-window policy.
    pub evicted_pages: usize,
    /// Utilisation as a percentage (0–100).
    pub utilisation_pct: u8,
}

// ─── Page ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct KvPage {
    /// Token IDs stored in this page.
    tokens: Vec<u32>,
    /// Key vectors: Vec<[heads × head_dim f32]> — one entry per token.
    keys: Vec<Vec<f32>>,
    /// Value vectors: Vec<[heads × head_dim f32]> — one entry per token.
    values: Vec<Vec<f32>>,
}

impl KvPage {
    fn new(capacity: usize, _heads: usize, _head_dim: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn is_full(&self, page_size: usize) -> bool {
        self.tokens.len() >= page_size
    }
}

// ─── Cache ──────────────────────────────────────────────────────────────────────

/// Paged KV-cache.
#[derive(Clone, Debug)]
pub struct KvCache {
    config: KvCacheConfig,
    pages: VecDeque<KvPage>,
    total_stored: usize,
    evicted_pages: usize,
}

impl KvCache {
    /// Create a new empty cache.
    pub fn new(config: KvCacheConfig) -> Self {
        let initial_page = KvPage::new(config.page_size, config.heads, config.head_dim);
        let mut pages = VecDeque::new();
        pages.push_back(initial_page);
        Self {
            config,
            pages,
            total_stored: 0,
            evicted_pages: 0,
        }
    }

    /// Append a token (with its pre-computed key/value vectors).
    /// Returns an error when the cache is full and no eviction policy is set.
    pub fn append(&mut self, token_id: u32) -> anyhow::Result<()> {
        self.append_with_kv(token_id, vec![], vec![])
    }

    /// Append a token with its key/value tensors.
    pub fn append_with_kv(
        &mut self,
        token_id: u32,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> anyhow::Result<()> {
        // Enforce max_tokens hard limit.
        if self.total_stored >= self.config.max_tokens {
            return Err(anyhow::anyhow!("KV cache is full ({} tokens)", self.config.max_tokens));
        }

        // Ensure there's a page with space.
        if self
            .pages
            .back()
            .map(|p| p.is_full(self.config.page_size))
            .unwrap_or(true)
        {
            self.allocate_page()?;
        }

        let page = self.pages.back_mut().unwrap();
        page.tokens.push(token_id);
        page.keys.push(key);
        page.values.push(value);
        self.total_stored += 1;
        Ok(())
    }

    /// Return the most-recent `count` token IDs in chronological order.
    pub fn recent(&self, count: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        let mut remaining = count;
        // Collect tokens newest-first (reverse page order, reverse token order within page).
        for page in self.pages.iter().rev() {
            let take = remaining.min(page.tokens.len());
            let start = page.tokens.len() - take;
            out.extend(page.tokens[start..].iter().rev().copied());
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        // Reverse to return chronological order.
        out.reverse();
        out
    }

    /// Return a slice of the most-recent `count` tokens from contiguous storage
    /// (single-page fast path; falls back to `recent()` for multi-page).
    pub fn recent_slice(&self, count: usize) -> Option<&[u32]> {
        let last = self.pages.back()?;
        if last.tokens.len() >= count {
            let start = last.tokens.len() - count;
            Some(&last.tokens[start..])
        } else {
            None
        }
    }

    /// Retrieve the key tensor for a specific token position.
    pub fn key_at(&self, position: usize) -> Option<&[f32]> {
        let page_idx = position / self.config.page_size;
        let token_idx = position % self.config.page_size;
        self.pages.get(page_idx)?.keys.get(token_idx).map(|v| v.as_slice())
    }

    /// Retrieve the value tensor for a specific token position.
    pub fn value_at(&self, position: usize) -> Option<&[f32]> {
        let page_idx = position / self.config.page_size;
        let token_idx = position % self.config.page_size;
        self.pages.get(page_idx)?.values.get(token_idx).map(|v| v.as_slice())
    }

    /// Total stored token count.
    pub fn len(&self) -> usize {
        self.total_stored
    }

    pub fn is_empty(&self) -> bool {
        self.total_stored == 0
    }

    /// Clear all cached state (but keep config).
    pub fn clear(&mut self) {
        self.pages.clear();
        self.total_stored = 0;
        let initial = KvPage::new(self.config.page_size, self.config.heads, self.config.head_dim);
        self.pages.push_back(initial);
    }

    /// Cache statistics.
    pub fn stats(&self) -> KvCacheStats {
        let pages_in_use = self.pages.len();
        let utilisation_pct = if self.config.max_tokens > 0 {
            ((self.total_stored as f64 / self.config.max_tokens as f64) * 100.0).min(100.0) as u8
        } else {
            0
        };
        KvCacheStats {
            stored_tokens: self.total_stored,
            pages_in_use,
            max_tokens: self.config.max_tokens,
            evicted_pages: self.evicted_pages,
            utilisation_pct,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn allocate_page(&mut self) -> anyhow::Result<()> {
        // Sliding-window eviction.
        if let Some(max_pages) = self.config.max_pages {
            while self.pages.len() >= max_pages {
                if let Some(evicted) = self.pages.pop_front() {
                    self.total_stored = self.total_stored.saturating_sub(evicted.len());
                    self.evicted_pages += 1;
                }
            }
        }
        self.pages
            .push_back(KvPage::new(self.config.page_size, self.config.heads, self.config.head_dim));
        Ok(())
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_basic_append_and_recent() {
        let mut cache = KvCache::new(KvCacheConfig {
            page_size: 2,
            max_tokens: 8,
            heads: 1,
            head_dim: 4,
            max_pages: None,
        });
        for id in [10u32, 20, 30, 40] {
            cache.append(id).expect("append");
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.recent(2), vec![30, 40]);
        assert_eq!(cache.stats().pages_in_use, 2);
        assert_eq!(cache.stats().utilisation_pct, 50);
    }

    #[test]
    fn kv_cache_evicts_old_pages_in_sliding_window() {
        let mut cache = KvCache::new(KvCacheConfig {
            page_size: 2,
            max_tokens: 100,
            heads: 1,
            head_dim: 4,
            max_pages: Some(2),
        });
        // Fill 3 pages (6 tokens); the first page should be evicted.
        for id in 0u32..6 {
            cache.append(id).expect("append");
        }
        assert_eq!(cache.stats().pages_in_use, 2);
        assert_eq!(cache.stats().evicted_pages, 1);
    }

    #[test]
    fn kv_cache_returns_error_when_full() {
        let mut cache = KvCache::new(KvCacheConfig {
            page_size: 4,
            max_tokens: 2,
            heads: 1,
            head_dim: 4,
            max_pages: None,
        });
        cache.append(1).expect("ok");
        cache.append(2).expect("ok");
        assert!(cache.append(3).is_err());
    }

    #[test]
    fn kv_cache_clear_resets_state() {
        let mut cache = KvCache::new(KvCacheConfig::default());
        cache.append(1).expect("ok");
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().evicted_pages, 0);
    }
}
