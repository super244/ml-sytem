use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub page_size: usize,
    pub max_tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct KvCacheStats {
    pub stored_tokens: usize,
    pub pages_in_use: usize,
    pub max_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct KvCache {
    config: KvCacheConfig,
    tokens: Vec<u32>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Self {
        Self {
            config,
            tokens: Vec::new(),
        }
    }

    pub fn append(&mut self, token_id: u32) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.tokens.len() < self.config.max_tokens,
            "KV cache is full"
        );
        self.tokens.push(token_id);
        Ok(())
    }

    pub fn recent(&self, count: usize) -> &[u32] {
        let start = self.tokens.len().saturating_sub(count);
        &self.tokens[start..]
    }

    pub fn stats(&self) -> KvCacheStats {
        let pages_in_use = self.tokens.len().div_ceil(self.config.page_size.max(1));
        KvCacheStats {
            stored_tokens: self.tokens.len(),
            pages_in_use,
            max_tokens: self.config.max_tokens,
        }
    }
}
