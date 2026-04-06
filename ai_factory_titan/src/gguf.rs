use anyhow::{ensure, Context};
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub alignment: u32,
}

pub fn parse_gguf_header(bytes: &[u8]) -> anyhow::Result<GgufHeader> {
    ensure!(bytes.len() >= 24, "GGUF header requires at least 24 bytes");
    ensure!(&bytes[..4] == b"GGUF", "invalid GGUF magic");

    let version = u32::from_le_bytes(
        bytes[4..8]
            .try_into()
            .context("invalid GGUF version bytes")?,
    );
    let tensor_count = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .context("invalid tensor count bytes")?,
    );
    let metadata_kv_count = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .context("invalid metadata count bytes")?,
    );

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
        alignment: 32,
    })
}
