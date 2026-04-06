//! Tensor layouts, quantization formats, and low-level block types.
//!
//! v0.3.0 additions: BF16 format, Q5K / Q6K blocks matching GGUF v3 spec.

use serde::Serialize;

// ─── Quantization Format ──────────────────────────────────────────────────────

/// Supported on-disk quantization formats (GGUF v3 compatible).
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    /// 4-bit with a scalar delta per 32-element block.
    Q4_0,
    /// 4-bit K-quant with super-block scales (higher quality).
    Q4K,
    /// 5-bit K-quant.
    Q5K,
    /// 6-bit K-quant — best quality below FP16.
    Q6K,
    /// 8-bit with a scalar delta per 32-element block.
    Q8_0,
    /// 16-bit IEEE-754 half precision.
    F16,
    /// 16-bit brain-float (exponent range of FP32, reduced mantissa).
    BF16,
}

impl QuantizationFormat {
    /// Number of elements packed into one block.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4K | Self::Q5K => 32,
            Self::Q6K => 256, // Q6K uses 256-element super-blocks in GGUF
            Self::Q8_0 => 32,
            Self::F16 | Self::BF16 => 1,
        }
    }

    /// Serialised bytes per block (matches GGUF on-disk layout).
    #[must_use]
    pub const fn bytes_per_block(self) -> usize {
        match self {
            Self::Q4_0 => 18,  // 2 bytes scale + 16 bytes packed nibbles
            Self::Q4K => 24,   // super-block scale header + nibbles
            Self::Q5K => 26,
            Self::Q6K => 210,  // 256-element super-block serialised size
            Self::Q8_0 => 34,  // 2 bytes scale + 32 bytes ints
            Self::F16 => 2,
            Self::BF16 => 2,
        }
    }

    /// Returns `true` if the format uses floating-point (no explicit blocks).
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16)
    }

    /// Human-readable short name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

// ─── Tensor Shape ─────────────────────────────────────────────────────────────

/// 2-D tensor shape (row-major).
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: usize,
    pub cols: usize,
}

impl TensorShape {
    /// Total element count.
    #[must_use]
    #[inline]
    pub fn element_count(self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    /// Number of blocks needed for a given format.
    #[must_use]
    pub fn blocks(self, format: QuantizationFormat) -> usize {
        if format.is_float() {
            return self.element_count();
        }
        self.element_count().div_ceil(format.block_size())
    }
}

// ─── Tensor Layout ────────────────────────────────────────────────────────────

/// Storage layout for a quantised tensor.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct QuantizedTensorLayout {
    pub format: QuantizationFormat,
    pub block_size: usize,
    pub bytes_per_block: usize,
    /// Required alignment for the first byte of weight data.
    pub alignment_bytes: usize,
    /// Logical storage order description.
    pub storage: &'static str,
}

impl QuantizedTensorLayout {
    /// Construct from a format, filling in standard GGUF-compatible values.
    pub fn for_format(format: QuantizationFormat) -> Self {
        Self {
            format,
            block_size: format.block_size(),
            bytes_per_block: format.bytes_per_block(),
            alignment_bytes: 64,
            storage: "blocked-row-major",
        }
    }

    /// Estimated serialised byte count for a tensor of `shape`.
    #[must_use]
    pub fn estimated_bytes(&self, shape: TensorShape) -> usize {
        let blocks = shape.blocks(self.format);
        blocks.saturating_mul(self.bytes_per_block)
    }
}

// ─── Low-level block types ────────────────────────────────────────────────────

/// Alias to avoid dragging in the `half` crate in block definitions.
#[allow(non_camel_case_types)]
pub type f16 = u16;

/// 4-bit Q4_0 block: 1 FP16 scale + 16 packed-nibble bytes = 18 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    /// FP16 delta (scale).
    pub d: f16,
    /// 16 bytes: two 4-bit weights per byte, 32 weights total.
    pub qs: [u8; 16],
}

/// 8-bit Q8_0 block: 1 FP16 scale + 32 int8 values = 34 bytes.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    /// FP16 delta (scale).
    pub d: f16,
    /// 32 quantised int8 weights.
    pub qs: [i8; 32],
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_gguf_spec() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn format_names_are_unique() {
        use std::collections::HashSet;
        let formats = [
            QuantizationFormat::Q4_0,
            QuantizationFormat::Q4K,
            QuantizationFormat::Q5K,
            QuantizationFormat::Q6K,
            QuantizationFormat::Q8_0,
            QuantizationFormat::F16,
            QuantizationFormat::BF16,
        ];
        let names: HashSet<_> = formats.iter().map(|f| f.name()).collect();
        assert_eq!(names.len(), formats.len(), "duplicate format names");
    }

    #[test]
    fn layout_estimated_bytes_q4k_is_nonzero() {
        let layout = QuantizedTensorLayout::for_format(QuantizationFormat::Q4K);
        let shape = TensorShape { rows: 128, cols: 64 };
        assert!(layout.estimated_bytes(shape) > 0);
    }
}
