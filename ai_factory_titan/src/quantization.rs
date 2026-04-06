//! Arrow-backed quantised weight catalogue and GGUF column helpers.
//!
//! v0.3.0: extended schema with format name, estimated bytes, and BF16 support.

use arrow_array::{ArrayRef, Float32Array, StringArray, UInt64Array, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::tensor::{QuantizationFormat, QuantizedTensorLayout, TensorShape};

// ─── Arrow Schema ─────────────────────────────────────────────────────────────

/// Canonical Arrow schema for quantised tensor metadata records.
///
/// Matches the GGUF v3 tensor-info table layout; suitable for Parquet
/// serialisation or in-process analytics.
pub fn quantized_weight_schema() -> Schema {
    Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("rows", DataType::UInt32, false),
        Field::new("cols", DataType::UInt32, false),
        Field::new("format", DataType::Utf8, false),         // e.g. "Q4_K"
        Field::new("block_size", DataType::UInt32, false),
        Field::new("bytes_per_block", DataType::UInt32, false),
        Field::new("estimated_bytes", DataType::UInt64, false),
        Field::new("q4_blocks", DataType::UInt8, false),
        Field::new("q8_blocks", DataType::UInt8, false),
        Field::new("delta_f32", DataType::Float32, true),    // optional scale factor
    ])
}

// ─── Column helpers ───────────────────────────────────────────────────────────

/// Wrap raw Q4 block bytes into an Arrow `UInt8Array`.
pub fn q4_arrow_column(data: Vec<u8>) -> ArrayRef {
    Arc::new(UInt8Array::from(data))
}

/// Create a `StringArray` column of format names for a set of tensors.
pub fn format_name_column(formats: &[QuantizationFormat]) -> ArrayRef {
    Arc::new(StringArray::from(
        formats.iter().map(|f| f.name()).collect::<Vec<_>>(),
    ))
}

/// Create a `UInt64Array` of estimated byte counts.
pub fn estimated_bytes_column(layouts: &[(QuantizedTensorLayout, TensorShape)]) -> ArrayRef {
    let bytes: Vec<u64> = layouts
        .iter()
        .map(|(layout, shape)| layout.estimated_bytes(*shape) as u64)
        .collect();
    Arc::new(UInt64Array::from(bytes))
}

/// Scale factor column (f32, nullable).
pub fn delta_column(deltas: &[Option<f32>]) -> ArrayRef {
    let values: Vec<Option<f32>> = deltas.to_vec();
    Arc::new(Float32Array::from(values))
}

// ─── Layout helpers ───────────────────────────────────────────────────────────

/// Default Q4K layout (highest-quality 4-bit format).
pub fn default_q4_layout() -> QuantizedTensorLayout {
    QuantizedTensorLayout::for_format(QuantizationFormat::Q4K)
}

/// Default Q8_0 layout (8-bit, best for fine-tuning activations).
pub fn default_q8_layout() -> QuantizedTensorLayout {
    QuantizedTensorLayout::for_format(QuantizationFormat::Q8_0)
}

/// Estimate the serialised byte count for a Q4K tensor of a given shape.
pub fn estimate_q4_bytes(rows: usize, cols: usize) -> usize {
    default_q4_layout().estimated_bytes(TensorShape { rows, cols })
}

/// Estimate the serialised byte count for an F16 tensor.
pub fn estimate_f16_bytes(element_count: usize) -> usize {
    element_count * 2 // 2 bytes per element
}

/// Estimate the serialised byte count for a BF16 tensor.
pub fn estimate_bf16_bytes(element_count: usize) -> usize {
    element_count * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_fields() {
        let schema = quantized_weight_schema();
        assert_eq!(schema.fields().len(), 10);
        assert!(schema.field_with_name("tensor_name").is_ok());
        assert!(schema.field_with_name("estimated_bytes").is_ok());
    }

    #[test]
    fn q4k_layout_estimate_is_nonzero() {
        assert!(estimate_q4_bytes(128, 64) > 0);
    }

    #[test]
    fn f16_estimate_is_2_bytes_per_element() {
        assert_eq!(estimate_f16_bytes(1024), 2048);
    }
}
