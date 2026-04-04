use arrow_array::{ArrayRef, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::tensor::{QuantizationFormat, QuantizedTensorLayout, TensorShape};

pub fn quantized_weight_schema() -> Schema {
    Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("rows", DataType::UInt32, false),
        Field::new("cols", DataType::UInt32, false),
        Field::new("layout", DataType::Utf8, false),
        Field::new("q4_blocks", DataType::UInt8, false),
        Field::new("q8_blocks", DataType::UInt8, false),
    ])
}

pub fn q4_arrow_column(data: Vec<u8>) -> ArrayRef {
    Arc::new(UInt8Array::from(data))
}

pub fn default_q4_layout() -> QuantizedTensorLayout {
    QuantizedTensorLayout::for_format(QuantizationFormat::Q4K)
}

pub fn estimate_q4_bytes(rows: usize, cols: usize) -> usize {
    default_q4_layout().estimated_bytes(TensorShape { rows, cols })
}
