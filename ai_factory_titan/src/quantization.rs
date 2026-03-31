use arrow_array::{ArrayRef, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub fn quantized_weight_schema() -> Schema {
    Schema::new(vec![
        Field::new("tensor_name", DataType::Utf8, false),
        Field::new("q4_blocks", DataType::UInt8, false),
        Field::new("q8_blocks", DataType::UInt8, false),
    ])
}

pub fn q4_arrow_column(data: Vec<u8>) -> ArrayRef {
    Arc::new(UInt8Array::from(data))
}
