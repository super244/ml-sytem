use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    Q4_0,
    Q4K,
    Q8_0,
    F16,
}

impl QuantizationFormat {
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4K | Self::Q8_0 => 32,
            Self::F16 => 1,
        }
    }

    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4K => 24,
            Self::Q8_0 => 34,
            Self::F16 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: usize,
    pub cols: usize,
}

impl TensorShape {
    pub fn element_count(self) -> usize {
        self.rows.saturating_mul(self.cols)
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct QuantizedTensorLayout {
    pub format: QuantizationFormat,
    pub block_size: usize,
    pub bytes_per_block: usize,
    pub alignment_bytes: usize,
    pub storage: &'static str,
}

impl QuantizedTensorLayout {
    pub fn for_format(format: QuantizationFormat) -> Self {
        Self {
            format,
            block_size: format.block_size(),
            bytes_per_block: format.bytes_per_block(),
            alignment_bytes: 64,
            storage: "blocked-row-major",
        }
    }

    pub fn estimated_bytes(&self, shape: TensorShape) -> usize {
        if self.format == QuantizationFormat::F16 {
            return shape.element_count().saturating_mul(self.bytes_per_block);
        }
        let blocks = shape.element_count().div_ceil(self.block_size);
        blocks.saturating_mul(self.bytes_per_block)
    }
}
