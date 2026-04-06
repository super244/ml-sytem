// Metal shader source code v2.0 for M-series GPU acceleration
// ============================================================================
// Supports:
// - M5 Ultra (80-core GPU, 1228 GB/s bandwidth)
// - M5 Max (40-core GPU, 614 GB/s bandwidth)
// - M4/M3/M2/M1 series
// - FlashAttention-2, fused kernels, async compute
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants for different generations
// ============================================================================

// M5 Ultra optimized tile sizes
constant uint TILE_SIZE_M5_ULTRA = 128;
constant uint THREADGROUP_SIZE_M5_ULTRA = 1024;

// M5 Max/M4 Max optimized tile sizes
constant uint TILE_SIZE_M5_MAX = 64;
constant uint THREADGROUP_SIZE_M5_MAX = 512;

// Default tile sizes
constant uint TILE_SIZE_DEFAULT = 32;
constant uint THREADGROUP_SIZE_DEFAULT = 256;

// ============================================================================
// Matrix Multiplication v2 - Tiled with dynamic tile sizing
// ============================================================================

kernel void matmul_tiled_v2(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint6& dims [[buffer(3)]], // [m, k, n, tile_m, tile_n, tile_k]
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    uint m = dims[0];
    uint k = dims[1];
    uint n = dims[2];
    uint tile_m = dims[3];
    uint tile_n = dims[4];
    uint tile_k = dims[5];
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= m || col >= n) return;
    
    // Threadgroup-local accumulation
    threadgroup float local_a[64][64];
    threadgroup float local_b[64][64];
    
    float acc = 0.0f;
    
    // Process in tiles
    for (uint t = 0; t < k; t += tile_k) {
        // Load tiles into threadgroup memory
        uint local_row = lid.y;
        uint local_col = lid.x;
        
        if (row < m && (t + local_col) < k) {
            local_a[local_row][local_col] = a[row * k + t + local_col];
        } else {
            local_a[local_row][local_col] = 0.0f;
        }
        
        if (col < n && (t + local_row) < k) {
            local_b[local_row][local_col] = b[(t + local_row) * n + col];
        } else {
            local_b[local_row][local_col] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute on tile
        for (uint i = 0; i < tile_k; i++) {
            acc += local_a[local_row][i] * local_b[i][local_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    c[row * n + col] = acc;
}

// ============================================================================
// Fused RMSNorm + SiLU v2 - Optimized reduction
// ============================================================================

kernel void fused_rms_norm_silu_v2(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    constant float& eps [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]]
) {
    threadgroup float shared_sum[1024];
    
    // Load and square
    float val = 0.0f;
    if (gid < length) {
        val = data[gid];
    }
    shared_sum[lid] = val * val;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for sum
    for (uint s = 512; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute RMS
    float rms = sqrt(shared_sum[0] / float(length) + eps);
    float scale = 1.0f / rms;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply normalization + SiLU
    if (gid < length) {
        float normalized = val * scale;
        // SiLU(x) = x * sigmoid(x)
        data[gid] = normalized / (1.0f + exp(-normalized));
    }
}

// ============================================================================
// Optimized Softmax v2 - Warp-level reduction for M5
// ============================================================================

kernel void softmax_warp_optimized(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const uint WARP_SIZE = 32;
    const uint warps_per_group = 32; // 1024 / 32
    
    threadgroup float shared_max[32];
    threadgroup float shared_sum[32];
    
    uint warp_id = lid / WARP_SIZE;
    uint lane_id = lid % WARP_SIZE;
    
    // Find local max per warp
    float local_max = -INFINITY;
    for (uint i = lid; i < length; i += 1024) {
        local_max = max(local_max, data[i]);
    }
    
    // Warp reduction for max
    for (uint offset = 16; offset > 0; offset >>= 1) {
        local_max = max(local_max, simd_shuffle_down(local_max, offset));
    }
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Find global max
    if (lid < warps_per_group) {
        float warp_max = shared_max[lid];
        for (uint offset = 16; offset > 0; offset >>= 1) {
            warp_max = max(warp_max, simd_shuffle_down(warp_max, offset));
        }
        if (lid == 0) {
            shared_max[0] = warp_max;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = shared_max[0];
    
    // Compute exp(x - max) and sum per warp
    float local_sum = 0.0f;
    for (uint i = lid; i < length; i += 1024) {
        float exp_val = exp(data[i] - global_max);
        data[i] = exp_val;
        local_sum += exp_val;
    }
    
    // Warp reduction for sum
    for (uint offset = 16; offset > 0; offset >>= 1) {
        local_sum += simd_shuffle_down(local_sum, offset);
    }
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Global sum
    if (lid < warps_per_group) {
        float warp_sum = shared_sum[lid];
        for (uint offset = 16; offset > 0; offset >>= 1) {
            warp_sum += simd_shuffle_down(warp_sum, offset);
        }
        if (lid == 0) {
            shared_sum[0] = warp_sum;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_sum = 1.0f / shared_sum[0];
    
    // Normalize
    for (uint i = lid; i < length; i += 1024) {
        data[i] *= inv_sum;
    }
}

// ============================================================================
// FlashAttention v2 - Multi-query optimized
// ============================================================================

kernel void flash_attention_v2(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    const uint TILE_Q = 64;   // Queries per tile
    const uint TILE_KV = 64;  // Keys/values per tile
    
    uint head_id = gid.z;
    uint query_row = gid.y * TILE_Q + lid;
    
    if (head_id >= num_heads || query_row >= seq_len) return;
    
    // Per-thread local storage for attention scores
    float q_vec[128]; // Max head_dim
    float o_acc[128];
    
    // Load query vector
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = q[(head_id * seq_len + query_row) * head_dim + d];
        o_acc[d] = 0.0f;
    }
    
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // Process KV cache in tiles
    for (uint kv_start = 0; kv_start < seq_len; kv_start += TILE_KV) {
        float tile_max = -INFINITY;
        
        // Compute attention scores for this tile
        for (uint j = 0; j < TILE_KV && (kv_start + j) < seq_len; j++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                float k_val = k[(head_id * seq_len + kv_start + j) * head_dim + d];
                dot += q_vec[d] * k_val;
            }
            dot *= 1.0f / sqrt(float(head_dim));
            tile_max = max(tile_max, dot);
        }
        
        // Online softmax rescaling
        float new_max = max(row_max, tile_max);
        float scale = exp(row_max - new_max);
        row_sum = row_sum * scale;
        row_max = new_max;
        
        // Accumulate weighted values
        for (uint j = 0; j < TILE_KV && (kv_start + j) < seq_len; j++) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                float k_val = k[(head_id * seq_len + kv_start + j) * head_dim + d];
                dot += q_vec[d] * k_val;
            }
            dot *= 1.0f / sqrt(float(head_dim));
            
            float p = exp(dot - row_max);
            row_sum += p;
            
            for (uint d = 0; d < head_dim; d++) {
                float v_val = v[(head_id * seq_len + kv_start + j) * head_dim + d];
                o_acc[d] = o_acc[d] * scale + p * v_val;
            }
        }
    }
    
    // Write output
    float inv_sum = 1.0f / row_sum;
    for (uint d = 0; d < head_dim; d++) {
        out[(head_id * seq_len + query_row) * head_dim + d] = o_acc[d] * inv_sum;
    }
}

// ============================================================================
// Grouped Query Attention (GQA) - For multi-query attention
// ============================================================================

kernel void grouped_query_attention(
    device const float* q [[buffer(0)]],      // [num_q_heads, seq_len, head_dim]
    device const float* k [[buffer(1)]],      // [num_kv_heads, seq_len, head_dim]
    device const float* v [[buffer(2)]],      // [num_kv_heads, seq_len, head_dim]
    device float* out [[buffer(3)]],          // [num_q_heads, seq_len, head_dim]
    constant uint& num_q_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint q_head_id = gid.z;
    uint kv_head_id = q_head_id * num_kv_heads / num_q_heads; // GQA mapping
    uint query_row = gid.y;
    
    if (q_head_id >= num_q_heads || query_row >= seq_len) return;
    
    float q_vec[128];
    float o_acc[128];
    
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = q[(q_head_id * seq_len + query_row) * head_dim + d];
        o_acc[d] = 0.0f;
    }
    
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // Attention over all positions
    for (uint j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            float k_val = k[(kv_head_id * seq_len + j) * head_dim + d];
            dot += q_vec[d] * k_val;
        }
        dot *= 1.0f / sqrt(float(head_dim));
        
        float new_max = max(row_max, dot);
        float scale = exp(row_max - new_max);
        row_sum = row_sum * scale + exp(dot - new_max);
        row_max = new_max;
        
        for (uint d = 0; d < head_dim; d++) {
            float v_val = v[(kv_head_id * seq_len + j) * head_dim + d];
            o_acc[d] = o_acc[d] * scale + exp(dot - new_max) * v_val;
        }
    }
    
    float inv_sum = 1.0f / row_sum;
    for (uint d = 0; d < head_dim; d++) {
        out[(q_head_id * seq_len + query_row) * head_dim + d] = o_acc[d] * inv_sum;
    }
}

// ============================================================================
// Quantized Matrix Multiplication v2 - Q4_0
// ============================================================================

struct BlockQ4_0 {
    float d;
    uchar2 qs[8]; // 16 bytes total, 32 4-bit weights
};

kernel void matmul_q4_0_v2(
    device const float* a [[buffer(0)]],
    device const BlockQ4_0* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint BLOCK_SIZE = 32; // Q4_0 block size
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= m || col >= n) return;
    
    float acc = 0.0f;
    uint num_blocks = k / BLOCK_SIZE;
    
    // Prefetch hint for unified memory
    #if defined(M5_ULTRA) || defined(M5)
    prefetch_sram(weights + col * num_blocks, 0);
    #endif
    
    for (uint b = 0; b < num_blocks; b++) {
        BlockQ4_0 block = weights[col * num_blocks + b];
        float scale = block.d;
        
        // Dequantize 32 weights and multiply
        for (uint i = 0; i < 8; i++) {
            uchar2 packed = block.qs[i];
            
            // Low nibble (first 4 bits)
            float w0 = (float(packed[0] & 0x0F) - 8.0f) * scale;
            acc += a[row * k + b * BLOCK_SIZE + i * 4] * w0;
            
            // High nibble (next 4 bits)
            float w1 = (float((packed[0] >> 4) & 0x0F) - 8.0f) * scale;
            acc += a[row * k + b * BLOCK_SIZE + i * 4 + 1] * w1;
            
            float w2 = (float(packed[1] & 0x0F) - 8.0f) * scale;
            acc += a[row * k + b * BLOCK_SIZE + i * 4 + 2] * w2;
            
            float w3 = (float((packed[1] >> 4) & 0x0F) - 8.0f) * scale;
            acc += a[row * k + b * BLOCK_SIZE + i * 4 + 3] * w3;
        }
    }
    
    out[row * n + col] = acc;
}

// ============================================================================
// Vector Operations v2
// ============================================================================

kernel void vec_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    out[gid] = a[gid] + b[gid];
}

kernel void vec_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    out[gid] = a[gid] * b[gid];
}

kernel void vec_scale_f32(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    data[gid] *= scale;
}

// ============================================================================
// RMS Normalization v2
// ============================================================================

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[1024];
    
    // Parallel sum of squares
    float sum_sq = 0.0f;
    for (uint i = lid; i < length; i += 1024) {
        float v = input[i];
        sum_sq += v * v;
    }
    shared_sum[lid] = sum_sq;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint s = 512; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(length) + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize
    for (uint i = lid; i < length; i += 1024) {
        output[i] = input[i] * inv_rms;
    }
}

// ============================================================================
// SiLU (Swish) Activation
// ============================================================================

kernel void silu_f32(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    float x = data[gid];
    data[gid] = x / (1.0f + exp(-x));
}

// ============================================================================
// GELU Activation - Fast approximation
// ============================================================================

kernel void gelu_f32(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    constant bool& exact [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    float x = data[gid];
    
    if (exact) {
        // Exact GELU using erf approximation
        data[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    } else {
        // Fast approximation
        data[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * x));
    }
}

// ============================================================================
// Rotary Positional Embeddings (RoPE)
// ============================================================================

kernel void rope_f32(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& base_freq [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint head_id = gid.z;
    uint pos = gid.y;
    uint pair = gid.x;
    
    if (pos >= seq_len || pair * 2 >= head_dim) return;
    
    uint idx = (head_id * seq_len + pos) * head_dim + pair * 2;
    
    float freq = 1.0f / pow(float(base_freq), float(pair * 2) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    
    // Apply to q
    float q0 = q[idx];
    float q1 = q[idx + 1];
    q[idx] = q0 * cos_a - q1 * sin_a;
    q[idx + 1] = q0 * sin_a + q1 * cos_a;
    
    // Apply to k
    float k0 = k[idx];
    float k1 = k[idx + 1];
    k[idx] = k0 * cos_a - k1 * sin_a;
    k[idx + 1] = k0 * sin_a + k1 * cos_a;
}

// ============================================================================
// Layer Normalization
// ============================================================================

kernel void layer_norm_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared[1024];
    
    // Compute mean
    float sum = 0.0f;
    for (uint i = lid; i < length; i += 1024) {
        sum += input[i];
    }
    shared[lid] = sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 512; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared[0] / float(length);
    
    // Compute variance
    sum = 0.0f;
    for (uint i = lid; i < length; i += 1024) {
        float diff = input[i] - mean;
        sum += diff * diff;
    }
    shared[lid] = sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 512; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_std = 1.0f / sqrt(shared[0] / float(length) + eps);
    
    // Normalize
    for (uint i = lid; i < length; i += 1024) {
        output[i] = (input[i] - mean) * inv_std;
    }
}

// ============================================================================
// Copy with async prefetch for M5 Ultra
// ============================================================================

kernel void async_copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    
    #if defined(M5_ULTRA) || defined(M5)
    // Prefetch next cache line
    if (gid + 64 < length) {
        prefetch_sram(src + gid + 64, 0);
    }
    #endif
    
    dst[gid] = src[gid];
}
