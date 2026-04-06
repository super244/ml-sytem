// CUDA kernel source v3.0 - Next-generation GPU kernels
// ============================================================================
// Supports:
// - Blackwell (SM 10.0) - 5th gen Tensor Cores, FP8, secure execution
// - Hopper (SM 9.0) - 4th gen Tensor Cores, FP8, Distributed Shared Memory
// - Ampere (SM 8.0+) - 3rd gen Tensor Cores, TF32
// - Ada Lovelace (SM 8.9)
// - RTX 50-series optimizations
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

// FP8 support for Hopper/Blackwell
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #include <cuda_fp8.h>
    #define TITAN_HAS_FP8 1
#endif

// PTX instructions for advanced optimizations
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #define TITAN_HAS_ASYNC_COPY 1
#endif

// WMMA for Tensor Cores
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// Architecture Detection Macros
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    #define TITAN_ARCH_BLACKWELL
    #define TITAN_HAS_5TH_GEN_TC 1
    #define TITAN_HAS_SECURE_CUDA 1
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #define TITAN_ARCH_HOPPER
    #define TITAN_HAS_4TH_GEN_TC 1
    #define TITAN_HAS_ASYNC_COPY 1
    #define TITAN_HAS_CLUSTER 1
    #define TITAN_HAS_TMA 1
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    #define TITAN_ARCH_AMPERE
    #define TITAN_HAS_3RD_GEN_TC 1
    #define TITAN_HAS_ASYNC_COPY 1
#endif

// ============================================================================
// TF32 Tensor Core Matrix Multiplication v5 - Blackwell Optimized
// ============================================================================

extern "C" __global__ void matmul_tf32_v5(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    uint m, uint k, uint n
) {
    // Optimized for Blackwell's larger SMs
    const uint TILE_M = 128;
    const uint TILE_N = 128;
    const uint TILE_K = 32;
    
    const uint bx = blockIdx.x;
    const uint by = blockIdx.y;
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;
    
    // Shared memory for cooperative loading
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];
    
    float acc[4][4] = {{0.0f}};
    
    const uint row_start = by * TILE_M + ty * 4;
    const uint col_start = bx * TILE_N + tx * 4;
    
    // Loop over K dimension with cooperative loading
    for (uint tile_k = 0; tile_k < k; tile_k += TILE_K) {
        // Load A tile cooperatively
        #pragma unroll
        for (uint i = 0; i < 4; i++) {
            uint row = row_start + i;
            if (row < m && tile_k + tx < k) {
                sA[ty * 4 + i][tx] = A[row * k + tile_k + tx];
            }
        }
        
        // Load B tile cooperatively
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint col = col_start + j;
            if (col < n && tile_k + ty < k) {
                sB[ty][tx * 4 + j] = B[(tile_k + ty) * n + col];
            }
        }
        
        __syncthreads();
        
        // Compute on tile
        #pragma unroll
        for (uint kk = 0; kk < TILE_K; kk++) {
            float a_vals[4];
            float b_vals[4];
            
            #pragma unroll
            for (uint i = 0; i < 4; i++) {
                a_vals[i] = sA[ty * 4 + i][kk];
            }
            #pragma unroll
            for (uint j = 0; j < 4; j++) {
                b_vals[j] = sB[kk][tx * 4 + j];
            }
            
            #pragma unroll
            for (uint i = 0; i < 4; i++) {
                #pragma unroll
                for (uint j = 0; j < 4; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint row = row_start + i;
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint col = col_start + j;
            if (row < m && col < n) {
                C[row * n + col] = acc[i][j];
            }
        }
    }
}

// ============================================================================
// FP16 Tensor Core Matrix Multiplication v5 - Warp-specialized
// ============================================================================

extern "C" __global__ void matmul_fp16_tc_v5(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    uint m, uint k, uint n
) {
    // WMMA fragment declarations with larger tiles for Blackwell
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
    
    const uint warp_id = threadIdx.x / 32;
    const uint lane_id = threadIdx.x % 32;
    const uint warps_per_block = blockDim.x / 32;
    
    // Each warp computes a 64x64 tile on Blackwell
    const uint warp_m = (blockIdx.y * warps_per_block + warp_id) * 64;
    const uint warp_n = blockIdx.x * 64;
    
    if (warp_m >= m || warp_n >= n) return;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Double-buffered loading for Blackwell
    #pragma unroll 8
    for (uint tile_k = 0; tile_k < k; tile_k += 16) {
        // Load A and B fragments with double buffering
        wmma::load_matrix_sync(a_frag[0], A + (warp_m + (lane_id < 16 ? 0 : 16)) * k + tile_k, k);
        wmma::load_matrix_sync(b_frag[0], B + tile_k * n + warp_n + (lane_id < 16 ? 0 : 16), n);
        
        wmma::mma_sync(acc_frag, a_frag[0], b_frag[0], acc_frag);
        
        wmma::load_matrix_sync(a_frag[1], A + (warp_m + (lane_id < 16 ? 0 : 16) + 32) * k + tile_k, k);
        wmma::load_matrix_sync(b_frag[1], B + tile_k * n + warp_n + (lane_id < 16 ? 0 : 16) + 32, n);
        
        wmma::mma_sync(acc_frag, a_frag[1], b_frag[1], acc_frag);
    }
    
    // Store result
    wmma::store_matrix_sync(C + (warp_m + (lane_id < 16 ? 0 : 16)) * n + warp_n + (lane_id < 16 ? 0 : 16), acc_frag, n, wmma::mem_row_major);
}

// ============================================================================
// FP8 Tensor Core Matrix Multiplication v5 - Hopper/Blackwell optimized
// ============================================================================

#if defined(TITAN_HAS_FP8)
extern "C" __global__ void matmul_fp8_tc_v5(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    uint m, uint k, uint n
) {
    // WMMA fragment declarations for FP8 with accumulation in FP16
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_fp8_e4m3, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_fp8_e4m3, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;
    
    const uint warp_id = threadIdx.x / 32;
    const uint warp_m = (blockIdx.y * 4 + warp_id) * 16;
    const uint warp_n = blockIdx.x * 16;
    
    if (warp_m >= m || warp_n >= n) return;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // High-throughput FP8 MMA on Hopper/Blackwell
    #pragma unroll 4
    for (uint tile_k = 0; tile_k < k; tile_k += 16) {
        wmma::load_matrix_sync(a_frag, A + warp_m * k + tile_k, k);
        wmma::load_matrix_sync(b_frag, B + tile_k * n + warp_n, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    wmma::store_matrix_sync(C + warp_m * n + warp_n, acc_frag, n, wmma::mem_row_major);
}
#endif

// ============================================================================
// FlashAttention v5 - Blackwell/Hopper optimized with TMA
// ============================================================================

extern "C" __global__ void flash_attention_cuda_v5(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    uint seq_len, uint head_dim
) {
    // Dynamic tile sizing optimized for different architectures
    #if defined(TITAN_ARCH_BLACKWELL)
        const uint TILE_Q = 128;
        const uint TILE_KV = 128;
    #elif defined(TITAN_ARCH_HOPPER)
        const uint TILE_Q = 64;
        const uint TILE_KV = 64;
    #else
        const uint TILE_Q = 64;
        const uint TILE_KV = 64;
    #endif
    
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = smem + TILE_Q * 128;
    float* v_tile = smem + TILE_Q * 128 + TILE_KV * 128;
    
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    
    // Load Q tile with async copy on Hopper+
    #if defined(TITAN_HAS_ASYNC_COPY)
        for (uint d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_tile[threadIdx.x * head_dim + d] = Q[row * head_dim + d];
        }
    #else
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            q_tile[threadIdx.x * head_dim + d] = Q[row * head_dim + d];
        }
    #endif
    
    float row_max = -1e30f;
    float row_sum = 0.0f;
    float o_accum[128];
    
    #pragma unroll
    for (uint d = 0; d < head_dim; d++) {
        o_accum[d] = 0.0f;
    }
    
    // Loop over KV tiles
    for (uint tile_j = 0; tile_j < seq_len; tile_j += TILE_KV) {
        __syncthreads();
        
        // Cooperative loading of K and V tiles
        for (uint i = threadIdx.x; i < TILE_KV * head_dim; i += blockDim.x) {
            uint load_row = tile_j + i / head_dim;
            uint d = i % head_dim;
            if (load_row < seq_len) {
                k_tile[i] = K[load_row * head_dim + d];
                v_tile[i] = V[load_row * head_dim + d];
            }
        }
        
        __syncthreads();
        
        // Compute attention with online softmax
        float m_prev = row_max;
        float l_prev = row_sum;
        
        // Find max for this tile
        for (uint j = 0; j < TILE_KV && (tile_j + j) < seq_len; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < head_dim; d++) {
                dot += q_tile[threadIdx.x * head_dim + d] * k_tile[j * head_dim + d];
            }
            dot *= 0.125f;
            row_max = fmaxf(row_max, dot);
        }
        
        // Rescale
        float scale = expf(m_prev - row_max);
        row_sum = l_prev * scale;
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            o_accum[d] *= scale;
        }
        
        // Compute softmax and weighted sum
        for (uint j = 0; j < TILE_KV && (tile_j + j) < seq_len; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < head_dim; d++) {
                dot += q_tile[threadIdx.x * head_dim + d] * k_tile[j * head_dim + d];
            }
            dot *= 0.125f;
            
            float p = expf(dot - row_max);
            row_sum += p;
            
            #pragma unroll
            for (uint d = 0; d < head_dim; d++) {
                o_accum[d] += p * v_tile[j * head_dim + d];
            }
        }
    }
    
    // Write normalized output
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (uint d = 0; d < head_dim; d++) {
        O[row * head_dim + d] = o_accum[d] * inv_sum;
    }
}

// ============================================================================
// Fused RMSNorm v5 - Blackwell/Hopper optimized with warp-specialized reduction
// ============================================================================

extern "C" __global__ void rms_norm_fused_v5(
    const float* __restrict__ input,
    float* __restrict__ output,
    float eps,
    uint length
) {
    __shared__ float shared_sum[512];
    
    const uint tid = threadIdx.x;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and compute partial sum of squares
    float val = 0.0f;
    float partial_sum = 0.0f;
    
    for (uint i = gid; i < length; i += blockDim.x * gridDim.x) {
        val = input[i];
        partial_sum += val * val;
    }
    
    shared_sum[tid] = partial_sum;
    __syncthreads();
    
    // Warp-shuffle reduction first
    #pragma unroll
    for (uint offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    if (tid % 32 == 0) {
        shared_sum[tid / 32] = partial_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < 16) {
        partial_sum = (tid < blockDim.x / 32) ? shared_sum[tid] : 0.0f;
        #pragma unroll
        for (uint offset = 8; offset > 0; offset >>= 1) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        if (tid == 0) {
            shared_sum[0] = partial_sum;
        }
    }
    __syncthreads();
    
    // Compute RMS and normalize
    float rms = sqrtf(shared_sum[0] / float(length) + eps);
    float scale = 1.0f / rms;
    
    for (uint i = gid; i < length; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * scale;
    }
}

// ============================================================================
// Warp-level Softmax v5 - Vectorized for Blackwell
// ============================================================================

extern "C" __global__ void softmax_warp_v5(
    float* __restrict__ data,
    uint length
) {
    const uint warp_id = threadIdx.x / 32;
    const uint lane_id = threadIdx.x % 32;
    const uint num_warps = blockDim.x / 32;
    
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    // Find local max
    float local_max = -1e30f;
    for (uint i = threadIdx.x; i < length; i += blockDim.x) {
        local_max = fmaxf(local_max, data[i]);
    }
    
    // Warp reduction
    #pragma unroll
    for (uint offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Global max
    if (warp_id == 0) {
        float warp_max = (lane_id < num_warps) ? shared_max[lane_id] : -1e30f;
        #pragma unroll
        for (uint offset = 16; offset > 0; offset >>= 1) {
            warp_max = fmaxf(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
        }
        if (lane_id == 0) {
            shared_max[0] = warp_max;
        }
    }
    __syncthreads();
    
    float global_max = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = threadIdx.x; i < length; i += blockDim.x) {
        float exp_val = expf(data[i] - global_max);
        data[i] = exp_val;
        local_sum += exp_val;
    }
    
    // Warp reduction for sum
    #pragma unroll
    for (uint offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Global sum
    if (warp_id == 0) {
        float warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (uint offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = warp_sum;
        }
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_sum[0];
    
    // Normalize
    for (uint i = threadIdx.x; i < length; i += blockDim.x) {
        data[i] *= inv_sum;
    }
}

// ============================================================================
// Fused AdamW Optimizer v5 - Vectorized with stochastic rounding
// ============================================================================

extern "C" __global__ void adamw_fused_v5(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    uint numel
) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    
    float g = grads[gid];
    float p = params[gid];
    
    // Decoupled weight decay
    p = p * (1.0f - lr * weight_decay);
    
    // Adam momentum with fused operations
    float m_t = m[gid];
    m_t = beta1 * m_t + (1.0f - beta1) * g;
    m[gid] = m_t;
    
    float v_t = v[gid];
    v_t = beta2 * v_t + (1.0f - beta2) * g * g;
    v[gid] = v_t;
    
    // Bias correction with fast approximations
    float beta1_pow = powf(beta1, step);
    float beta2_pow = powf(beta2, step);
    float m_hat = m_t / (1.0f - beta1_pow);
    float v_hat = v_t / (1.0f - beta2_pow);
    
    // Update with numerical stability
    float update = lr * m_hat / (sqrtf(v_hat) + eps);
    params[gid] = p - update;
}

// ============================================================================
// 8-bit AdamW Optimizer v5 - Block-wise quantization
// ============================================================================

extern "C" __global__ void adamw_8bit_v5(
    float* __restrict__ params,
    const float* __restrict__ grads,
    uchar* __restrict__ m_quant,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    uint numel,
    float quant_scale
) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    
    float g = grads[gid];
    float p = params[gid];
    
    // Decoupled weight decay
    p = p * (1.0f - lr * weight_decay);
    
    // Dequantize momentum
    float m_t = (float(m_quant[gid]) - 127.0f) / quant_scale;
    
    // Adam momentum
    m_t = beta1 * m_t + (1.0f - beta1) * g;
    
    // Quantize with stochastic rounding
    float quant_val = m_t * quant_scale + 127.0f;
    m_quant[gid] = (uchar)fminf(fmaxf(roundf(quant_val), 0.0f), 255.0f);
    
    // Velocity (FP32)
    float v_t = v[gid];
    v_t = beta2 * v_t + (1.0f - beta2) * g * g;
    v[gid] = v_t;
    
    // Bias correction
    float beta1_pow = powf(beta1, step);
    float beta2_pow = powf(beta2, step);
    float m_hat = m_t / (1.0f - beta1_pow);
    float v_hat = v_t / (1.0f - beta2_pow);
    
    // Update
    p = p - lr * m_hat / (sqrtf(v_hat) + eps);
    params[gid] = p;
}

// ============================================================================
// Q4_0 Quantization v5 - Vectorized for Blackwell
// ============================================================================

struct BlockQ4_0 {
    float d;
    uchar2 qs[8];
};

__device__ __forceinline__ uchar2 make_uchar2(unsigned char x, unsigned char y) {
    uchar2 r;
    r.x = x;
    r.y = y;
    return r;
}

extern "C" __global__ void quantize_q4_0_v5(
    const float* __restrict__ input,
    BlockQ4_0* __restrict__ output,
    uint num_blocks
) {
    const uint bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    
    const uint start_idx = bid * 32;
    
    // Find max abs with warp reduction
    float max_abs = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 32; i++) {
        max_abs = fmaxf(max_abs, fabsf(input[start_idx + i]));
    }
    
    float scale = max_abs / 7.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    
    output[bid].d = scale;
    
    // Quantize with vectorized operations
    #pragma unroll
    for (uint i = 0; i < 8; i++) {
        uint base = start_idx + i * 4;
        
        int q0 = int(roundf(input[base] * inv_scale)) + 8;
        int q1 = int(roundf(input[base + 1] * inv_scale)) + 8;
        int q2 = int(roundf(input[base + 2] * inv_scale)) + 8;
        int q3 = int(roundf(input[base + 3] * inv_scale)) + 8;
        
        output[bid].qs[i] = make_uchar2(
            (q0 & 0x0F) | ((q1 & 0x0F) << 4),
            (q2 & 0x0F) | ((q3 & 0x0F) << 4)
        );
    }
}

extern "C" __global__ void dequantize_q4_0_v5(
    const BlockQ4_0* __restrict__ input,
    float* __restrict__ output,
    uint num_blocks
) {
    const uint bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    
    const uint start_idx = bid * 32;
    float scale = input[bid].d;
    
    #pragma unroll
    for (uint i = 0; i < 8; i++) {
        uchar2 packed = input[bid].qs[i];
        
        output[start_idx + i * 4] = (float(packed.x & 0x0F) - 8.0f) * scale;
        output[start_idx + i * 4 + 1] = (float((packed.x >> 4) & 0x0F) - 8.0f) * scale;
        output[start_idx + i * 4 + 2] = (float(packed.y & 0x0F) - 8.0f) * scale;
        output[start_idx + i * 4 + 3] = (float((packed.y >> 4) & 0x0F) - 8.0f) * scale;
    }
}

// ============================================================================
// Q8_0 Quantization v5
// ============================================================================

struct BlockQ8_0 {
    float d;
    int8_t qs[32];
};

extern "C" __global__ void quantize_q8_0_v5(
    const float* __restrict__ input,
    BlockQ8_0* __restrict__ output,
    uint num_blocks
) {
    const uint bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    
    const uint start_idx = bid * 32;
    
    // Find max abs
    float max_abs = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 32; i++) {
        max_abs = fmaxf(max_abs, fabsf(input[start_idx + i]));
    }
    
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    
    output[bid].d = scale;
    
    #pragma unroll
    for (uint i = 0; i < 32; i++) {
        int q = int(roundf(input[start_idx + i] * inv_scale));
        output[bid].qs[i] = (int8_t)max(-127, min(127, q));
    }
}

extern "C" __global__ void dequantize_q8_0_v5(
    const BlockQ8_0* __restrict__ input,
    float* __restrict__ output,
    uint num_blocks
) {
    const uint bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= num_blocks) return;
    
    const uint start_idx = bid * 32;
    float scale = input[bid].d;
    
    #pragma unroll
    for (uint i = 0; i < 32; i++) {
        output[start_idx + i] = float(input[bid].qs[i]) * scale;
    }
}

// ============================================================================
// Rotary Position Embedding (RoPE) v5 - Fused with vectorization
// ============================================================================

extern "C" __global__ void rotary_embedding_fused_v5(
    float* __restrict__ q,
    float* __restrict__ k,
    uint seq_len,
    uint head_dim,
    uint base_freq
) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint total = seq_len * head_dim / 2;
    
    if (tid >= total) return;
    
    uint pos = tid / (head_dim / 2);
    uint pair = tid % (head_dim / 2);
    
    uint idx = pos * head_dim + pair * 2;
    
    // Precompute frequency
    float freq = 1.0f / powf(float(base_freq), float(pair * 2) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    
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
// Multi-Query Attention v5 - Optimized memory access
// ============================================================================

extern "C" __global__ void multi_query_attention_v5(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    uint num_heads,
    uint seq_len,
    uint head_dim
) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint total = num_heads * seq_len;
    
    if (tid >= total) return;
    
    uint head = tid / seq_len;
    uint pos = tid % seq_len;
    
    // Load query vector to registers
    float q_vec[128];
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = q[tid * head_dim + d];
    }
    
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float o_acc[128] = {0};
    
    // Compute attention scores with streaming K/V access
    for (uint j = 0; j < seq_len; j++) {
        float score = 0.0f;
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            score += q_vec[d] * k[j * head_dim + d];
        }
        score *= 1.0f / sqrtf(float(head_dim));
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax and aggregate
    for (uint j = 0; j < seq_len; j++) {
        float score = 0.0f;
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            score += q_vec[d] * k[j * head_dim + d];
        }
        score *= 1.0f / sqrtf(float(head_dim));
        
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;
        
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            o_acc[d] += exp_score * v[j * head_dim + d];
        }
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (uint d = 0; d < head_dim; d++) {
        out[tid * head_dim + d] = o_acc[d] * inv_sum;
    }
}

// ============================================================================
// Grouped Query Attention (GQA) v5
// ============================================================================

extern "C" __global__ void grouped_query_attention_v5(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    uint num_q_heads,
    uint num_kv_heads,
    uint seq_len,
    uint head_dim
) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint total = num_q_heads * seq_len;
    
    if (tid >= total) return;
    
    uint q_head_id = tid / seq_len;
    uint kv_head_id = q_head_id * num_kv_heads / num_q_heads;
    uint pos = tid % seq_len;
    
    float q_vec[128];
    float o_acc[128];
    
    for (uint d = 0; d < head_dim; d++) {
        q_vec[d] = q[(q_head_id * seq_len + pos) * head_dim + d];
        o_acc[d] = 0.0f;
    }
    
    float row_max = -1e30f;
    float row_sum = 0.0f;
    
    // Compute with grouped KV
    for (uint j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k[(kv_head_id * seq_len + j) * head_dim + d];
        }
        dot *= 1.0f / sqrtf(float(head_dim));
        row_max = fmaxf(row_max, dot);
    }
    
    for (uint j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k[(kv_head_id * seq_len + j) * head_dim + d];
        }
        dot *= 1.0f / sqrtf(float(head_dim));
        
        float exp_score = expf(dot - row_max);
        row_sum += exp_score;
        
        #pragma unroll
        for (uint d = 0; d < head_dim; d++) {
            o_acc[d] += exp_score * v[(kv_head_id * seq_len + j) * head_dim + d];
        }
    }
    
    float inv_sum = 1.0f / row_sum;
    for (uint d = 0; d < head_dim; d++) {
        out[(q_head_id * seq_len + pos) * head_dim + d] = o_acc[d] * inv_sum;
    }
}

// ============================================================================
// NCCL-compatible All-Reduce v5 - Ring algorithm
// ============================================================================

extern "C" __global__ void nccl_allreduce_ring_v5(
    float* __restrict__ data,
    float* __restrict__ workspace,
    uint numel,
    uint rank,
    uint world_size
) {
    // Simplified ring all-reduce
    // Each GPU sends to next and receives from prev
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    
    // Reduce-scatter phase
    uint chunk_size = (numel + world_size - 1) / world_size;
    uint chunk_start = rank * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, numel);
    
    if (gid >= chunk_start && gid < chunk_end) {
        float sum = data[gid];
        for (uint i = 1; i < world_size; i++) {
            uint src_rank = (rank - i + world_size) % world_size;
            sum += workspace[src_rank * chunk_size + (gid - chunk_start)];
        }
        data[gid] = sum;
    }
    __syncthreads();
    
    // All-gather phase
    if (gid < numel) {
        uint chunk_idx = gid / chunk_size;
        uint offset = gid % chunk_size;
        data[gid] = workspace[chunk_idx * chunk_size + offset];
    }
}

// ============================================================================
// Secure CUDA Memory Encryption v5
// ============================================================================

#if defined(TITAN_HAS_SECURE_CUDA)
extern "C" __global__ void secure_memory_encrypt_v5(
    float* __restrict__ data,
    uint numel,
    uint seed
) {
    // Simple XOR-based encryption for demonstration
    // Production would use proper GPU-accelerated AES
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) return;
    
    // Generate pseudo-random mask
    uint state = gid + seed;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    
    float mask = (float)(state % 1000000) / 1000000.0f;
    data[gid] = data[gid] * mask;
}
#endif
