#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" float titan_dot_f32(const float* lhs, const float* rhs, std::size_t len) {
    float acc = 0.0f;
    std::size_t i = 0;

#if defined(__AVX2__)
    __m256 sum256 = _mm256_setzero_ps();
    std::size_t vec_len = len & ~7;
    for (; i < vec_len; i += 8) {
        __m256 v_lhs = _mm256_loadu_ps(lhs + i);
        __m256 v_rhs = _mm256_loadu_ps(rhs + i);
        sum256 = _mm256_fmadd_ps(v_lhs, v_rhs, sum256);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, sum256);
    for (int j = 0; j < 8; ++j) {
        acc += buffer[j];
    }
#elif defined(__ARM_NEON)
    float32x4_t sum128 = vdupq_n_f32(0.0f);
    std::size_t vec_len = len & ~3;
    for (; i < vec_len; i += 4) {
        float32x4_t v_lhs = vld1q_f32(lhs + i);
        float32x4_t v_rhs = vld1q_f32(rhs + i);
        sum128 = vmlaq_f32(sum128, v_lhs, v_rhs);
    }
    acc += vgetq_lane_f32(sum128, 0);
    acc += vgetq_lane_f32(sum128, 1);
    acc += vgetq_lane_f32(sum128, 2);
    acc += vgetq_lane_f32(sum128, 3);
#endif

    for (; i < len; ++i) {
        acc += lhs[i] * rhs[i];
    }
    return acc;
}

extern "C" void titan_matmul_f32(
    const float* lhs,
    std::size_t m,
    std::size_t k,
    const float* rhs,
    std::size_t n,
    float* out
) {
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (std::size_t inner = 0; inner < k; ++inner) {
                acc += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}
