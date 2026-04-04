#include <cstddef>

extern "C" float titan_dot_f32(const float* lhs, const float* rhs, std::size_t len) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
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
