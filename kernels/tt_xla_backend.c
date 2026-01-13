#include <stddef.h>
#include <string.h>

// Optimized implementation for TensorTorrent N300s
void tt_matmul_c(
    const float* a,
    const float* b,
    float* c,
    size_t M,
    size_t N,
    size_t K
) {
    // Zero output
    memset(c, 0, M * N * sizeof(float));
    
    // Cache-optimized tiling for N300s
    const size_t TILE_M = 32;
    const size_t TILE_N = 32;
    const size_t TILE_K = 64;
    
    for (size_t ii = 0; ii < M; ii += TILE_M) {
        for (size_t jj = 0; jj < N; jj += TILE_N) {
            for (size_t kk = 0; kk < K; kk += TILE_K) {
                size_t m_end = (ii + TILE_M < M) ? ii + TILE_M : M;
                size_t n_end = (jj + TILE_N < N) ? jj + TILE_N : N;
                size_t k_end = (kk + TILE_K < K) ? kk + TILE_K : K;
                
                // Micro-kernel optimized for TT architecture
                for (size_t i = ii; i < m_end; i++) {
                    for (size_t k = kk; k < k_end; k++) {
                        float a_val = a[i * K + k];
                        size_t c_idx = i * N + jj;
                        size_t b_idx = k * N + jj;
                        
                        // Unrolled inner loop for better ILP
                        size_t j = jj;
                        for (; j + 4 <= n_end; j += 4) {
                            c[c_idx] += a_val * b[b_idx];
                            c[c_idx + 1] += a_val * b[b_idx + 1];
                            c[c_idx + 2] += a_val * b[b_idx + 2];
                            c[c_idx + 3] += a_val * b[b_idx + 3];
                            c_idx += 4;
                            b_idx += 4;
                        }
                        
                        // Handle remaining elements
                        for (; j < n_end; j++) {
                            c[c_idx++] += a_val * b[b_idx++];
                        }
                    }
                }
            }
        }
    }
}