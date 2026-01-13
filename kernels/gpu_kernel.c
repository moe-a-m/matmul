#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>

void matmul_tiled(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    const int M,
    const int N,
    const int K
) {
    memset(C, 0, M * N * sizeof(float));
    
    const int BLOCK_SIZE = 64;
    
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                int m_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
                int k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                int n_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
                
                for (int i = ii; i < m_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[i * K + k];
                        int j = jj;
                        
                        // Vectorized inner loop
                        for (; j <= n_end - 8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 a_vec = _mm256_set1_ps(a_val);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }
                        
                        // Handle remaining elements
                        for (; j < n_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}