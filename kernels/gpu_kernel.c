#include <stdint.h>
#include <stddef.h>
#include <string.h>

// TT-XLA optimized kernel for N300s
void matmul_tiled(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    const int M,
    const int N,
    const int K
) {
    // Zero output
    memset(C, 0, M * N * sizeof(float));
    
    // Tiled implementation optimized for TT N300s
    const int TILE_SIZE = 64;
    
    for (int ii = 0; ii < M; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < K; kk += TILE_SIZE) {
                int m_end = (ii + TILE_SIZE < M) ? ii + TILE_SIZE : M;
                int n_end = (jj + TILE_SIZE < N) ? jj + TILE_SIZE : N;
                int k_end = (kk + TILE_SIZE < K) ? kk + TILE_SIZE : K;
                
                for (int i = ii; i < m_end; i++) {
                    for (int j = jj; j < n_end; j++) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_end; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}