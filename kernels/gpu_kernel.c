#include <stdint.h>
#include <stddef.h>

// TensorTorrent GPU kernel stub - replace with actual TT Metal API calls
void matmul_tiled(
    const float* restrict A,
    const float* restrict B,
    float* restrict C,
    const int M,
    const int N,
    const int K
) {
    // Fallback CPU implementation when TT headers unavailable
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}