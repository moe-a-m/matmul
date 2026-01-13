#include <stddef.h>

void tt_matmul_c(
    const float* a,
    const float* b,
    float* c,
    size_t M,
    size_t N,
    size_t K
) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += a[i*K + k] * b[k*N + j];
            }
            c[i*N + j] = acc;
        }
    }
}