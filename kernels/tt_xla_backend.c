// tt_matmul_blis.c
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define CACHELINE 64

// BLIS-style blocking parameters (safe defaults)
#define MR 6
#define NR 16

#define MC 192
#define NC 256
#define KC 256

// ------------------------------------------------------------
// Pack B: layout = [jr][k][NR]
// ------------------------------------------------------------
static inline void pack_b(
    const float* B,
    float* PB,
    size_t N,
    size_t kc,
    size_t nc,
    size_t pc,
    size_t jc
) {
    for (size_t jr = 0; jr < nc; jr += NR) {
        size_t nr = (jr + NR <= nc) ? NR : nc - jr;

        for (size_t k = 0; k < kc; k++) {
            float* dst = PB + jr * kc + k * NR;
            const float* src = B + (pc + k) * N + (jc + jr);

            // copy valid
            for (size_t j = 0; j < nr; j++)
                dst[j] = src[j];

            // pad
            for (size_t j = nr; j < NR; j++)
                dst[j] = 0.0f;
        }
    }
}

// ------------------------------------------------------------
// 6x16 AVX2 microkernel
// ------------------------------------------------------------
static inline void microkernel_6x16(
    const float* restrict A,
    const float* restrict PB,
    float* restrict C,
    size_t kc,
    size_t lda,
    size_t ldc
) {
    __m256 c[MR][2];

    // load C
    for (int i = 0; i < MR; i++) {
        c[i][0] = _mm256_loadu_ps(&C[i * ldc + 0]);
        c[i][1] = _mm256_loadu_ps(&C[i * ldc + 8]);
    }

    // compute
    for (size_t k = 0; k < kc; k++) {
        __m256 b0 = _mm256_load_ps(PB + k * NR + 0);
        __m256 b1 = _mm256_load_ps(PB + k * NR + 8);

        for (int i = 0; i < MR; i++) {
            __m256 a = _mm256_broadcast_ss(&A[i * lda + k]);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }

    // store C
    for (int i = 0; i < MR; i++) {
        _mm256_storeu_ps(&C[i * ldc + 0], c[i][0]);
        _mm256_storeu_ps(&C[i * ldc + 8], c[i][1]);
    }
}

// ------------------------------------------------------------
// Reference BLIS-style GEMM
// C := A * B
// ------------------------------------------------------------
void tt_matmul_c(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
) {
    memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel
    {
        float* PB = aligned_alloc(CACHELINE, NC * KC * sizeof(float));
        if (!PB) return;

        #pragma omp for schedule(static)
        for (size_t jc = 0; jc < N; jc += NC) {
            size_t nc = (jc + NC <= N) ? NC : N - jc;

            for (size_t pc = 0; pc < K; pc += KC) {
                size_t kc = (pc + KC <= K) ? KC : K - pc;

                pack_b(B, PB, N, kc, nc, pc, jc);

                for (size_t ic = 0; ic < M; ic += MC) {
                    size_t mc = (ic + MC <= M) ? MC : M - ic;

                    for (size_t jr = 0; jr < nc; jr += NR) {
                        for (size_t ir = 0; ir < mc; ir += MR) {

                            if (ir + MR <= mc && jr + NR <= nc) {
                                microkernel_6x16(
                                    &A[(ic + ir) * K + pc],
                                    PB + jr * kc,
                                    &C[(ic + ir) * N + (jc + jr)],
                                    kc,
                                    K,
                                    N
                                );
                            } else {
                                // ---- scalar remainder (correctness) ----
                                for (size_t i = 0; i < MR && ir + i < mc; i++) {
                                    for (size_t j = 0; j < NR && jr + j < nc; j++) {
                                        float sum = C[(ic + ir + i) * N + (jc + jr + j)];
                                        for (size_t k = 0; k < kc; k++) {
                                            sum +=
                                                A[(ic + ir + i) * K + (pc + k)] *
                                                B[(pc + k) * N + (jc + jr + j)];
                                        }
                                        C[(ic + ir + i) * N + (jc + jr + j)] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        free(PB);
    }
}
