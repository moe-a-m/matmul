#include <stddef.h>
#include <string.h>
#include <immintrin.h>

// 4x4 register microkernel using AVX2
static inline void microkernel_4x4(
    const float* restrict a,
    const float* restrict b,
    float* restrict c,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc
) {
    __m256 c0 = _mm256_loadu_ps(&c[0 * ldc]);
    __m256 c1 = _mm256_loadu_ps(&c[1 * ldc]);
    __m256 c2 = _mm256_loadu_ps(&c[2 * ldc]);
    __m256 c3 = _mm256_loadu_ps(&c[3 * ldc]);
    
    for (size_t k = 0; k < K; k++) {
        __m256 b_vec = _mm256_loadu_ps(&b[k * ldb]);
        
        __m256 a0 = _mm256_broadcast_ss(&a[0 * lda + k]);
        __m256 a1 = _mm256_broadcast_ss(&a[1 * lda + k]);
        __m256 a2 = _mm256_broadcast_ss(&a[2 * lda + k]);
        __m256 a3 = _mm256_broadcast_ss(&a[3 * lda + k]);
        
        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
    }
    
    _mm256_storeu_ps(&c[0 * ldc], c0);
    _mm256_storeu_ps(&c[1 * ldc], c1);
    _mm256_storeu_ps(&c[2 * ldc], c2);
    _mm256_storeu_ps(&c[3 * ldc], c3);
}

void tt_matmul_c(
    const float* a,
    const float* b,
    float* c,
    size_t M,
    size_t N,
    size_t K
) {
    memset(c, 0, M * N * sizeof(float));
    
    // L3 cache blocking (optimized for typical CPU)
    const size_t MC = 256;  // M dimension block
    const size_t NC = 4096; // N dimension block  
    const size_t KC = 256;  // K dimension block
    
    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = (jc + NC < N) ? NC : N - jc;
        
        for (size_t pc = 0; pc < K; pc += KC) {
            size_t kc = (pc + KC < K) ? KC : K - pc;
            
            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = (ic + MC < M) ? MC : M - ic;
                
                // L1 cache blocking with microkernel
                const size_t MR = 4;  // Microkernel M dimension
                const size_t NR = 8;  // Microkernel N dimension (AVX2 width)
                
                for (size_t jr = 0; jr < nc; jr += NR) {
                    size_t nr = (jr + NR < nc) ? NR : nc - jr;
                    
                    for (size_t ir = 0; ir < mc; ir += MR) {
                        size_t mr = (ir + MR < mc) ? MR : mc - ir;
                        
                        if (mr == 4 && nr == 8) {
                            // Use optimized 4x8 microkernel
                            microkernel_4x4(
                                &a[(ic + ir) * K + pc],
                                &b[pc * N + (jc + jr)],
                                &c[(ic + ir) * N + (jc + jr)],
                                kc, K, N, N
                            );
                        } else {
                            // Fallback for edge cases
                            for (size_t i = 0; i < mr; i++) {
                                for (size_t j = 0; j < nr; j++) {
                                    float sum = c[(ic + ir + i) * N + (jc + jr + j)];
                                    for (size_t k = 0; k < kc; k++) {
                                        sum += a[(ic + ir + i) * K + (pc + k)] * 
                                               b[(pc + k) * N + (jc + jr + j)];
                                    }
                                    c[(ic + ir + i) * N + (jc + jr + j)] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}