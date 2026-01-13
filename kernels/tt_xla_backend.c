#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

// Memory alignment for cache line optimization
#define CACHE_LINE_SIZE 64
#define ALIGN_TO_CACHE_LINE __attribute__((aligned(CACHE_LINE_SIZE)))

// Thread-local buffer for B matrix packing
static __thread float* packed_b_buffer = NULL;
static __thread size_t packed_b_size = 0;

// Pack B matrix for better cache utilization
static void pack_b_matrix(
    const float* b, 
    float* packed_b,
    size_t K, 
    size_t N,
    size_t kc,
    size_t nc,
    size_t pc,
    size_t jc
) {
    const size_t NR = 16;
    
    for (size_t jr = 0; jr < nc; jr += NR) {
        size_t nr = (jr + NR < nc) ? NR : nc - jr;
        
        for (size_t k = 0; k < kc; k++) {
            for (size_t j = 0; j < nr; j++) {
                packed_b[jr * kc + k * NR + j] = b[(pc + k) * N + (jc + jr + j)];
            }
            // Pad to maintain alignment
            for (size_t j = nr; j < NR; j++) {
                packed_b[jr * kc + k * NR + j] = 0.0f;
            }
        }
    }
}

// Optimized 6x16 microkernel with packed B
static inline void microkernel_6x16_packed(
    const float* restrict a,
    const float* restrict packed_b,
    float* restrict c,
    size_t K,
    size_t lda,
    size_t ldc,
    size_t jr
) {
    __m256 c00 = _mm256_loadu_ps(&c[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&c[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&c[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&c[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&c[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&c[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&c[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&c[3 * ldc + 8]);
    __m256 c40 = _mm256_loadu_ps(&c[4 * ldc + 0]);
    __m256 c41 = _mm256_loadu_ps(&c[4 * ldc + 8]);
    __m256 c50 = _mm256_loadu_ps(&c[5 * ldc + 0]);
    __m256 c51 = _mm256_loadu_ps(&c[5 * ldc + 8]);
    
    for (size_t k = 0; k < K; k++) {
        __m256 b0 = _mm256_load_ps(&packed_b[jr * K + k * 16 + 0]);
        __m256 b1 = _mm256_load_ps(&packed_b[jr * K + k * 16 + 8]);
        
        __m256 a0 = _mm256_broadcast_ss(&a[0 * lda + k]);
        __m256 a1 = _mm256_broadcast_ss(&a[1 * lda + k]);
        __m256 a2 = _mm256_broadcast_ss(&a[2 * lda + k]);
        __m256 a3 = _mm256_broadcast_ss(&a[3 * lda + k]);
        __m256 a4 = _mm256_broadcast_ss(&a[4 * lda + k]);
        __m256 a5 = _mm256_broadcast_ss(&a[5 * lda + k]);
        
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }
    
    _mm256_storeu_ps(&c[0 * ldc + 0], c00);
    _mm256_storeu_ps(&c[0 * ldc + 8], c01);
    _mm256_storeu_ps(&c[1 * ldc + 0], c10);
    _mm256_storeu_ps(&c[1 * ldc + 8], c11);
    _mm256_storeu_ps(&c[2 * ldc + 0], c20);
    _mm256_storeu_ps(&c[2 * ldc + 8], c21);
    _mm256_storeu_ps(&c[3 * ldc + 0], c30);
    _mm256_storeu_ps(&c[3 * ldc + 8], c31);
    _mm256_storeu_ps(&c[4 * ldc + 0], c40);
    _mm256_storeu_ps(&c[4 * ldc + 8], c41);
    _mm256_storeu_ps(&c[5 * ldc + 0], c50);
    _mm256_storeu_ps(&c[5 * ldc + 8], c51);
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
    
    const size_t MC = 384;
    const size_t NC = 4080;
    const size_t KC = 384;
    
    // Parallel outer loop with dynamic scheduling
    #pragma omp parallel
    {
        // Thread-local packed B buffer
        int tid = omp_get_thread_num();
        size_t required_size = NC * KC * sizeof(float);
        
        if (packed_b_size < required_size) {
            if (packed_b_buffer) free(packed_b_buffer);
            packed_b_buffer = (float*)aligned_alloc(CACHE_LINE_SIZE, required_size);
            packed_b_size = required_size;
        }
        
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (size_t jc = 0; jc < N; jc += NC) {
            for (size_t pc = 0; pc < K; pc += KC) {
                size_t nc = (jc + NC < N) ? NC : N - jc;
                size_t kc = (pc + KC < K) ? KC : K - pc;
                
                // Pack B matrix once per thread
                pack_b_matrix(b, packed_b_buffer, K, N, kc, nc, pc, jc);
                
                for (size_t ic = 0; ic < M; ic += MC) {
                    size_t mc = (ic + MC < M) ? MC : M - ic;
                    
                    const size_t MR = 6;
                    const size_t NR = 16;
                    
                    for (size_t jr = 0; jr < nc; jr += NR) {
                        size_t nr = (jr + NR < nc) ? NR : nc - jr;
                        
                        for (size_t ir = 0; ir < mc; ir += MR) {
                            size_t mr = (ir + MR < mc) ? MR : mc - ir;
                            
                            if (mr == 6 && nr == 16) {
                                microkernel_6x16_packed(
                                    &a[(ic + ir) * K + pc],
                                    packed_b_buffer,
                                    &c[(ic + ir) * N + (jc + jr)],
                                    kc, K, N, jr
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}