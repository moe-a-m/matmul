#include "matmul_kernel.h"
/* ===========================
 * Counters
 * =========================== */
static inline uint64_t read_cycle(void) {
#if defined(__riscv)
    uint64_t x;
    asm volatile("read_cycle() %0" : "=r"(x));
    return x;
#else
    return 0;
#endif
}

static inline uint64_t read_instret(void) {
#if defined(__riscv)
    uint64_t x;
    asm volatile("read_instret() %0" : "=r"(x));
    return x;
#else
    return 0;
#endif
}


/* ===========================
 * Packing
 * =========================== */

static void pack_B(const float *B, float *Bp, size_t n, size_t kk, size_t jj,
                   size_t kt, size_t nt) {
  for (size_t l = 0; l < kt; l++) {
    memcpy(&Bp[l * nt], &B[(kk + l) * n + jj], nt * sizeof(float));
  }
}

static void pack_A(const float *A, float *Ap, size_t k, size_t ii, size_t kk,
                   size_t mt, size_t kt) {
  for (size_t i = 0; i < mt; i++) {
    memcpy(&Ap[i * kt], &A[(ii + i) * k + kk], kt * sizeof(float));
  }
}

/* ===========================
 * Kernel
 * =========================== */

void matmul_kernel_c(const float *A, const float *B, float *C, size_t m,
                     size_t n, size_t k) {
  memset(C, 0, m * n * sizeof(float));

  float *Bp = NULL;
  float *Ap = NULL;
  if (posix_memalign((void**)&Bp, 64, TILE_K * TILE_N * sizeof(float)) != 0) return;
  if (posix_memalign((void**)&Ap, 64, TILE_M * TILE_K * sizeof(float)) != 0) {
    free(Bp);
    return;
  }

  uint64_t c0 = read_cycle();
  uint64_t i0 = read_instret();

  for (size_t ii = 0; ii < m; ii += TILE_M) {
    for (size_t jj = 0; jj < n; jj += TILE_N) {
      for (size_t kk = 0; kk < k; kk += TILE_K) {

        size_t mt = (ii + TILE_M < m) ? TILE_M : (m - ii);
        size_t nt = (jj + TILE_N < n) ? TILE_N : (n - jj);
        size_t kt = (kk + TILE_K < k) ? TILE_K : (k - kk);

        pack_B(B, Bp, n, kk, jj, kt, nt);
        pack_A(A, Ap, k, ii, kk, mt, kt);

        for (size_t i = 0; i < mt; i++) {
          size_t j = 0;

          while (j < nt) {
#ifdef __riscv_vector
            size_t vl = vsetvl_e32m1(nt - j);

#if UNROLL == 4
            vfloat32m1_t acc0 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 0 * vl)], vl);
            vfloat32m1_t acc1 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 1 * vl)], vl);
            vfloat32m1_t acc2 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 2 * vl)], vl);
            vfloat32m1_t acc3 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 3 * vl)], vl);
#elif UNROLL == 2
            vfloat32m1_t acc0 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 0 * vl)], vl);
            vfloat32m1_t acc1 =
                vle32_v_f32m1(&C[(ii + i) * n + (jj + j + 1 * vl)], vl);
#else
            vfloat32m1_t acc0 = vle32_v_f32m1(&C[(ii + i) * n + (jj + j)], vl);
#endif

            for (size_t l = 0; l < kt; l++) {
              float aval = Ap[i * kt + l];

              __builtin_prefetch(&Bp[(l + 1) * nt + j], 0, 3);

#if UNROLL == 4
              vfloat32m1_t b0 = vle32_v_f32m1(&Bp[l * nt + j + 0 * vl], vl);
              vfloat32m1_t b1 = vle32_v_f32m1(&Bp[l * nt + j + 1 * vl], vl);
              vfloat32m1_t b2 = vle32_v_f32m1(&Bp[l * nt + j + 2 * vl], vl);
              vfloat32m1_t b3 = vle32_v_f32m1(&Bp[l * nt + j + 3 * vl], vl);
              acc0 = vfmacc_vf_f32m1(acc0, aval, b0, vl);
              acc1 = vfmacc_vf_f32m1(acc1, aval, b1, vl);
              acc2 = vfmacc_vf_f32m1(acc2, aval, b2, vl);
              acc3 = vfmacc_vf_f32m1(acc3, aval, b3, vl);
#elif UNROLL == 2
              vfloat32m1_t b0 = vle32_v_f32m1(&Bp[l * nt + j + 0 * vl], vl);
              vfloat32m1_t b1 = vle32_v_f32m1(&Bp[l * nt + j + 1 * vl], vl);
              acc0 = vfmacc_vf_f32m1(acc0, aval, b0, vl);
              acc1 = vfmacc_vf_f32m1(acc1, aval, b1, vl);
#else
              vfloat32m1_t b0 = vle32_v_f32m1(&Bp[l * nt + j], vl);
              acc0 = vfmacc_vf_f32m1(acc0, aval, b0, vl);
#endif
            }

#if UNROLL == 4
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 0 * vl)], acc0, vl);
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 1 * vl)], acc1, vl);
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 2 * vl)], acc2, vl);
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 3 * vl)], acc3, vl);
            j += 4 * vl;
#elif UNROLL == 2
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 0 * vl)], acc0, vl);
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j + 1 * vl)], acc1, vl);
            j += 2 * vl;
#else
            vse32_v_f32m1(&C[(ii + i) * n + (jj + j)], acc0, vl);
            j += vl;
#endif

#else
            for (; j < nt; j++) {
              float sum = C[(ii + i) * n + (jj + j)];
              for (size_t l = 0; l < kt; l++) {
                sum += Ap[i * kt + l] * Bp[l * nt + j];
              }
              C[(ii + i) * n + (jj + j)] = sum;
            }
#endif
          }
        }
      }
    }
  }

  uint64_t c1 = read_cycle();
  uint64_t i1 = read_instret();

  /* Optional: expose these via globals if needed */
  (void)c0;
  (void)c1;
  (void)i0;
  (void)i1;

  free(Ap);
  free(Bp);
}
