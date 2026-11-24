#ifndef MATMUL_KERNEL_
#define MATMUL_KERNEL_

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
/* RVV intrinsics were renamed with a __riscv_ prefix in the v0.11+ spec
 * (GCC 13+, Clang 17+). RVV(x) picks the right spelling for the compiler. */
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 11000
#define RVV(x) __riscv_##x
#else
#define RVV(x) x
#endif
#endif

/* ===========================
 * Tunables
 * =========================== */
#define TILE_M 4
#define TILE_N 64
#define TILE_K 64

#define UNROLL 4   /* set to 1, 2, or 4 */

#pragma once

void matmul_kernel_c(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);


#endif

