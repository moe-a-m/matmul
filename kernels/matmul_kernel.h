#ifndef MATMUL_KERNEL_
#define MATMUL_KERNEL_

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
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

