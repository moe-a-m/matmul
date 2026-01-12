#ifndef MATMUL_KERNEL_
#define MATMUL_KERNEL_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

/* ===========================
 * Tunables
 * =========================== */
#define TILE_M 4
#define TILE_N 64
#define TILE_K 64

#define UNROLL 4   /* set to 1, 2, or 4 */


#endif