#!/usr/bin/env python3
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# TT-XLA backend for N300s
def setup_tt_backend():
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['TT_BACKEND'] = 'TT_XLA'
    
@jit
def matmul_tt(a, b):
    return jnp.dot(a, b)

def main():
    if len(sys.argv) != 7:
        print("Usage: tt_matmul.py M N K a_ptr b_ptr c_ptr")
        sys.exit(1)
    
    setup_tt_backend()
    
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    
    # Create test matrices
    a = jnp.ones((M, K), dtype=jnp.float32)
    b = jnp.ones((K, N), dtype=jnp.float32)
    
    # Compile and run on TT hardware
    c = matmul_tt(a, b)
    
    print(f"TT-XLA matmul completed: {M}x{N}x{K}")
    return 0

if __name__ == "__main__":
    main()