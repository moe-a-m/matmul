#include <tt_metal/host_api.hpp>
#include <tt_metal/detail/tt_metal.hpp>
#include <tt_metal/impl/kernels/kernel.hpp>

using namespace tt;
using namespace tt::tt_metal;

void tt_matmul_c(
    const float* a,
    const float* b,
    float* c,
    size_t M,
    size_t N,
    size_t K
) {
    try {
        Device *device = CreateDevice(0);
        if (!device) {
            // Fallback to CPU
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        acc += a[i*K + k] * b[k*N + j];
                    }
                    c[i*N + j] = acc;
                }
            }
            return;
        }

        Program program = CreateProgram();
        
        // Create buffers on device
        auto a_buffer = CreateBuffer(device, M * K * sizeof(float), BufferType::DRAM);
        auto b_buffer = CreateBuffer(device, K * N * sizeof(float), BufferType::DRAM);
        auto c_buffer = CreateBuffer(device, M * N * sizeof(float), BufferType::DRAM);
        
        // Copy data to device
        EnqueueWriteBuffer(device->command_queue(), a_buffer, a, false);
        EnqueueWriteBuffer(device->command_queue(), b_buffer, b, false);
        
        // Launch kernel
        auto kernel = CreateKernel(
            program,
            "kernels/matmul_tt.cpp",
            CoreCoord{0, 0}
        );
        
        SetRuntimeArgs(kernel, {a_buffer->address(), b_buffer->address(), c_buffer->address(), M, N, K});
        EnqueueProgram(device->command_queue(), program, false);
        
        // Read result back
        EnqueueReadBuffer(device->command_queue(), c_buffer, c, true);
        
        CloseDevice(device);
        
    } catch (...) {
        // Fallback to CPU on any error
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
}