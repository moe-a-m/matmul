#[cfg(feature = "gpu")]
mod gpu_impl {
    use std::ffi::c_void;

    #[link(name = "tt_metal")]
    extern "C" {
        fn tt_device_open(device_id: u32) -> *mut c_void;
        fn tt_device_close(device: *mut c_void);
        fn tt_malloc(device: *mut c_void, size: usize) -> *mut c_void;
        fn tt_free(device: *mut c_void, ptr: *mut c_void);
        fn tt_memcpy_h2d(device: *mut c_void, dst: *mut c_void, src: *const c_void, size: usize);
        fn tt_memcpy_d2h(device: *mut c_void, dst: *mut c_void, src: *const c_void, size: usize);
        fn tt_launch_kernel(
            device: *mut c_void,
            kernel_name: *const u8,
            grid_x: u32, grid_y: u32,
            block_x: u32, block_y: u32,
            args: *const *const c_void,
            arg_count: u32,
        );
    }

    const TILE_SIZE: usize = 32;

    pub fn matmul_gpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        unsafe {
            let device = tt_device_open(0);
            if device.is_null() {
                super::fallback_matmul(a, b, c, m, n, k);
                return;
            }

            let size_a = m * k * std::mem::size_of::<f32>();
            let size_b = k * n * std::mem::size_of::<f32>();
            let size_c = m * n * std::mem::size_of::<f32>();

            let d_a = tt_malloc(device, size_a);
            let d_b = tt_malloc(device, size_b);
            let d_c = tt_malloc(device, size_c);

            if d_a.is_null() || d_b.is_null() || d_c.is_null() {
                cleanup_gpu_memory(device, d_a, d_b, d_c);
                super::fallback_matmul(a, b, c, m, n, k);
                return;
            }

            tt_memcpy_h2d(device, d_a, a.as_ptr() as *const c_void, size_a);
            tt_memcpy_h2d(device, d_b, b.as_ptr() as *const c_void, size_b);

            let grid_x = (n + TILE_SIZE - 1) / TILE_SIZE;
            let grid_y = (m + TILE_SIZE - 1) / TILE_SIZE;

            let args = [
                &d_a as *const _ as *const c_void,
                &d_b as *const _ as *const c_void,
                &d_c as *const _ as *const c_void,
                &m as *const _ as *const c_void,
                &n as *const _ as *const c_void,
                &k as *const _ as *const c_void,
            ];

            tt_launch_kernel(
                device,
                b"matmul_tiled\0".as_ptr(),
                grid_x as u32, grid_y as u32,
                TILE_SIZE as u32, TILE_SIZE as u32,
                args.as_ptr(),
                args.len() as u32,
            );

            tt_memcpy_d2h(device, c.as_mut_ptr() as *mut c_void, d_c, size_c);

            cleanup_gpu_memory(device, d_a, d_b, d_c);
            tt_device_close(device);
        }
    }

    unsafe fn cleanup_gpu_memory(device: *mut c_void, d_a: *mut c_void, d_b: *mut c_void, d_c: *mut c_void) {
        if !d_a.is_null() { tt_free(device, d_a); }
        if !d_b.is_null() { tt_free(device, d_b); }
        if !d_c.is_null() { tt_free(device, d_c); }
    }
}

fn fallback_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(feature = "gpu")]
    {
        gpu_impl::matmul_gpu(a, b, c, m, n, k);
    }
    #[cfg(not(feature = "gpu"))]
    {
        fallback_matmul(a, b, c, m, n, k);
    }
}