#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f64(
    target_ptr: *mut f64,
    source_ptr: *mut f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
) {
    use crate::evaluation::direct::assemble_laplace_kernel;

    let target = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    assemble_laplace_kernel(&target, &sources, &mut result);
}

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f32(
    target_ptr: *mut f32,
    source_ptr: *mut f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
) {
    use crate::evaluation::direct::assemble_laplace_kernel;

    let target = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    assemble_laplace_kernel(&target, &sources, &mut result);
}
