#[no_mangle]
pub extern "C" fn laplace_kernel_f64(
    target_ptr: *mut f64,
    source_ptr: *mut f64,
    result_ptr: *mut f64,
    nsources: usize,
) {
    use crate::kernels::laplace_kernel;

    let target = unsafe { ndarray::ArrayView1::from_shape_ptr(3, target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut1::from_shape_ptr(nsources, result_ptr) };

    laplace_kernel(&target, &sources, &mut result);
}


#[no_mangle]
pub extern "C" fn laplace_kernel_f32(
    target_ptr: *mut f32,
    source_ptr: *mut f32,
    result_ptr: *mut f32,
    nsources: usize,
) {
    use crate::kernels::laplace_kernel;

    let target = unsafe { ndarray::ArrayView1::from_shape_ptr(3, target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut1::from_shape_ptr(nsources, result_ptr) };

    laplace_kernel(&target, &sources, &mut result);
}
