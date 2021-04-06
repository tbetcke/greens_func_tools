#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f64(
    target_ptr: *const f64,
    source_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::GftBase;
    use crate::evaluation::direct::DirectEvaluator;

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    GftBase::<f64>::assemble_laplace_kernel(&targets, &sources, &mut result, parallel);
}

#[no_mangle]
pub extern "C" fn assemble_laplace_kernel_f32(
    target_ptr: *const f32,
    source_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::GftBase;
    use crate::evaluation::direct::DirectEvaluator;

    let target = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let mut result = unsafe { ndarray::ArrayViewMut2::from_shape_ptr((ntargets, nsources), result_ptr) };

    GftBase::<f32>::assemble_laplace_kernel(&target, &sources, &mut result, parallel);
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f64(
    target_ptr: *const f64,
    source_ptr: *const f64,
    charge_ptr: *const f64,
    result_ptr: *mut f64,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::GftBase;
    use crate::evaluation::direct::DirectEvaluator;

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe { ndarray::ArrayView1::from_shape_ptr(nsources, charge_ptr)};
    let mut result = unsafe { ndarray::ArrayViewMut1::from_shape_ptr(ntargets, result_ptr) };

    GftBase::<f64>::evaluate_laplace_kernel(&targets, &sources, &charges, &mut result, parallel);
}

#[no_mangle]
pub extern "C" fn evaluate_laplace_kernel_f32(
    target_ptr: *const f32,
    source_ptr: *const f32,
    charge_ptr: *const f32,
    result_ptr: *mut f32,
    nsources: usize,
    ntargets: usize,
    parallel: bool,
) {
    use crate::GftBase;
    use crate::evaluation::direct::DirectEvaluator;

    let targets = unsafe { ndarray::ArrayView2::from_shape_ptr((3, ntargets), target_ptr) };
    let sources = unsafe { ndarray::ArrayView2::from_shape_ptr((3, nsources), source_ptr) };
    let charges = unsafe { ndarray::ArrayView1::from_shape_ptr(nsources, charge_ptr)};
    let mut result = unsafe { ndarray::ArrayViewMut1::from_shape_ptr(ntargets, result_ptr) };

    GftBase::<f32>::evaluate_laplace_kernel(&targets, &sources, &charges, &mut result, parallel);
}
