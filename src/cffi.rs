
#[no_mangle]
pub unsafe extern "C" fn rust_function(ptr: *mut f64, rows : usize, cols : usize) {

    let arr = ndarray::ArrayViewMut2::from_shape_ptr((rows, cols), ptr);
    println!("Value: {}", arr[[0, 0]]);

}