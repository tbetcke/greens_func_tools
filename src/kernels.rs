use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};
use num_traits;

pub fn laplace_kernel<T>(
    target: &ArrayView1<T>,
    sources: &ArrayView2<T>,
    result: &mut ArrayViewMut1<T>,
) where
    T: num_traits::Float,
    T: num_traits::FloatConst,
    T: num_traits::NumAssignOps,
{
    use ndarray::Zip;

    let zero: T = num_traits::zero();

    let m_inv_4pi: T =
        num_traits::cast::cast::<f64, T>(0.25).unwrap() * num_traits::FloatConst::FRAC_1_PI();

    result.fill(zero);

    Zip::from(target)
        .and(sources.rows())
        .for_each(|&target_value, source_row| {
            Zip::from(source_row)
                .and(&mut *result)
                .for_each(|&source_value, result_ref| {
                    *result_ref += (target_value - source_value) * (target_value - source_value)
                })
        });

    result.mapv_inplace(|item| m_inv_4pi / item.sqrt());
    result
        .iter_mut()
        .filter(|item| item.is_infinite())
        .for_each(|item| *item = zero);
}

pub fn evaluate_laplace_kernel<T>(
    target: &ArrayView1<T>,
    sources: &ArrayView2<T>,
    charges: &ArrayView1<T>,
) -> T
where
    T: num_traits::Float,
    T: num_traits::FloatConst,
    T: num_traits::NumAssignOps,
    T: std::fmt::Display,
{
    use ndarray::{Array1, Axis, Zip};

    let mut result = num_traits::zero();
    let mut row = Array1::<T>::zeros(sources.len_of(Axis(1)));

    laplace_kernel(target, sources, &mut row.view_mut());
    Zip::from(charges)
        .and(&row)
        .for_each(|row_val, charge| result = (*row_val).mul_add(*charge, result));
    result
}
