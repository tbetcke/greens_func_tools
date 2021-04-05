use ndarray::{ArrayView1, ArrayViewMut1, ArrayView2};
use num_traits;

pub fn laplace_kernel<T>(target: &ArrayView1<T>, sources: &ArrayView2<T>, result: &mut ArrayViewMut1<T>)
where
    T: num_traits::Float,
    T: num_traits::FloatConst,
    T: num_traits::NumAssignOps,
{
    use ndarray::Zip;

    let zero: T = num_traits::zero();

    let m_inv_4pi: T = num_traits::cast::cast::<f64, T>(0.25).unwrap()
        * num_traits::FloatConst::FRAC_1_PI();

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

    result.mapv_inplace(|item| m_inv_4pi / item.sqrt() );
    result.iter_mut().filter(|item| item.is_infinite()).for_each(
        |item| *item = zero
    );


}
