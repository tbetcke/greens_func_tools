use ndarray::{Array1, Array2};
use num_traits;

pub fn laplace_kernel<T>(target: &Array1<T>, sources: &Array2<T>, result: &mut Array1<T>)
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
        .for_each(|target_value, source_row| {
            Zip::from(source_row)
                .and(&mut *result)
                .for_each(|source_value, result_ref| {
                    *result_ref += (*target_value - *source_value) * (*target_value - *source_value)
                })
        });

    result.iter_mut().for_each(|item| *item = m_inv_4pi / item.sqrt() );
    result.iter_mut().filter(|item| item.is_infinite()).for_each(
        |item| *item = zero
    )
}


#[cfg(test)]
 mod tests {

    use ndarray::array;

    #[test]
    fn test_laplace_kernel() {


        let target = array![1.0, 0.0, 0.0];
        let sources = array![[ 1.0],
                             [ 1.0],
                             [ 1.0]];
        
        let mut result = array![0.0];

        crate::kernels::laplace_kernel(&target, &sources, &mut result);
    }
}