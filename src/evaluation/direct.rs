use crate::GftBase;
use ndarray;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub trait DirectEvaluator {
    type Item: num_traits::Float
        + num_traits::FloatConst
        + num_traits::NumAssignOps
        + std::marker::Send
        + std::marker::Sync;

    fn evaluate_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
        parallel: bool,
    ) {
        use ndarray::Zip;

        if parallel {
            Zip::from(targets.columns())
                .and(result)
                .par_for_each(|target, result_elem| {
                    *result_elem = evaluate_laplace_kernel_row(&target, sources, charges)
                });
        } else {
            Zip::from(targets.columns())
                .and(result)
                .for_each(|target, result_elem| {
                    *result_elem = evaluate_laplace_kernel_row(&target, sources, charges)
                });
        }
    }

    fn assemble_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
        parallel: bool,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        if parallel {
            Zip::from(targets.columns())
                .and(result.rows_mut())
                .par_for_each(|target, mut result_row| {
                    laplace_kernel(&target, &sources, &mut result_row)
                })
        } else {
            Zip::from(targets.columns())
                .and(result.rows_mut())
                .for_each(|target, mut result_row| {
                    laplace_kernel(&target, &sources, &mut result_row)
                })
        }
    }
}

impl DirectEvaluator for GftBase<f32> {
    type Item = f32;
}

impl DirectEvaluator for GftBase<f64> {
    type Item = f64;
}

fn evaluate_laplace_kernel_row<T>(
    target: &ArrayView1<T>,
    sources: &ArrayView2<T>,
    charges: &ArrayView1<T>,
) -> T
where
    T: num_traits::Float,
    T: num_traits::FloatConst,
    T: num_traits::NumAssignOps,
{
    use ndarray::{Array1, Axis, Zip};
    use crate::kernels::laplace_kernel;

    let mut result = num_traits::zero();
    let mut row = Array1::<T>::zeros(sources.len_of(Axis(1)));

    laplace_kernel(target, sources, &mut row.view_mut());
    Zip::from(charges)
        .and(&row)
        .for_each(|row_val, charge| result = (*row_val).mul_add(*charge, result));
    result
}
