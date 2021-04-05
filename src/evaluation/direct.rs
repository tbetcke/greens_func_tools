use ndarray;
use ndarray::{ArrayView2, ArrayViewMut2};

pub fn assemble_laplace_kernel<T>(
    targets: &ArrayView2<T>,
    sources: &ArrayView2<T>,
    result: &mut ArrayViewMut2<T>,
) where
    T: DirectAssembler<Item = T>,
{
    T::assemble_laplace_kernel(targets, sources, result);
}

pub trait DirectAssembler {
    type Item;

    fn assemble_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    );
}

impl DirectAssembler for f64 {
    type Item = f64;

    fn assemble_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, mut result_row| {
                laplace_kernel(&target, &sources, &mut result_row)
            })
    }
}

impl DirectAssembler for f32 {
    type Item = f32;

    fn assemble_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result.rows_mut())
            .par_for_each(|target, mut result_row| {
                laplace_kernel(&target, &sources, &mut result_row)
            })
    }
}
