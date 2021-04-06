use ndarray;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use crate::GftBase;

pub trait DirectEvaluator{
    type Item;

    fn evaluate_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
        parallel: bool,
    ) {
        if parallel {
            Self::evaluate_laplace_kernel_parallel(targets, sources, charges, result);
        } else {
            Self::evaluate_laplace_kernel_serial(targets, sources, charges, result);
        }
    }

    fn assemble_laplace_kernel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
        parallel: bool,
    ) {
        if parallel {
            Self::assemble_laplace_kernel_parallel(targets, sources, result);
        } else {
            Self::assemble_laplace_kernel_serial(targets, sources, result);
        }
    }

    fn assemble_laplace_kernel_parallel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    );
    fn assemble_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    );

    fn evaluate_laplace_kernel_parallel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    );
    fn evaluate_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    );
}

impl DirectEvaluator for GftBase<f32> {
    type Item = f32;

    fn assemble_laplace_kernel_parallel(
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

    fn assemble_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, mut result_row| laplace_kernel(&target, &sources, &mut result_row))
    }

    fn evaluate_laplace_kernel_parallel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    ) {
        use crate::kernels::evaluate_laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result)
            .par_for_each(|target, result_elem| {
                *result_elem = evaluate_laplace_kernel(&target, sources, charges)
            });
    }

    fn evaluate_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    ) {
        use crate::kernels::evaluate_laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result)
            .for_each(|target, result_elem| {
                *result_elem = evaluate_laplace_kernel(&target, sources, charges)
            });
    }
}

impl DirectEvaluator for GftBase<f64> {
    type Item = f64;

    fn assemble_laplace_kernel_parallel(
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

    fn assemble_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        result: &mut ArrayViewMut2<Self::Item>,
    ) {
        use crate::kernels::laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result.rows_mut())
            .for_each(|target, mut result_row| laplace_kernel(&target, &sources, &mut result_row))
    }

    fn evaluate_laplace_kernel_parallel(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    ) {
        use crate::kernels::evaluate_laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result)
            .par_for_each(|target, result_elem| {
                *result_elem = evaluate_laplace_kernel(&target, sources, charges)
            });
    }

    fn evaluate_laplace_kernel_serial(
        targets: &ArrayView2<Self::Item>,
        sources: &ArrayView2<Self::Item>,
        charges: &ArrayView1<Self::Item>,
        result: &mut ArrayViewMut1<Self::Item>,
    ) {
        use crate::kernels::evaluate_laplace_kernel;
        use ndarray::Zip;

        Zip::from(targets.columns())
            .and(result)
            .for_each(|target, result_elem| {
                *result_elem = evaluate_laplace_kernel(&target, sources, charges)
            });
    }
}
