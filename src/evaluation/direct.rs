pub mod direct {
    use ndarray::Array2;
    use num_traits;
    use std::clone;

    pub fn _laplace_kernel_matrix<T>(_sources: &Array2<T>, _targets: &Array2<T>) -> Array2<T>
    where
        T: clone::Clone,
        T: num_traits::Num,
    {
        Array2::<T>::zeros((2, 2))
    }
}
