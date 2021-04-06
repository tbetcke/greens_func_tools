pub mod evaluation;
pub mod kernels;
pub mod cffi;

pub struct GftBase<T: GftSupportedType> {
    phantom: std::marker::PhantomData<T>,
}

pub trait GftSupportedType {
    type Item;
}

impl GftSupportedType for f32 {
    type Item = f32;
}

impl GftSupportedType for f64 {
    type Item = f64;
}

pub use evaluation::direct::*;