use rayon::prelude::*;

pub fn cpu_calc(data:&mut[u32]) {
    data.par_iter_mut().for_each(|x| {
        *x *=2;
    })
}