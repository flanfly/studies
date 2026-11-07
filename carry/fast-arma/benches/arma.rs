#![feature(test)]
extern crate test;

#[path = "../tests/common/mod.rs"]
mod common;

use common::{fixture_dir, load_testcases, zip_simd, zip_simd4, zip_simd8};
use fast_arma::arma;
use simba::simd::{SimdValue, f64x2, f64x4, f64x8};
use test::{Bencher, black_box};

const MAX_STEPS: usize = 100;

/// The observations of the eight samples of the `(ar, ma)` parameter set.
fn load_order(ar: usize, ma: usize) -> Vec<Box<[f64]>> {
    let cases: Vec<_> = load_testcases(&fixture_dir())
        .into_iter()
        .filter(|case| case.order() == (ar, ma))
        .map(|case| case.obs())
        .collect();

    assert_eq!(cases.len(), 8, "expected 8 samples for ARMA({ar}, {ma})");
    cases
}

/// Throughput benchmarks: every bench fits all eight samples of one parameter
/// set per iteration, with wider lanes fitting more of them per `arma` call.
macro_rules! arma_benches {
    ($name:ident, $ar:literal, $ma:literal) => {
        #[allow(clippy::zero_repeat_side_effects)]
        mod $name {
            use super::*;

            #[bench]
            fn f64(b: &mut Bencher) {
                let cases = load_order($ar, $ma);
                b.iter(|| {
                    for obs in &cases {
                        let mut phi = [0.0f64; $ar];
                        let mut theta = [0.0f64; $ma];
                        let fit = arma(black_box(&obs[..]), &mut phi, &mut theta, 1e-3, MAX_STEPS)
                            .expect("success");
                        black_box((fit, phi, theta));
                    }
                });
            }

            #[bench]
            fn f64x2(b: &mut Bencher) {
                let cases = load_order($ar, $ma);
                let packed: Vec<_> = cases
                    .chunks_exact(2)
                    .map(|c| zip_simd(&c[0], &c[1]))
                    .collect();

                b.iter(|| {
                    for obs in &packed {
                        let mut phi = [f64x2::splat(0.0); $ar];
                        let mut theta = [f64x2::splat(0.0); $ma];
                        let fit = arma(black_box(&obs[..]), &mut phi, &mut theta, 1e-3, MAX_STEPS)
                            .expect("success");
                        black_box((fit, phi, theta));
                    }
                });
            }

            #[bench]
            fn f64x4(b: &mut Bencher) {
                let cases = load_order($ar, $ma);
                let packed: Vec<_> = cases
                    .chunks_exact(4)
                    .map(|c| zip_simd4(&c[0], &c[1], &c[2], &c[3]))
                    .collect();

                b.iter(|| {
                    for obs in &packed {
                        let mut phi = [f64x4::splat(0.0); $ar];
                        let mut theta = [f64x4::splat(0.0); $ma];
                        let fit = arma(black_box(&obs[..]), &mut phi, &mut theta, 1e-3, MAX_STEPS)
                            .expect("success");
                        black_box((fit, phi, theta));
                    }
                });
            }

            #[bench]
            fn f64x8(b: &mut Bencher) {
                let cases = load_order($ar, $ma);
                let packed: Vec<_> = cases
                    .chunks_exact(8)
                    .map(|c| zip_simd8(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7]))
                    .collect();

                b.iter(|| {
                    for obs in &packed {
                        let mut phi = [f64x8::splat(0.0); $ar];
                        let mut theta = [f64x8::splat(0.0); $ma];
                        let fit = arma(black_box(&obs[..]), &mut phi, &mut theta, 1e-3, MAX_STEPS)
                            .expect("success");
                        black_box((fit, phi, theta));
                    }
                });
            }
        }
    };
}

arma_benches!(arma01, 0, 1);
arma_benches!(arma10, 1, 0);
arma_benches!(arma11, 1, 1);
arma_benches!(arma12, 1, 2);
arma_benches!(arma21, 2, 1);
arma_benches!(arma22, 2, 2);
