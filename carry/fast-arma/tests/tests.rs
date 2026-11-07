use std::fmt::Debug;

use fast_arma::{Fit, arma};

use simba::simd::SimdBool;
use simba::simd::f64x2;
use simba::simd::{SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue};

mod common;
use common::{fixture_dir, load_testcases, zip_simd};

// `body[k].phi`/`body[k].theta` are the exact CSS fits `generate.py` computed
// for the window starting at observation `k`, so they can be matched tightly.
const COLD_PARAM_TOLERANCE: f64 = 1e-6;
const WARM_PARAM_TOLERANCE: f64 = 1e-5;
const DBIC_TOLERANCE: f64 = 1e-6;
const COLD_START_ITER: usize = 30;
const WARM_START_ITER: usize = 5;

fn is_close<T>(expected: T, got: T, tol: f64)
where
    T: SimdValue<Element = f64> + Copy,
{
    for lane in 0..T::LANES {
        let expected = expected.extract(lane);
        let got = got.extract(lane);
        let diff = (expected - got).abs();
        assert!(
            diff <= tol,
            "lane {lane}: expected={expected}, got={got}, diff={diff}",
        );
    }
}

fn dyn_arma<const P: usize, const Q: usize, T>(
    obs: &[T],
    max_iter: usize,
) -> (Box<[T]>, Box<[T]>, Fit<T>)
where
    T: SimdRealField + Copy + Debug,
    T::SimdBool: SimdValue<Element = bool> + Debug,
{
    let mut phi = [T::zero(); P];
    let mut theta = [T::zero(); Q];
    let fit = arma(obs, &mut phi, &mut theta, 1e-3, max_iter).expect("success");

    (
        phi.to_vec().into_boxed_slice(),
        theta.to_vec().into_boxed_slice(),
        fit,
    )
}

#[test]
fn f64x1() {
    for fix in load_testcases(&fixture_dir()) {
        let obs = fix.obs();
        let dbic = fix.dbic();
        let win_sz = fix.win_sz();
        // Exact CSS reference for the first window.
        let expected_phi = &fix.body[0].phi;
        let expected_theta = &fix.body[0].theta;

        let (phi, theta, fit) = match fix.order() {
            (1, 0) => dyn_arma::<1, 0, _>(&obs[..win_sz], COLD_START_ITER),
            (0, 1) => dyn_arma::<0, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (1, 1) => dyn_arma::<1, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (1, 2) => dyn_arma::<1, 2, _>(&obs[..win_sz], COLD_START_ITER),
            (2, 1) => dyn_arma::<2, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (2, 2) => dyn_arma::<2, 2, _>(&obs[..win_sz], COLD_START_ITER),
            _ => panic!(),
        };

        assert!(fit.converged.all());

        assert_eq!(expected_phi.len(), phi.len());
        for (expected, got) in expected_phi.iter().zip(phi.iter()) {
            is_close(*expected, *got, COLD_PARAM_TOLERANCE);
        }
        assert_eq!(expected_theta.len(), theta.len());
        for (expected, got) in expected_theta.iter().zip(theta.iter()) {
            is_close(*expected, *got, COLD_PARAM_TOLERANCE);
        }
        assert!((dbic[0] - fit.delta_bic()).abs() < DBIC_TOLERANCE);
    }
}

#[test]
fn simd_f64x2() {
    let mut cases = load_testcases(&fixture_dir());
    cases.sort_by_key(|f| f.order());

    for pair in cases.chunks_exact(2) {
        let (a, b) = (&pair[0], &pair[1]);

        if a.order() != b.order() {
            continue;
        }

        assert_eq!(a.body.len(), b.body.len());
        assert_eq!(a.win_sz(), b.win_sz());

        let obs = zip_simd(&a.obs(), &b.obs());
        // Exact CSS reference for the first window of each case.
        let expected_phi = zip_simd(&a.body[0].phi, &b.body[0].phi);
        let expected_theta = zip_simd(&a.body[0].theta, &b.body[0].theta);
        let expected_dbic = zip_simd(&a.dbic(), &b.dbic());
        let win_sz = a.win_sz();

        let (phi, theta, fit) = match a.order() {
            (1, 0) => dyn_arma::<1, 0, _>(&obs[..win_sz], COLD_START_ITER),
            (0, 1) => dyn_arma::<0, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (1, 1) => dyn_arma::<1, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (1, 2) => dyn_arma::<1, 2, _>(&obs[..win_sz], COLD_START_ITER),
            (2, 1) => dyn_arma::<2, 1, _>(&obs[..win_sz], COLD_START_ITER),
            (2, 2) => dyn_arma::<2, 2, _>(&obs[..win_sz], COLD_START_ITER),
            _ => panic!(),
        };

        assert!(fit.converged.all());

        assert_eq!(expected_phi.len(), phi.len());
        for (expected, got) in expected_phi.into_iter().zip(phi) {
            is_close(expected, got, COLD_PARAM_TOLERANCE);
        }
        assert_eq!(expected_theta.len(), theta.len());
        for (expected, got) in expected_theta.into_iter().zip(theta) {
            is_close(expected, got, COLD_PARAM_TOLERANCE);
        }

        let d = (expected_dbic[0] - fit.delta_bic()).simd_abs();
        println!("{:?}", d);
        assert!(d.simd_lt(f64x2::splat(DBIC_TOLERANCE)).all());
    }
}

#[test]
fn simd_f64x2_warm() {
    let mut cases = load_testcases(&fixture_dir());
    cases.sort_by_key(|f| f.order());

    for pair in cases.chunks_exact(2) {
        let (a, b) = (&pair[0], &pair[1]);

        if a.order() != b.order() || a.order() != (1, 1) {
            continue;
        }

        assert_eq!(a.win_sz(), b.win_sz());
        assert_eq!(a.body.len(), b.body.len());

        let obs = zip_simd(&a.obs(), &b.obs());
        let expected_dbic = zip_simd(&a.dbic(), &b.dbic());
        let win_sz = a.win_sz();

        let mut phi = [f64x2::splat(0.0); 1];
        let mut theta = [f64x2::splat(0.0); 1];

        // cold: exact reference for the first window
        let fit = arma::<_, 1, 1>(&obs[..win_sz], &mut phi, &mut theta, 1e-3, COLD_START_ITER)
            .expect("success");
        assert!(fit.converged.all());

        let expected_phi = zip_simd(&a.body[0].phi, &b.body[0].phi);
        let expected_theta = zip_simd(&a.body[0].theta, &b.body[0].theta);
        assert_eq!(expected_phi.len(), phi.len());
        for (expected, got) in expected_phi.into_iter().zip(phi) {
            is_close(expected, got, COLD_PARAM_TOLERANCE);
        }
        assert_eq!(expected_theta.len(), theta.len());
        for (expected, got) in expected_theta.into_iter().zip(theta) {
            is_close(expected, got, COLD_PARAM_TOLERANCE);
        }

        // warm: exact reference for each subsequent window
        for i in 1..=10 {
            let win = &obs[i..(i + win_sz)];

            let fit =
                arma::<_, 1, 1>(win, &mut phi, &mut theta, 1e-3, WARM_START_ITER).expect("success");

            let d = (expected_dbic[i] - fit.delta_bic()).simd_abs();
            assert!(d.simd_lt(f64x2::splat(DBIC_TOLERANCE)).all());

            let expected_phi = zip_simd(&a.body[i].phi, &b.body[i].phi);
            let expected_theta = zip_simd(&a.body[i].theta, &b.body[i].theta);
            assert_eq!(expected_phi.len(), phi.len());
            for (expected, got) in expected_phi.into_iter().zip(phi) {
                is_close(expected, got, WARM_PARAM_TOLERANCE);
            }
            assert_eq!(expected_theta.len(), theta.len());
            for (expected, got) in expected_theta.into_iter().zip(theta) {
                is_close(expected, got, WARM_PARAM_TOLERANCE);
            }
        }
    }
}
