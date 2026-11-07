use std::cmp::min;
use std::fmt::Debug;

use simba::simd::SimdBool;
use simba::simd::{SimdRealField, SimdValue};

use anyhow::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fit<T>
where
    T: SimdRealField + Copy,
{
    pub sse: T,
    pub sse_null: T,
    pub num_residuals: usize,
    pub num_parameters: usize,
    pub converged: T::SimdBool,
}

impl<T> Fit<T>
where
    T: SimdRealField + Copy,
{
    pub fn delta_bic(&self) -> T {
        let m = self.num_residuals as f64;
        T::from_subset(&m) * (self.sse_null / self.sse).simd_ln()
            - T::from_subset(&(self.num_parameters as f64 * m.ln()))
    }
}

#[allow(non_snake_case)]
pub fn residuals<T, const P: usize, const Q: usize>(
    x: &[T],
    φ: &[T; P],
    θ: &[T; Q],
    ε: &mut [T],
) -> T
where
    T: SimdRealField + Copy + Debug,
    T::SimdBool: SimdValue<Element = bool> + Debug,
{
    let mut S = T::zero();

    for i in P..x.len() {
        let mut err = T::zero();

        for j in 0..P {
            err += φ[j] * x[i - 1 - j];
        }
        for k in 0..min(Q, i - P) {
            err += θ[k] * ε[i - 1 - k];
        }

        err = x[i] - err;
        ε[i] = err;
        S += err * err;
    }

    S
}

#[allow(non_snake_case)]
pub fn jacobian<T, const P: usize, const Q: usize>(
    x: &[T],
    _φ: &[T; P],
    θ: &[T; Q],
    ε: &[T],
    flag: &T::SimdBool,
    J: &mut [T],
) where
    T: SimdRealField + Copy + Debug,
    T::SimdBool: SimdValue<Element = bool> + Debug,
{
    let K = P + Q;

    for t in P..x.len() {
        let m = min(Q, t - P);
        let row = t * K;

        for c in 0..K {
            J[row + c] = flag.if_else(|| T::zero(), || J[row + c]);
        }

        // e'_t(phi_i) = -X_{t-i}
        for i in 0..P {
            J[row + i] = flag.if_else(|| -x[t - 1 - i], || J[row + i]);
        }
        // e'_t(theta_i) = -e_{t-i}
        for i in 0..m {
            J[row + P + i] = flag.if_else(|| -ε[t - 1 - i], || J[row + P + i]);
        }

        // -\sum_{k=1}^{m_t} theta_k * r_{t-k}
        for c in 0..K {
            let mut s = T::zero();
            for k in 0..m {
                s += θ[k] * J[(t - 1 - k) * K + c];
            }
            J[row + c] -= flag.if_else(|| s, || T::zero());
        }
    }
}

#[allow(non_snake_case)]
pub fn cholesky_decomposition<T>(A: &[T], g: &[T], L: &mut [T], ω: &mut [T], K: usize)
where
    T: SimdRealField + Copy + Debug,
{
    for j in 0..K {
        // diag L_jj
        let mut diag = A[j * K + j];
        for k in 0..j {
            diag -= L[j * K + k] * L[j * K + k];
        }
        diag = diag.simd_sqrt();
        L[j * K + j] = diag;

        // lower triag L_ij for i > j
        for i in (j + 1)..K {
            let mut triag = A[i * K + j];
            for k in 0..j {
                triag -= L[i * K + k] * L[j * K + k];
            }

            L[i * K + j] = triag / diag;
        }
    }

    // forward pass
    for i in 0..K {
        let mut s = T::zero();
        for j in 0..i {
            s += L[i * K + j] * ω[j];
        }
        ω[i] = (-g[i] - s) / L[i * K + i];
    }

    // backward pass
    for i in (0..K).rev() {
        let mut s = T::zero();
        for j in (i + 1)..K {
            s += L[j * K + i] * ω[j];
        }
        ω[i] = (ω[i] - s) / L[i * K + i];
    }
}

#[allow(non_snake_case)]
pub fn solve<T, const P: usize, const Q: usize>(
    y: &[T], J: &[T], ε: &[T], g: &mut [T], A: &mut [T]
) where
    T: SimdRealField + Copy + Debug,
{
    let K = P + Q;
    for t in P..y.len() {
        let row = t * K;
        for i in 0..K {
            g[i] += J[row + i] * ε[t];
            // lower triangle
            for j in 0..=i {
                A[i * K + j] += J[row + i] * J[row + j];
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn arma<T, const P: usize, const Q: usize>(
    x: &[T],
    φ: &mut [T; P],
    θ: &mut [T; Q],
    lambda: f64,
    max_steps: usize,
) -> Result<Fit<T>>
where
    T: SimdRealField + Copy + Debug,
    T::SimdBool: SimdValue<Element = bool> + Debug,
{
    const { assert!(P + Q > 0, "ARMA needs p + q > 0") };
    assert!(x.len() > P, "ARMA needs |x| > p");

    let K = P + Q;
    let gtol = T::from_subset(&1e-8);
    let noise = T::from_subset(&1e-30);
    let (lambda_downstep, lambda_upstep) = (T::from_subset(&0.3), T::from_subset(&10.0));
    let (lambda_min, lambda_max) = (T::from_subset(&1e-12), T::from_subset(&1e12));

    let mut λ = T::from_subset(&lambda);

    // remove mean
    let n = T::from_subset(&(x.len() as f64));
    let μ = x.iter().fold(T::zero(), |acc, &v| acc + v) / n;
    let y: Box<[T]> = x.iter().map(|&v| v - μ).collect();

    let sse_null = y[P..].iter().fold(T::zero(), |acc, &v| acc + v * v);

    let mut ε = vec![T::zero(); y.len()].into_boxed_slice();
    let mut ε_prime = vec![T::zero(); y.len()].into_boxed_slice();
    let mut J = vec![T::zero(); K * y.len()].into_boxed_slice();
    let mut L = vec![T::zero(); K * K].into_boxed_slice();
    let mut ω = vec![T::zero(); K].into_boxed_slice();
    let mut φ_prime = [T::zero(); P];
    let mut θ_prime = [T::zero(); Q];
    let mut d = vec![T::zero(); K].into_boxed_slice();

    // A = sum of outer products of r_t
    // g = J^T * e
    let mut A = vec![T::zero(); K * K].into_boxed_slice();
    let mut g = vec![T::zero(); K].into_boxed_slice();

    let mut has_converged = T::SimdBool::splat(false);
    let mut compute_jacobian = T::SimdBool::splat(true);
    let mut S_best = T::zero();

    for _ in 0..max_steps {
        // errors
        let S = residuals::<T, P, Q>(&y, φ, θ, &mut ε);
        S_best = S;

        // jacobian
        if compute_jacobian.any() {
            jacobian(&y, φ, θ, &ε, &compute_jacobian, &mut J);
        }

        // A = sum of outer products of r_t
        // g = J^T * e
        A.fill(T::zero());
        g.fill(T::zero());
        solve::<T, P, Q>(&y, &J, &ε, &mut g, &mut A);

        // scale
        for i in 0..K {
            d[i] = (A[i * K + i] + noise).simd_sqrt().simd_recip();
            g[i] = g[i] * d[i];

            for j in 0..=i {
                A[i * K + j] = d[i] * A[i * K + j] * d[j];
            }
        }

        // check if the gradient is too small to make improvements
        let mut small_gradient = T::SimdBool::splat(true);
        let rs = S.simd_sqrt();
        for i in 0..K {
            small_gradient = small_gradient & g[i].simd_abs().simd_le(gtol * rs);
        }

        // dampen
        for i in 0..K {
            A[i * K + i] += λ;
        }

        cholesky_decomposition(&A, &g, &mut L, &mut ω, K);

        // unscale
        for i in 0..K {
            ω[i] *= d[i];
        }

        // trail
        for i in 0..P {
            φ_prime[i] = φ[i] + ω[i];
        }
        for i in 0..Q {
            θ_prime[i] = θ[i] + ω[i + P];
        }

        // errors
        let S_prime = residuals::<T, P, Q>(&y, &φ_prime, &θ_prime, &mut ε_prime);

        let accepted = S_prime.simd_lt(S) & !has_converged;
        S_best = accepted.if_else(|| S_prime, || S_best);
        for i in 0..P {
            φ[i] = accepted.if_else(|| φ_prime[i], || φ[i]);
        }
        for i in 0..Q {
            θ[i] = accepted.if_else(|| θ_prime[i], || θ[i]);
        }
        for i in 0..ε.len() {
            ε[i] = accepted.if_else(|| ε_prime[i], || ε[i]);
        }

        compute_jacobian = accepted & !has_converged;

        let λ_prime = accepted
            .if_else(|| λ * lambda_downstep, || λ * lambda_upstep)
            .simd_clamp(lambda_min, lambda_max);
        λ = has_converged.if_else(|| λ, || λ_prime);

        let min_error = ((S - S_prime) / S).simd_lt(T::from_subset(&1e-10)) & accepted;
        let mut no_improv = T::SimdBool::splat(true);
        for i in 0..K {
            no_improv = no_improv & ω[i].simd_abs().simd_lt(T::from_subset(&1e-8));
        }

        has_converged = has_converged | (min_error & no_improv) | small_gradient;
        if has_converged.all() {
            break;
        }
    }

    let fit = Fit {
        sse: S_best,
        sse_null,
        num_residuals: x.len() - P,
        num_parameters: P + Q,
        converged: has_converged,
    };
    Ok(fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_zero() {
        let x = [0.0f64; 1000];
        let mut φ = [2.0f64; 2];
        let mut θ = [2.0f64; 2];

        let fit = arma(&x, &mut φ, &mut θ, 1e-3, 20).expect("success");
        println!("{:?}", fit);
        println!("{:?} {:?}", φ, θ);
        assert!(fit.converged.all());
        let fit = arma(&x, &mut φ, &mut θ, 1e-6, 3).expect("success");
        assert!(fit.converged.all());
        println!("{:?}", fit);
    }
}
