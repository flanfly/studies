// tests/common/mod.rs
#![allow(dead_code)] // not every user needs every function

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use simba::simd::{f64x2, f64x4, f64x8};

#[derive(Deserialize)]
pub struct Header {
    /// True AR coefficients used to generate the process.
    pub phi: Vec<f64>,
    /// True MA coefficients used to generate the process.
    pub theta: Vec<f64>,
    pub dbic_window: usize,
}

#[derive(Deserialize)]
pub struct Row {
    pub x: f64,
    pub dbic: Option<f64>,
    /// CSS estimate for the window starting at this observation (empty when
    /// the observation has no window fit).
    pub phi: Vec<f64>,
    pub theta: Vec<f64>,
}

#[derive(Deserialize)]
pub struct TestCase {
    pub header: Header,
    pub body: Vec<Row>,
}

impl TestCase {
    /// The `(AR, MA)` orders of this case.
    pub fn order(&self) -> (usize, usize) {
        (self.header.phi.len(), self.header.theta.len())
    }

    pub fn win_sz(&self) -> usize {
        self.header.dbic_window
    }

    pub fn obs(&self) -> Box<[f64]> {
        self.body.iter().map(|r| r.x).collect()
    }

    /// Per-observation delta BIC, with `NaN` where no window fit exists.
    pub fn dbic(&self) -> Box<[f64]> {
        self.body
            .iter()
            .map(|r| r.dbic.unwrap_or(f64::NAN))
            .collect()
    }
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Load every JSON fixture in `dir`, ordered by file name so that lane pairing
/// is deterministic.
pub fn load_testcases(dir: &Path) -> Vec<TestCase> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("dir")
        .filter_map(|e| {
            let p = e.expect("entry").path();
            if !p.is_file() {
                return None;
            }
            if p.extension().map(OsStr::to_string_lossy) != Some("json".into()) {
                return None;
            }

            let text = fs::read_to_string(&p).expect("read fixture");
            let case: TestCase = serde_json::from_str(&text).expect("json");

            assert!(!case.body.is_empty(), "empty body");
            let first = &case.body[0];
            assert_eq!(first.phi.len(), case.header.phi.len());
            assert_eq!(first.theta.len(), case.header.theta.len());

            Some((p, case))
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    entries.into_iter().map(|(_, case)| case).collect()
}

/// Interleave two `f64` slices into a slice of two-lane SIMD values.
pub fn zip_simd(a: &[f64], b: &[f64]) -> Box<[f64x2]> {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| f64x2::new(a, b))
        .collect()
}

/// Interleave four `f64` slices into a slice of four-lane SIMD values.
pub fn zip_simd4(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Box<[f64x4]> {
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .zip(d.iter())
        .map(|(((&a, &b), &c), &d)| f64x4::new(a, b, c, d))
        .collect()
}

/// Interleave eight `f64` slices into a slice of eight-lane SIMD values.
#[allow(clippy::too_many_arguments)]
pub fn zip_simd8(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    e: &[f64],
    f: &[f64],
    g: &[f64],
    h: &[f64],
) -> Box<[f64x8]> {
    let n = a.len();
    assert!(
        [b, c, d, e, f, g, h].iter().all(|s| s.len() == n),
        "zip_simd8: mismatched lengths"
    );

    (0..n)
        .map(|i| f64x8::new(a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i]))
        .collect()
}
