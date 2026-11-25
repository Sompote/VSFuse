//! Small numerical routines that follow numpy / scipy conventions, so the front end
//! reproduces the reference Python pipeline.

use std::f64::consts::PI;

pub const TWO_PI: f64 = 2.0 * PI;

/// `np.interp`: linear interpolation on increasing `xp`, held constant outside.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 0 {
        return f64::NAN;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|v| *v <= x) - 1;
    if x == xp[j] {
        return fp[j];
    }
    let r = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (x - xp[j]) + fp[j];
    if r.is_nan() && x == xp[j + 1] {
        fp[j + 1]
    } else {
        r
    }
}

pub fn interp_all(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    x.iter().map(|v| interp(*v, xp, fp)).collect()
}

pub fn unwrap(w: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(w.len());
    let mut corr = 0.0;
    for i in 0..w.len() {
        if i > 0 {
            let d = w[i] - w[i - 1];
            let mut dm = (d + PI).rem_euclid(TWO_PI) - PI;
            if dm == -PI && d > 0.0 {
                dm = PI;
            }
            if d.abs() >= PI {
                corr += dm - d;
            }
        }
        out.push(w[i] + corr);
    }
    out
}

/// `np.gradient(y, x)` (second order inside, first order at the ends).
pub fn gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    g[0] = (y[1] - y[0]) / (x[1] - x[0]);
    g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    for i in 1..n - 1 {
        let (hs, hd) = (x[i] - x[i - 1], x[i + 1] - x[i]);
        g[i] = (hs * hs * y[i + 1] + (hd * hd - hs * hs) * y[i] - hd * hd * y[i - 1]) / (hs * hd * (hd + hs));
    }
    g
}

pub fn median(v: &[f64]) -> f64 {
    percentile(v, 50.0)
}

/// `np.percentile` with linear interpolation.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let pos = q / 100.0 * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Population standard deviation (`np.std`).
pub fn std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Slope of the least-squares line.
pub fn slope(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    sxy / sxx
}

/// `scipy.signal.medfilt` with a kernel of 5 (zero padding at the ends).
pub fn medfilt5(v: &[f64]) -> Vec<f64> {
    let n = v.len() as isize;
    (0..n)
        .map(|i| {
            let mut w: Vec<f64> = (i - 2..=i + 2).map(|k| if k < 0 || k >= n { 0.0 } else { v[k as usize] }).collect();
            w.sort_by(|a, b| a.total_cmp(b));
            w[2]
        })
        .collect()
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        idx[i..=j].iter().for_each(|k| r[*k] = rank);
        i = j + 1;
    }
    r
}

/// Spearman rank correlation.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let (ma, mb) = (mean(&ra), mean(&rb));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    // rounded so that perfectly monotone records tie exactly, as they do in the reference
    ((sab / (saa * sbb).sqrt()) * 1e10).round() / 1e10
}

/// `scipy.signal.savgol_filter(y, window, degree)` in its default `interp` mode: every sample
/// is the value of a polynomial fitted to the window around it, and the ends use the fit of
/// the first and last full windows.
pub fn savgol(y: &[f64], window: usize, degree: usize) -> Vec<f64> {
    let n = y.len();
    if n < window {
        return y.to_vec();
    }
    let m = degree + 1;
    let xs: Vec<f64> = (0..window).map(|k| k as f64 - (window - 1) as f64 / 2.0).collect();
    // normal equations (A^T A) c = A^T y, inverted once
    let mut ata = vec![vec![0.0; 2 * m]; m];
    for (r, row) in ata.iter_mut().enumerate() {
        for c in 0..m {
            row[c] = xs.iter().map(|x| x.powi((r + c) as i32)).sum();
        }
        row[m + r] = 1.0;
    }
    for c in 0..m {
        let p = (c..m).max_by(|a, b| ata[*a][c].abs().total_cmp(&ata[*b][c].abs())).unwrap();
        ata.swap(c, p);
        let d = ata[c][c];
        ata[c].iter_mut().for_each(|v| *v /= d);
        for r in 0..m {
            if r != c {
                let f = ata[r][c];
                let pivot = ata[c].clone();
                ata[r].iter_mut().zip(&pivot).for_each(|(v, p)| *v -= f * p);
            }
        }
    }
    // weights[pos][k]: contribution of sample k of a window to the fit evaluated at position pos
    let weights: Vec<Vec<f64>> = (0..window)
        .map(|pos| {
            (0..window)
                .map(|k| {
                    let mut s = 0.0;
                    for r in 0..m {
                        for c in 0..m {
                            s += xs[pos].powi(r as i32) * ata[r][m + c] * xs[k].powi(c as i32);
                        }
                    }
                    s
                })
                .collect()
        })
        .collect();
    let half = window / 2;
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half).min(n - window);
            weights[i - start].iter().zip(&y[start..start + window]).map(|(w, v)| w * v).sum()
        })
        .collect()
}

pub fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
}

/// Values of `v` in the order that sorts `key`.
pub fn sorted_by(key: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..key.len()).collect();
    idx.sort_by(|a, b| key[*a].total_cmp(&key[*b]));
    (idx.iter().map(|i| key[*i]).collect(), idx.iter().map(|i| v[*i]).collect())
}
