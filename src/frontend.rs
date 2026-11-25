//! From the raw SASW records of one foundation to its composite dispersion curve.
//! A port of the GateSASW pipeline (run recovery, cross-spacing phase unwrapping, consensus
//! acceptance, composite curve), kept numerically faithful to the Python reference.
//!
//! Layout of a site folder:  `<record>/Run*/ASCII/Hxysv00000.txt` (transfer function) and
//! `Cxysv00000.txt` (coherence); the record name holds spacing and direction, e.g. `H4mR_HM.trf`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::num::*;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub coh_short: f64,
    pub coh_long: f64,
    pub coh_break: f64,
    pub lam_min_ratio: f64,
    pub lam_max_ratio: f64,
    pub vph_min: f64,
    pub vph_max: f64,
    pub min_pts: usize,
    pub spearman_min: f64,
    pub fr_agree: f64,
    pub cross_dev: f64,
    pub n_bins: usize,
    pub sigma_floor: f64,
    pub trust: [f64; 3],
    pub anchor_lo: f64,
    pub ambiguous: f64,
    pub pair_max: f64,
    pub res_div: f64,
    pub coh_share_min: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { coh_short: 0.95, coh_long: 0.90, coh_break: 4.0, lam_min_ratio: 0.5, lam_max_ratio: 3.0, vph_min: 50.0, vph_max: 1500.0, min_pts: 10, spearman_min: 0.5, fr_agree: 0.20, cross_dev: 0.25, n_bins: 28, sigma_floor: 0.08, trust: [1.0, 1.5, 2.0], anchor_lo: 0.25, ambiguous: 0.35, pair_max: 0.5, res_div: 2.5, coh_share_min: 0.30 }
    }
}

#[derive(Clone, Debug)]
pub struct Test {
    pub label: String,
    pub run: PathBuf,
    pub spacing: f64,
    pub forward: bool,
}

/// Accepted dispersion points of one spacing.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SpacingCurve {
    pub spacing: f64,
    pub trust: f64,
    pub wavelength: Vec<f64>,
    pub velocity: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct Processed {
    pub wavelength: Vec<f64>,
    pub velocity: Vec<f64>,
    /// Absolute uncertainty of the composite curve (m/s).
    pub sigma: Vec<f64>,
    pub accepted: Vec<SpacingCurve>,
    pub report: Vec<String>,
    pub records: usize,
    pub coherent_share: f64,
    pub z_res: f64,
}

/// Spacing and direction from a record name: `H<spacing>m<F|R>`, anywhere in the name.
pub fn parse_record(name: &str) -> Option<(f64, bool)> {
    let b = name.as_bytes();
    for i in 0..b.len() {
        if !b[i].eq_ignore_ascii_case(&b'h') {
            continue;
        }
        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == i + 1 {
            continue;
        }
        if j + 1 < b.len() && b[j] == b'.' && b[j + 1].is_ascii_digit() {
            j += 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
        }
        if j + 1 < b.len() && b[j].eq_ignore_ascii_case(&b'm') && (b[j + 1].eq_ignore_ascii_case(&b'f') || b[j + 1].eq_ignore_ascii_case(&b'r')) {
            return Some((name[i + 1..j].parse().ok()?, b[j + 1].eq_ignore_ascii_case(&b'f')));
        }
    }
    None
}

fn sorted_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    v.sort();
    v
}

/// Every run of every record that holds an exported spectrum, short spacings first.
pub fn detect(site: &Path) -> Vec<Test> {
    let mut tests = Vec::new();
    for rec in sorted_dirs(site) {
        let name = rec.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let Some((spacing, forward)) = parse_record(&name) else { continue };
        for run in sorted_dirs(&rec) {
            let rn = run.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            if rn.starts_with("Run") && run.join("ASCII/Hxysv00000.txt").exists() && run.join("ASCII/Cxysv00000.txt").exists() {
                tests.push(Test { label: format!("{name}/{rn}"), run, spacing, forward });
            }
        }
    }
    // stable: spacing, then forward before reverse
    tests.sort_by(|a, b| a.spacing.total_cmp(&b.spacing).then(b.forward.cmp(&a.forward)));
    tests
}

/// True when the folder itself holds SASW records.
pub fn is_site(dir: &Path) -> bool {
    !detect(dir).is_empty()
}

fn read_file(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let text = String::from_utf8_lossy(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?).to_string();
    let (mut started, mut ncol, mut cols) = (false, None, vec![Vec::new(), Vec::new(), Vec::new()]);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if started {
            let vals: Option<Vec<f64>> = line.split_whitespace().take(3).map(|s| s.parse::<f64>().ok()).collect();
            if let Some(vals) = vals.filter(|v| v.len() >= 2) {
                let n = *ncol.get_or_insert(vals.len().min(3));
                if vals.len() >= n {
                    cols[0].push(vals[0]);
                    cols[1].push(vals[1]);
                    if n > 2 {
                        cols[2].push(vals[2]);
                    }
                }
            }
        }
        if line.starts_with("Hz") {
            started = true;
        }
    }
    Ok(cols)
}

struct Record {
    freq: Vec<f64>,
    real: Vec<f64>,
    imag: Vec<f64>,
    coh: Vec<f64>,
}

fn read_record(run: &Path) -> Result<Record, String> {
    let mut h = read_file(&run.join("ASCII/Hxysv00000.txt"))?;
    let c = read_file(&run.join("ASCII/Cxysv00000.txt"))?;
    if h[0].len() < 8 || h[2].len() != h[0].len() || c[0].len() < 2 {
        return Err("no spectrum in the export".into());
    }
    let coh = interp_all(&h[0], &c[0], &c[1]);
    Ok(Record { freq: std::mem::take(&mut h[0]), real: std::mem::take(&mut h[1]), imag: std::mem::take(&mut h[2]), coh })
}

fn threshold(spacing: f64, p: &Params) -> f64 {
    if spacing <= p.coh_break {
        p.coh_short
    } else {
        p.coh_long
    }
}

/// Index ranges of consecutive `true` samples of length `min_len` or more.
fn segments(good: &[bool], min_len: usize) -> Vec<(usize, usize)> {
    let (mut segs, mut start) = (Vec::new(), None);
    for i in 0..=good.len() {
        let g = i < good.len() && good[i];
        match (g, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if i - s >= min_len {
                    segs.push((s, i - 1));
                }
                start = None;
            }
            _ => {}
        }
    }
    segs
}

/// Masked dispersion points (wavelength, velocity, frequency) of an unwrapped phase.
fn to_dispersion(freq: &[f64], phase: &[f64], coh: &[f64], spacing: f64, p: &Params) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = gradient(phase, freq);
    let pos: Vec<f64> = d.iter().copied().filter(|v| *v > 0.0).collect();
    let (mut wl, mut v, mut f) = (Vec::new(), Vec::new(), Vec::new());
    if pos.is_empty() {
        return (wl, v, f);
    }
    let floor = 0.2 * median(&pos);
    let thr = threshold(spacing, p);
    for i in 0..freq.len() {
        if freq[i] > 0.0 && coh[i] >= thr && phase[i] > 0.1 && d[i] > floor {
            let w = TWO_PI * spacing / phase[i];
            let vp = freq[i] * w;
            if w >= p.lam_min_ratio * spacing && w <= p.lam_max_ratio * spacing && vp > p.vph_min && vp < p.vph_max {
                wl.push(w);
                v.push(vp);
                f.push(freq[i]);
            }
        }
    }
    (wl, v, f)
}

fn quality(wl: &[f64], v: &[f64], f: &[f64], p: &Params) -> (usize, f64) {
    if wl.len() < p.min_pts {
        return (wl.len(), f64::NAN);
    }
    let s: Vec<f64> = f.iter().zip(v).map(|(f, v)| f / v).collect();
    (wl.len(), spearman(f, &s))
}

fn common(wa: &[f64], va: &[f64], wb: &[f64], vb: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = (min(wa).max(min(wb)), max(wa).min(max(wb)));
    if lo >= hi {
        return None;
    }
    let wt = linspace(lo, hi, 20);
    let (xa, ya) = sorted_by(wa, va);
    let (xb, yb) = sorted_by(wb, vb);
    Some((interp_all(&wt, &xa, &ya), interp_all(&wt, &xb, &yb)))
}

/// Mean relative difference of two curves over their common wavelengths.
fn fr_disagreement(wa: &[f64], va: &[f64], wb: &[f64], vb: &[f64]) -> f64 {
    match common(wa, va, wb, vb) {
        Some((a, b)) => mean(&a.iter().zip(&b).map(|(a, b)| (a - b).abs() / ((a + b) / 2.0)).collect::<Vec<_>>()),
        None => f64::INFINITY,
    }
}

fn overlap_dev(wa: &[f64], va: &[f64], wb: &[f64], vb: &[f64]) -> f64 {
    match common(wa, va, wb, vb) {
        Some((a, b)) => median(&a.iter().zip(&b).map(|(a, b)| (a - b).abs() / b).collect::<Vec<_>>()),
        None => f64::NAN,
    }
}

/// Velocity against frequency from the spacings already resolved.
#[derive(Default)]
struct Reference {
    parts: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Reference {
    fn add(&mut self, spacing: f64, f: &[f64], phase: &[f64]) {
        let ok: Vec<usize> = (0..f.len()).filter(|&i| phase[i].is_finite() && phase[i] > 0.3 && f[i] > 0.0).collect();
        if ok.len() < 6 {
            return;
        }
        let v: Vec<f64> = ok.iter().map(|&i| TWO_PI * f[i] * spacing / phase[i]).collect();
        self.parts.push((ok.iter().map(|&i| f[i]).collect(), medfilt5(&v)));
    }

    fn velocity(&self, f: &[f64]) -> Vec<f64> {
        let mut out = vec![f64::NAN; f.len()];
        for (fp, vp) in &self.parts {
            let (lo, hi) = (fp[0], fp[fp.len() - 1]);
            for (o, x) in out.iter_mut().zip(f) {
                if *x >= lo && *x <= hi {
                    *o = interp(*x, fp, vp);
                }
            }
        }
        if out.iter().any(|v| v.is_nan()) {
            let fa: Vec<f64> = self.parts.iter().flat_map(|p| p.0.iter().copied()).collect();
            let va: Vec<f64> = self.parts.iter().flat_map(|p| p.1.iter().copied()).collect();
            let (fa, va) = sorted_by(&fa, &va);
            for (o, x) in out.iter_mut().zip(f) {
                if o.is_nan() {
                    *o = interp(*x, &fa, &va);
                }
            }
        }
        out
    }
}

/// Integer cycles of every coherent segment. Returns the phase (NaN outside the segments)
/// and the worst residual against the prediction, in cycles.
fn resolve(freq: &[f64], w: &[f64], segs: &[(usize, usize)], spacing: f64, reference: &Reference, p: &Params) -> (Vec<f64>, f64) {
    let mut phase = vec![f64::NAN; freq.len()];
    let (mut worst, mut prev): (f64, Option<(usize, usize)>) = (0.0, None);
    let vref = (!reference.parts.is_empty()).then(|| reference.velocity(freq));
    for &(i0, i1) in segs {
        let seg = unwrap(&w[i0..=i1]);
        let n = if let Some(vref) = &vref {
            let pred: Vec<f64> = (i0..=i1).map(|i| TWO_PI * freq[i] * spacing / vref[i]).collect();
            let n = (median(&pred.iter().zip(&seg).map(|(a, b)| a - b).collect::<Vec<_>>()) / TWO_PI).round();
            let res = median(&seg.iter().zip(&pred).map(|(s, q)| (s + TWO_PI * n - q).abs()).collect::<Vec<_>>()) / TWO_PI;
            worst = worst.max(res);
            n
        } else if let Some((j0, j1)) = prev {
            let k = j0.max(j1.saturating_sub(8));
            let s = if j1 - k >= 2 { slope(&freq[k..=j1], &phase[k..=j1]).max(0.0) } else { 0.0 };
            ((phase[j1] + s * (freq[i0] - freq[j1]) - seg[0]) / TWO_PI).round()
        } else {
            // anchor of the shortest spacing: the first coherent phase lies in [-lo, 1 - lo) cycle
            -(median(&seg[..seg.len().min(3)]) / TWO_PI + p.anchor_lo).floor()
        };
        for (i, s) in (i0..=i1).zip(&seg) {
            phase[i] = s + TWO_PI * n;
        }
        prev = Some((i0, i1));
    }
    (phase, worst)
}

fn filled(freq: &[f64], phase: &[f64]) -> Vec<f64> {
    let (fx, fy): (Vec<f64>, Vec<f64>) = freq.iter().zip(phase).filter(|(_, p)| p.is_finite()).map(|(f, p)| (*f, *p)).unzip();
    let out = interp_all(freq, &fx, &fy);
    if out.len() > 15 {
        savgol(&out, 11, 3)
    } else {
        out
    }
}

struct Resolved {
    freq: Vec<f64>,
    phase: Vec<f64>,
    coh: Vec<f64>,
}

/// Cross-spacing phase unwrapping of every record of a site, short spacing first.
fn chain(tests: &[Test], p: &Params, report: &mut Vec<String>) -> BTreeMap<usize, Resolved> {
    struct Raw {
        test: usize,
        forward: bool,
        freq: Vec<f64>,
        w: Vec<f64>,
        coh: Vec<f64>,
        segs: Vec<(usize, usize)>,
    }
    let mut raw: Vec<(f64, Vec<Raw>)> = Vec::new();
    for (k, t) in tests.iter().enumerate() {
        let r = match read_record(&t.run) {
            Ok(r) => r,
            Err(e) => {
                report.push(format!("{}: read error {e}", t.label));
                continue;
            }
        };
        let thr = threshold(t.spacing, p);
        let mut w: Vec<f64> = r.imag.iter().zip(&r.real).map(|(i, r)| i.atan2(*r)).collect();
        let good: Vec<bool> = r.coh.iter().zip(&r.freq).map(|(c, f)| *c >= thr && *f > 0.0).collect();
        let segs = segments(&good, 4);
        if segs.is_empty() {
            continue;
        }
        // sign that makes the phase grow with frequency, judged on the longest coherent segment
        let mut longest = segs[0];
        for s in &segs {
            if s.1 - s.0 > longest.1 - longest.0 {
                longest = *s;
            }
        }
        if slope(&r.freq[longest.0..=longest.1], &unwrap(&w[longest.0..=longest.1])) < 0.0 {
            w.iter_mut().for_each(|v| *v = -*v);
        }
        let item = Raw { test: k, forward: t.forward, freq: r.freq, w, coh: r.coh, segs };
        match raw.iter_mut().find(|g| g.0 == t.spacing) {
            Some(g) => g.1.push(item),
            None => raw.push((t.spacing, vec![item])),
        }
    }
    raw.sort_by(|a, b| a.0.total_cmp(&b.0));
    struct Cand {
        forward: bool,
        freq: Vec<f64>,
        ph: Vec<f64>,
        wl: Vec<f64>,
        v: Vec<f64>,
        worst: f64,
    }
    let (mut reference, mut out) = (Reference::default(), BTreeMap::new());
    for (spacing, group) in raw {
        let mut cands = Vec::new();
        for g in group {
            let (ph, worst) = resolve(&g.freq, &g.w, &g.segs, spacing, &reference, p);
            let full = filled(&g.freq, &ph);
            let (wl, v, f) = to_dispersion(&g.freq, &full, &g.coh, spacing, p);
            let (n, rho) = quality(&wl, &v, &f, p);
            if worst > p.ambiguous {
                report.push(format!("{}: cycle count ambiguous (residual {worst:.2} cycle)", tests[g.test].label));
            }
            if n >= p.min_pts && rho >= p.spearman_min {
                cands.push(Cand { forward: g.forward, freq: g.freq.clone(), ph, wl, v, worst });
            }
            out.insert(g.test, Resolved { freq: g.freq, phase: full, coh: g.coh });
        }
        if cands.is_empty() {
            continue;
        }
        // extend the predictor: the best-agreeing forward/reverse pair, else the best single record
        let mut best: Option<(f64, usize, usize)> = None;
        for (i, a) in cands.iter().enumerate().filter(|c| c.1.forward) {
            for (j, b) in cands.iter().enumerate().filter(|c| !c.1.forward) {
                let dis = fr_disagreement(&a.wl, &a.v, &b.wl, &b.v);
                if best.is_none_or(|x| dis < x.0) {
                    best = Some((dis, i, j));
                }
            }
        }
        match best {
            Some((dis, i, j)) if dis <= p.pair_max => {
                let (a, b) = (&cands[i], &cands[j]);
                let pb = interp_all(&a.freq, &b.freq, &b.ph);
                let ph: Vec<f64> = a.ph.iter().zip(&pb).map(|(x, y)| if x.is_finite() && y.is_finite() { (x + y) / 2.0 } else { f64::NAN }).collect();
                reference.add(spacing, &a.freq, &ph);
            }
            _ => {
                let mut c = &cands[0];
                for x in &cands {
                    if x.worst < c.worst {
                        c = x;
                    }
                }
                reference.add(spacing, &c.freq, &c.ph);
            }
        }
    }
    out
}

/// Composite curve: median in logarithmic wavelength bins, IQR rejection, trust-weighted scatter.
fn composite(wl: &[f64], v: &[f64], trust: &[f64], p: &Params) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let nb = p.n_bins.min(wl.len() / 4).max(6);
    let (lo, hi) = (wl.iter().cloned().fold(f64::INFINITY, f64::min) * 0.99, wl.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.01);
    let edges: Vec<f64> = (0..=nb).map(|j| 10f64.powf(lo.log10() + (hi.log10() - lo.log10()) * j as f64 / nb as f64)).collect();
    let (mut wc, mut vc, mut sg) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..nb {
        let idx: Vec<usize> = (0..wl.len()).filter(|&i| wl[i] >= edges[j] && wl[i] < edges[j + 1]).collect();
        if idx.len() < 2 {
            continue;
        }
        let vals: Vec<f64> = idx.iter().map(|&i| v[i]).collect();
        let (q1, q3) = (percentile(&vals, 25.0), percentile(&vals, 75.0));
        let iqr = q3 - q1;
        let inl: Vec<f64> = if iqr > 0.0 { vals.iter().copied().filter(|x| *x >= q1 - 1.5 * iqr && *x <= q3 + 1.5 * iqr).collect() } else { vals.clone() };
        let med = median(&inl);
        wc.push((edges[j] * edges[j + 1]).sqrt());
        vc.push(med);
        sg.push(std(&inl).max(p.sigma_floor * med) * mean(&idx.iter().map(|&i| trust[i]).collect::<Vec<_>>()));
    }
    if wc.len() < 4 {
        return Err(format!("composite curve too sparse ({} bins)", wc.len()));
    }
    if vc.len() > 7 {
        vc = savgol(&vc, 7, 2);
    }
    Ok((wc, vc, sg))
}

/// Raw records of one site folder to its composite dispersion curve.
pub fn process(site: &Path, p: &Params) -> Result<Processed, String> {
    let tests = detect(site);
    if tests.is_empty() {
        return Err(format!("{}: no SASW records (folders named like H4mF_HM.trf with Run*/ASCII exports)", site.display()));
    }
    let mut report = Vec::new();
    let resolved = chain(&tests, p, &mut report);
    // per-record gate, grouped by spacing and direction
    struct Rec<'a> {
        r: &'a Resolved,
        label: &'a str,
        wl: Vec<f64>,
        v: Vec<f64>,
        rho: f64,
    }
    let mut groups: BTreeMap<u64, (f64, Vec<Rec>, Vec<Rec>)> = BTreeMap::new();
    for (k, t) in tests.iter().enumerate() {
        let Some(r) = resolved.get(&k) else {
            report.push(format!("{}: rejected, no coherent segment", t.label));
            continue;
        };
        let (wl, v, f) = to_dispersion(&r.freq, &r.phase, &r.coh, t.spacing, p);
        let (n, rho) = quality(&wl, &v, &f, p);
        if n >= p.min_pts && rho >= p.spearman_min {
            let g = groups.entry((t.spacing * 1000.0).round() as u64).or_insert((t.spacing, Vec::new(), Vec::new()));
            let rec = Rec { r, label: &t.label, wl, v, rho };
            if t.forward {
                g.1.push(rec)
            } else {
                g.2.push(rec)
            }
        } else {
            report.push(format!("{}: rejected by the record gate (n={n}, rho={rho:+.2})", t.label));
        }
    }
    // one candidate curve per spacing
    struct Cand {
        spacing: f64,
        wl: Vec<f64>,
        v: Vec<f64>,
        trust: f64,
        dis: f64,
        label: String,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for (spacing, fwd, rev) in groups.values() {
        let mut best: Option<(f64, &Rec, &Rec)> = None;
        for a in fwd {
            for b in rev {
                let dis = fr_disagreement(&a.wl, &a.v, &b.wl, &b.v);
                if best.as_ref().is_none_or(|x| dis < x.0) {
                    best = Some((dis, a, b));
                }
            }
        }
        if let Some((dis, a, b)) = best {
            // forward/reverse phase average on a common frequency grid
            let (f_min, f_max) = (a.r.freq[0].max(b.r.freq[0]), a.r.freq[a.r.freq.len() - 1].min(b.r.freq[b.r.freq.len() - 1]));
            let df = (a.r.freq[1] - a.r.freq[0]).max(b.r.freq[1] - b.r.freq[0]);
            let n = (((f_max + df / 2.0) - f_min) / df).ceil().max(0.0) as usize;
            let fg: Vec<f64> = (0..n).map(|i| f_min + i as f64 * df).collect();
            let ph: Vec<f64> = fg.iter().map(|x| (interp(*x, &a.r.freq, &a.r.phase) + interp(*x, &b.r.freq, &b.r.phase)) / 2.0).collect();
            let ch: Vec<f64> = fg.iter().map(|x| interp(*x, &a.r.freq, &a.r.coh).min(interp(*x, &b.r.freq, &b.r.coh))).collect();
            let (wl, v, _) = to_dispersion(&fg, &ph, &ch, *spacing, p);
            if wl.len() >= p.min_pts {
                let trust = if dis <= p.fr_agree { p.trust[0] } else { p.trust[1] };
                cands.push(Cand { spacing: *spacing, wl, v, trust, dis, label: format!("forward/reverse average {} + {} (differ {:.0}%)", a.label, b.label, dis * 100.0) });
                continue;
            }
        }
        let mut pick: Option<&Rec> = None;
        for r in fwd.iter().chain(rev) {
            if pick.is_none_or(|q| (r.rho, r.wl.len()) > (q.rho, q.wl.len())) {
                pick = Some(r);
            }
        }
        if let Some(r) = pick {
            cands.push(Cand { spacing: *spacing, wl: r.wl.clone(), v: r.v.clone(), trust: p.trust[2], dis: f64::NAN, label: format!("{} single direction", r.label) });
        }
    }
    if cands.is_empty() {
        return Err("all records were rejected by the quality gates".into());
    }
    // consensus: spacings whose curves agree over their common wavelengths form a group
    let n = cands.len();
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let dev = overlap_dev(&cands[i].wl, &cands[i].v, &cands[j].wl, &cands[j].v);
            if dev.is_finite() && dev <= p.cross_dev {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }
    let (mut seen, mut comps) = (vec![false; n], Vec::new());
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let (mut stack, mut comp) = (vec![s], Vec::new());
        while let Some(x) = stack.pop() {
            if seen[x] {
                continue;
            }
            seen[x] = true;
            comp.push(x);
            (0..n).filter(|y| adj[x][*y] && !seen[*y]).for_each(|y| stack.push(y));
        }
        comp.sort();
        comps.push(comp);
    }
    let key = |c: &Vec<usize>| {
        let d: Vec<f64> = c.iter().map(|i| cands[*i].dis).filter(|d| d.is_finite()).collect();
        (c.len(), -(if d.is_empty() { 9.0 } else { mean(&d) }))
    };
    let mut best = comps[0].clone();
    for c in &comps {
        if key(c) > key(&best) {
            best = c.clone();
        }
    }
    if best.len() == 1 {
        let rank = |i: usize| (if cands[i].dis.is_nan() { 9.0 } else { cands[i].dis }, cands[i].trust);
        let mut b = 0;
        for i in 0..n {
            if rank(i) < rank(b) {
                b = i;
            }
        }
        best = vec![b];
        report.push(format!("unvalidated site: no two spacings agree; kept d = {} m", cands[b].spacing));
    }
    for (i, c) in cands.iter().enumerate() {
        if best.contains(&i) {
            let (lo, hi) = (c.wl.iter().cloned().fold(f64::INFINITY, f64::min), c.wl.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            report.push(format!("d = {} m accepted: {} points, wavelength {lo:.1} to {hi:.1} m, trust x{:.1} [{}]", c.spacing, c.wl.len(), c.trust, c.label));
        } else {
            report.push(format!("d = {} m rejected, outside the consensus group [{}]", c.spacing, c.label));
        }
    }
    let (mut wl, mut v, mut tr) = (Vec::new(), Vec::new(), Vec::new());
    for &i in &best {
        wl.extend(&cands[i].wl);
        v.extend(&cands[i].v);
        tr.extend(std::iter::repeat_n(cands[i].trust, cands[i].wl.len()));
    }
    let (wc, vc, sg) = composite(&wl, &v, &tr, p)?;
    // record quality: share of the spectrum above 5 Hz whose coherence passes the mask
    let shares: Vec<f64> = tests
        .iter()
        .filter_map(|t| read_record(&t.run).ok().map(|r| (r, threshold(t.spacing, p))))
        .map(|(r, thr)| {
            let above: Vec<bool> = r.freq.iter().zip(&r.coh).filter(|(f, _)| **f > 5.0).map(|(_, c)| *c >= thr).collect();
            above.iter().filter(|b| **b).count() as f64 / above.len().max(1) as f64
        })
        .collect();
    let z_res = wc.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / p.res_div;
    let accepted = best.iter().map(|&i| SpacingCurve { spacing: cands[i].spacing, trust: cands[i].trust, wavelength: cands[i].wl.clone(), velocity: cands[i].v.clone() }).collect();
    Ok(Processed { wavelength: wc, velocity: vc, sigma: sg, accepted, report, records: tests.len(), coherent_share: if shares.is_empty() { f64::NAN } else { mean(&shares) }, z_res })
}
