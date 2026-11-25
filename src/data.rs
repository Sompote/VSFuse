//! Project data: one dispersion curve and an optional KPT sounding per foundation.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::Meta;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct KptRow {
    /// Mid-depth of the 200 mm increment (m).
    pub z: f64,
    /// Blows per 200 mm.
    pub n20: f64,
    /// False for a partial increment (refusal before 200 mm).
    #[serde(default = "yes")]
    pub full: bool,
}

fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Site {
    /// Composite dispersion curve: wavelength (m), phase velocity (m/s), relative uncertainty.
    pub wavelength: Vec<f64>,
    pub velocity: Vec<f64>,
    #[serde(default)]
    pub sigma_rel: Vec<f64>,
    /// Apparent / fundamental-mode velocity from a reference inversion, where one was run.
    #[serde(default)]
    pub mode_ratio: Vec<f64>,
    /// Resolved depth; taken as the longest wavelength / 2.5 when absent.
    #[serde(default)]
    pub z_res: f64,
    #[serde(default)]
    pub lam_min: f64,
    #[serde(default)]
    pub kpt: Vec<KptRow>,
    /// Accepted dispersion points by spacing, when the curve came from raw records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted: Vec<crate::frontend::SpacingCurve>,
    /// Processing report of the raw records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub report: Vec<String>,
    /// Share of the spectrum whose coherence passes the mask (below 0.30: low-coherence site).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coherent_share: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records: Option<usize>,
}

impl Site {
    /// Processes the raw SASW records of a site folder into a site.
    pub fn from_raw(dir: &Path) -> Result<Site, String> {
        let r = crate::frontend::process(dir, &crate::frontend::Params::default())?;
        let mut s = Site {
            sigma_rel: r.sigma.iter().zip(&r.velocity).map(|(s, v)| s / v).collect(),
            wavelength: r.wavelength,
            velocity: r.velocity,
            z_res: r.z_res,
            accepted: r.accepted,
            report: r.report,
            coherent_share: r.coherent_share.is_finite().then_some(r.coherent_share),
            records: Some(r.records),
            ..Default::default()
        };
        s.normalise();
        Ok(s)
    }
}

/// Site folders under `dir`: the folder itself when it holds records, else its subfolders that do.
pub fn raw_sites(dir: &Path) -> Vec<std::path::PathBuf> {
    if crate::frontend::is_site(dir) {
        return vec![dir.to_path_buf()];
    }
    let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir() && crate::frontend::is_site(p)).collect()).unwrap_or_default();
    v.sort();
    v
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Project {
    #[serde(default)]
    pub name: String,
    pub sites: BTreeMap<String, Site>,
}

/// KPT blow count to shear-wave velocity: N = 0.539 (N20 + 0.954), Vs = 96.9 N^0.314.
pub fn vs_kpt(n20: f64) -> f64 {
    96.9 * (0.539 * (n20 + 0.954)).powf(0.314)
}

/// The conversion holds for full increments with 6 <= N20 <= 100 (147 to 340 m/s).
pub fn admissible(r: &KptRow) -> bool {
    r.full && (6.0..=100.0).contains(&r.n20)
}

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    for i in 1..xs.len() {
        if x <= xs[i] {
            let t = if xs[i] > xs[i - 1] { (x - xs[i - 1]) / (xs[i] - xs[i - 1]) } else { 0.0 };
            return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
    }
    ys[ys.len() - 1]
}

/// Network inputs of one foundation.
pub struct Inputs {
    pub lnv: Vec<f64>,
    pub mask: Vec<bool>,
    pub sig: Vec<f64>,
    pub kpt: Vec<f64>,
    pub kmask: Vec<bool>,
}

impl Site {
    pub fn normalise(&mut self) {
        let mut idx: Vec<usize> = (0..self.wavelength.len().min(self.velocity.len())).filter(|&i| self.wavelength[i] > 0.0 && self.velocity[i] > 0.0).collect();
        idx.sort_by(|a, b| self.wavelength[*a].total_cmp(&self.wavelength[*b]));
        let pick = |v: &Vec<f64>, d: f64| -> Vec<f64> { idx.iter().map(|&i| v.get(i).copied().unwrap_or(d)).collect() };
        let (w, v, s) = (pick(&self.wavelength, 0.0), pick(&self.velocity, 0.0), pick(&self.sigma_rel, 0.12));
        let r = if self.mode_ratio.len() == self.wavelength.len() { pick(&self.mode_ratio, 1.0) } else { vec![] };
        self.wavelength = w;
        self.velocity = v;
        self.sigma_rel = s;
        self.mode_ratio = r;
        if let (Some(lo), Some(hi)) = (self.wavelength.first(), self.wavelength.last()) {
            if self.z_res <= 0.0 {
                self.z_res = hi / 2.5;
            }
            if self.lam_min <= 0.0 {
                self.lam_min = *lo;
            }
        }
        self.kpt.sort_by(|a, b| a.z.total_cmp(&b.z));
    }

    pub fn has_kpt(&self) -> bool {
        self.kpt.iter().any(admissible)
    }

    /// Curve and sounding on the grids of the network. `corrected` divides the curve by the
    /// mode ratio where the site has one.
    pub fn inputs(&self, m: &Meta, corrected: bool, use_kpt: bool) -> Inputs {
        let n = self.wavelength.len();
        let ratio = |i: usize| if corrected && self.mode_ratio.len() == n { self.mode_ratio[i] } else { 1.0 };
        let lx: Vec<f64> = self.wavelength.iter().map(|w| w.ln()).collect();
        let ly: Vec<f64> = (0..n).map(|i| (self.velocity[i] / ratio(i)).ln()).collect();
        let (mut lnv, mut mask, mut sig) = (vec![0.0; m.g], vec![false; m.g], vec![0.0; m.g]);
        if n >= 2 {
            for (j, lam) in m.lam.iter().enumerate() {
                if *lam >= self.wavelength[0] && *lam <= self.wavelength[n - 1] {
                    mask[j] = true;
                    lnv[j] = interp(lam.ln(), &lx, &ly);
                    sig[j] = interp(lam.ln(), &lx, &self.sigma_rel);
                }
            }
        }
        let (mut kpt, mut kmask) = (vec![0.0; m.k], vec![false; m.k]);
        if use_kpt {
            for r in self.kpt.iter().filter(|r| admissible(r)) {
                let k = ((r.z - m.kz[0]) / 0.2).round();
                if k >= 0.0 && (k as usize) < m.k && (m.kz[k as usize] - r.z).abs() < 0.05 {
                    kpt[k as usize] = vs_kpt(r.n20).ln();
                    kmask[k as usize] = true;
                }
            }
        }
        Inputs { lnv, mask, sig, kpt, kmask }
    }
}

impl Project {
    pub fn load(path: &Path) -> Result<Project, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Project, String> {
        let mut p: Project = serde_json::from_str(text).map_err(|e| format!("project file: {e}"))?;
        p.sites.values_mut().for_each(Site::normalise);
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, serde_json::to_string(self).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Numeric rows of a CSV or whitespace-separated text file; lines that do not start with
/// a number (headers, comments) are skipped.
pub fn numeric_rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter_map(|l| {
            let v: Vec<f64> = l.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).filter(|s| !s.is_empty()).map(|s| s.parse::<f64>()).collect::<Result<_, _>>().ok()?;
            (v.len() >= 2).then_some(v)
        })
        .collect()
}

/// Dispersion curve from a text file: wavelength (m), velocity (m/s), optional relative uncertainty.
pub fn curve_from_text(text: &str) -> Result<Site, String> {
    let rows = numeric_rows(text);
    if rows.len() < 3 {
        return Err("a dispersion curve needs at least three rows of: wavelength, velocity".into());
    }
    let mut s = Site { wavelength: rows.iter().map(|r| r[0]).collect(), velocity: rows.iter().map(|r| r[1]).collect(), sigma_rel: rows.iter().map(|r| r.get(2).copied().unwrap_or(0.12)).collect(), ..Default::default() };
    s.normalise();
    Ok(s)
}

/// KPT sounding from a text file: mid-depth (m), N20.
pub fn kpt_from_text(text: &str) -> Result<Vec<KptRow>, String> {
    let rows = numeric_rows(text);
    if rows.is_empty() {
        return Err("a sounding needs rows of: depth, N20".into());
    }
    Ok(rows.iter().map(|r| KptRow { z: r[0], n20: r[1], full: true }).collect())
}
