//! Runs the estimator on a foundation and writes the results table.

use std::collections::HashMap;

use crate::data::{Project, Site};
use crate::decision::{evaluate, Rule};
use crate::model::{Model, Posterior};

#[derive(Clone, Debug, Default)]
pub struct SiteResult {
    /// From the dispersion curve alone.
    pub sasw: Posterior,
    /// With the sounding, where the site has admissible increments.
    pub fused: Option<Posterior>,
    pub wavelengths: usize,
    pub increments: usize,
    pub ms: f64,
}

impl SiteResult {
    /// The posterior the decision uses.
    pub fn best(&self, use_kpt: bool) -> &Posterior {
        match (&self.fused, use_kpt) {
            (Some(f), true) => f,
            _ => &self.sasw,
        }
    }
}

pub fn run_site(model: &Model, site: &Site, corrected: bool) -> SiteResult {
    let t0 = std::time::Instant::now();
    let a = site.inputs(&model.meta, corrected, false);
    let sasw = model.infer(&a.lnv, &a.mask, &a.sig, &a.kpt, &a.kmask);
    let b = site.inputs(&model.meta, corrected, true);
    let increments = b.kmask.iter().filter(|k| **k).count();
    let fused = (increments > 0).then(|| model.infer(&b.lnv, &b.mask, &b.sig, &b.kpt, &b.kmask));
    let n = if fused.is_some() { 2.0 } else { 1.0 };
    SiteResult { sasw, fused, wavelengths: a.mask.iter().filter(|k| **k).count(), increments, ms: t0.elapsed().as_secs_f64() * 1000.0 / n }
}

pub fn results_csv(project: &Project, results: &HashMap<String, SiteResult>, thick: &[f64], rule: &Rule, use_kpt: bool) -> String {
    let mut s = String::from("site,sounding_used,z_res_m,vs_at_df_mps,sd_ln_vs,p_vs_ge_soft,class,qa_kpa,g0_mpa,bmax_m,bearing_layer_m,recommendation\n");
    for (name, site) in &project.sites {
        let Some(r) = results.get(name) else { continue };
        let e = evaluate(r.best(use_kpt), thick, site.z_res, rule);
        let opt = |v: Option<f64>| v.map(|x| format!("{x:.2}")).unwrap_or_default();
        s += &format!(
            "{name},{},{:.2},{:.0},{:.3},{:.2},{},{:.0},{:.0},{},{},{}\n",
            if use_kpt && r.fused.is_some() { "yes" } else { "no" },
            site.z_res,
            e.vs,
            e.sd,
            e.p_firm,
            e.class,
            e.qa,
            e.g0,
            opt(e.bmax),
            opt(e.bearing_depth),
            e.verdict.text()
        );
    }
    s
}

/// Posterior of every layer of one site, as a table.
pub fn profile_csv(p: &Posterior, thick: &[f64]) -> String {
    let mut s = String::from("top_m,bottom_m,vs_mps,vs_lo95_mps,vs_hi95_mps,sd_ln_vs\n");
    let mut top = 0.0;
    for i in 0..p.mean.len() {
        let bot = thick.get(i).map(|h| top + h);
        s += &format!("{top:.2},{},{:.0},{:.0},{:.0},{:.3}\n", bot.map(|b| format!("{b:.2}")).unwrap_or_else(|| "half-space".into()), p.mean[i].exp(), (p.mean[i] - 1.96 * p.sd[i]).exp(), (p.mean[i] + 1.96 * p.sd[i]).exp(), p.sd[i]);
        top = bot.unwrap_or(top);
    }
    s
}
