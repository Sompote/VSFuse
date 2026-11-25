//! Founding-level evaluation of one foundation from the posterior of ln Vs.

use crate::model::Posterior;

#[derive(Clone, Copy, Debug)]
pub struct Rule {
    /// Planned founding depth (m).
    pub df: f64,
    /// Unit weight (kN/m3) of the screening pressure q_a = 0.024 gamma Vs.
    pub gamma: f64,
    /// Density (Mg/m3) of G0 = rho Vs^2.
    pub rho: f64,
    /// Velocity below which the ground is called soft (m/s).
    pub soft: f64,
    /// Velocity of the bearing layer (m/s).
    pub bearing: f64,
    /// A bearing layer deeper than this is not a founding option (m).
    pub max_deeper: f64,
}

impl Default for Rule {
    fn default() -> Self {
        Rule { df: 1.5, gamma: 18.1, rho: 1.85, soft: 180.0, bearing: 360.0, max_deeper: 6.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    Found,
    Deeper(f64),
    Soft,
    NotCertifiable,
}

impl Verdict {
    pub fn text(&self) -> String {
        match self {
            Verdict::Found => "found at the planned level".into(),
            Verdict::Deeper(d) => format!("soft at the planned level; found at {d:.1} m"),
            Verdict::Soft => "soft within the resolved depth".into(),
            Verdict::NotCertifiable => "not certifiable from these records".into(),
        }
    }

    pub fn short(&self) -> &'static str {
        match self {
            Verdict::Found => "found",
            Verdict::Deeper(_) => "deeper",
            Verdict::Soft => "soft",
            Verdict::NotCertifiable => "n/c",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Evaluation {
    pub vs: f64,
    /// Standard deviation of ln Vs at the founding depth.
    pub sd: f64,
    /// Probability that Vs at the founding depth reaches the soft limit.
    pub p_firm: f64,
    pub qa: f64,
    pub g0: f64,
    pub bmax: Option<f64>,
    pub bearing_depth: Option<f64>,
    pub class: char,
    pub verdict: Verdict,
}

fn phi(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

pub fn evaluate(p: &Posterior, thick: &[f64], z_res: f64, r: &Rule) -> Evaluation {
    let k = Posterior::layer_at(thick, r.df).min(p.mean.len() - 1);
    let vs = p.mean[k].exp();
    let p_firm = phi((p.mean[k] - r.soft.ln()) / p.sd[k].max(1e-6));
    let mut top = 0.0;
    let mut bearing_depth = None;
    for (i, h) in thick.iter().enumerate() {
        if p.mean[i].exp() >= r.bearing && top <= z_res {
            bearing_depth = Some(top);
            break;
        }
        top += h;
    }
    let verdict = if z_res <= r.df {
        Verdict::NotCertifiable
    } else if vs >= r.soft {
        Verdict::Found
    } else {
        match bearing_depth {
            Some(d) if d <= z_res.min(r.max_deeper) => Verdict::Deeper(d),
            _ => Verdict::Soft,
        }
    };
    Evaluation {
        vs,
        sd: p.sd[k],
        p_firm,
        qa: 0.024 * r.gamma * vs,
        g0: r.rho * vs * vs / 1000.0,
        bmax: (z_res > r.df).then(|| (z_res - r.df) / 1.5),
        bearing_depth,
        class: if vs < 180.0 { 'E' } else if vs < 360.0 { 'D' } else { 'C' },
        verdict,
    }
}
