//! The neural fusion estimator: a residual MLP that maps a dispersion curve and an optional
//! KPT sounding to the mean and standard deviation of ln Vs in graded layers.
//! Weights come from `tools/export_weights.py` (flat little-endian f32 with a JSON header).

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct Meta {
    #[serde(rename = "G")]
    pub g: usize,
    #[serde(rename = "K")]
    pub k: usize,
    pub nin: usize,
    pub nout: usize,
    pub vmu: f32,
    pub vsd: f32,
    /// Wavelength grid of the dispersion input (m).
    pub lam: Vec<f64>,
    /// Layer thicknesses (m); the last output is the half-space below them.
    pub thick: Vec<f64>,
    /// Mid-depths of the KPT increments (m).
    pub kz: Vec<f64>,
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    nin: usize,
}

impl Linear {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.w.chunks_exact(self.nin).zip(&self.b).map(|(row, b)| row.iter().zip(x).map(|(a, c)| a * c).sum::<f32>() + b).collect()
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        x.iter().zip(self.g.iter().zip(&self.b)).map(|(v, (g, b))| (v - mean) * inv * g + b).collect()
    }
}

struct Block {
    norm: Norm,
    l1: Linear,
    l2: Linear,
}

pub struct Model {
    pub meta: Meta,
    inp: Linear,
    blocks: Vec<Block>,
    out_norm: Norm,
    out: Linear,
    pub params: usize,
}

/// Posterior of ln Vs per layer.
#[derive(Clone, Debug, Default)]
pub struct Posterior {
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
}

impl Posterior {
    /// Index of the layer that contains depth `z`.
    pub fn layer_at(thick: &[f64], z: f64) -> usize {
        let mut top = 0.0;
        for (i, h) in thick.iter().enumerate() {
            top += h;
            if z < top {
                return i;
            }
        }
        thick.len()
    }

    pub fn vs_at(&self, thick: &[f64], z: f64) -> f64 {
        self.mean[Self::layer_at(thick, z).min(self.mean.len() - 1)].exp()
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<usize, String> {
        let s = self.b.get(self.p..self.p + 4).ok_or("weights file is truncated")?;
        self.p += 4;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.p..self.p + n).ok_or("weights file is truncated")?;
        self.p += n;
        Ok(s)
    }
}

impl Model {
    pub fn load(bytes: &[u8]) -> Result<Model, String> {
        let mut r = Reader { b: bytes, p: 0 };
        if r.bytes(4)? != b"VSF1" {
            return Err("not a VsFuse weights file".into());
        }
        let n = r.u32()?;
        let meta: Meta = serde_json::from_slice(r.bytes(n)?).map_err(|e| format!("weights header: {e}"))?;
        let count = r.u32()?;
        let mut t = std::collections::HashMap::new();
        let mut params = 0;
        for _ in 0..count {
            let n = r.u32()?;
            let name = String::from_utf8_lossy(r.bytes(n)?).to_string();
            let nd = r.u32()?;
            let mut len = 1;
            for _ in 0..nd {
                len *= r.u32()?;
            }
            let data: Vec<f32> = r.bytes(len * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            params += len;
            t.insert(name, data);
        }
        let mut take = |k: &str| t.remove(k).ok_or(format!("weights file has no tensor {k}"));
        let inp_b = take("inp.bias")?;
        let width = inp_b.len();
        let inp = Linear { w: take("inp.weight")?, b: inp_b, nin: meta.nin };
        let mut blocks = Vec::new();
        let mut i = 0;
        while let Ok(g) = take(&format!("blocks.{i}.f.0.weight")) {
            blocks.push(Block {
                norm: Norm { g, b: take(&format!("blocks.{i}.f.0.bias"))? },
                l1: Linear { w: take(&format!("blocks.{i}.f.1.weight"))?, b: take(&format!("blocks.{i}.f.1.bias"))?, nin: width },
                l2: Linear { w: take(&format!("blocks.{i}.f.3.weight"))?, b: take(&format!("blocks.{i}.f.3.bias"))?, nin: width },
            });
            i += 1;
        }
        let out_norm = Norm { g: take("out.0.weight")?, b: take("out.0.bias")? };
        let out = Linear { w: take("out.1.weight")?, b: take("out.1.bias")?, nin: width };
        if meta.nin != 3 * meta.g + 2 * meta.k || out.b.len() != 2 * meta.nout {
            return Err("weights file does not match its header".into());
        }
        Ok(Model { meta, inp, blocks, out_norm, out, params })
    }

    /// `lnv`, `mask`, `sig` on the wavelength grid; `kpt` (ln Vs from the sounding) and its mask
    /// on the increment grid. Masked entries are ignored.
    pub fn infer(&self, lnv: &[f64], mask: &[bool], sig: &[f64], kpt: &[f64], kmask: &[bool]) -> Posterior {
        let m = &self.meta;
        let mut x = Vec::with_capacity(m.nin);
        x.extend(lnv.iter().zip(mask).map(|(v, k)| if *k { (*v as f32 - m.vmu) / m.vsd } else { 0.0 }));
        x.extend(mask.iter().map(|k| if *k { 1.0 } else { 0.0 }));
        x.extend(sig.iter().zip(mask).map(|(s, k)| if *k { (s.clamp(0.05, 0.4) * 5.0) as f32 } else { 0.0 }));
        x.extend(kpt.iter().zip(kmask).map(|(v, k)| if *k { (*v as f32 - m.vmu) / m.vsd } else { 0.0 }));
        x.extend(kmask.iter().map(|k| if *k { 1.0 } else { 0.0 }));
        let mut h = self.inp.apply(&x);
        for b in &self.blocks {
            let mut a = b.l1.apply(&b.norm.apply(&h));
            a.iter_mut().for_each(|v| *v = gelu(*v));
            let a = b.l2.apply(&a);
            h.iter_mut().zip(&a).for_each(|(h, a)| *h += a);
        }
        let o = self.out.apply(&self.out_norm.apply(&h));
        Posterior {
            mean: o[..m.nout].iter().map(|v| (v * m.vsd + m.vmu) as f64).collect(),
            sd: o[m.nout..].iter().map(|v| (v.clamp(-6.0, 3.0) as f64).exp()).collect(),
        }
    }
}
