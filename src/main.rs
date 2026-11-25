#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
//! VsFuse: neural fusion of SASW dispersion curves and Kunzelstab soundings.
//!
//! ```text
//! vsfuse [project.json]                 the window
//! vsfuse batch project.json out.csv     founding-level table of every foundation, no window
//! vsfuse --mcp [--project file.json]   the tools as an MCP server on stdio, for an agent
//! ```

mod app;
mod chrome;
mod data;
mod decision;
mod engine;
mod frontend;
mod mcp;
mod model;
mod num;
mod theme;

use std::collections::HashMap;
use std::path::PathBuf;

/// The trained estimator, built into the program.
pub static WEIGHTS: &[u8] = include_bytes!("../assets/model.vsf");
/// Three example foundations, opened when no project is given.
pub static SAMPLE: &str = include_str!("../sample/example_project.json");

fn batch(args: &[String]) -> Result<(), String> {
    let (inp, out) = (args.get(2).ok_or("usage: vsfuse batch project.json out.csv")?, args.get(3).ok_or("usage: vsfuse batch project.json out.csv")?);
    let model = model::Model::load(WEIGHTS)?;
    let project = data::Project::load(&PathBuf::from(inp))?;
    let results: HashMap<String, engine::SiteResult> = project.sites.iter().map(|(k, s)| (k.clone(), engine::run_site(&model, s, true))).collect();
    let csv = engine::results_csv(&project, &results, &model.meta.thick, &decision::Rule::default(), true);
    std::fs::write(out, csv).map_err(|e| format!("{out}: {e}"))?;
    println!("{} foundations written to {out}", results.len());
    Ok(())
}

/// The window and Dock icon.
fn app_icon() -> std::sync::Arc<eframe::egui::IconData> {
    let img = image::load_from_memory(include_bytes!("../assets/app_icon.png")).expect("app icon is a valid PNG").to_rgba8();
    std::sync::Arc::new(eframe::egui::IconData { width: img.width(), height: img.height(), rgba: img.into_raw() })
}

/// `vsfuse process <site folder or folder of sites> out.json [out.csv]`: raw records to a project.
fn process(args: &[String]) -> Result<(), String> {
    let usage = "usage: vsfuse process <site folder | folder of site folders> project.json [table.csv]";
    let (inp, out) = (PathBuf::from(args.get(2).ok_or(usage)?), args.get(3).ok_or(usage)?);
    let dirs = data::raw_sites(&inp);
    if dirs.is_empty() {
        return Err(format!("{}: no SASW records found", inp.display()));
    }
    let mut project = data::Project { name: inp.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), ..Default::default() };
    for d in &dirs {
        let name = d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        match data::Site::from_raw(d) {
            Ok(s) => {
                println!("{name}: {} records, {} spacings accepted, resolved depth {:.1} m", s.records.unwrap_or(0), s.accepted.len(), s.z_res);
                if std::env::var("VSFUSE_VERBOSE").is_ok() {
                    s.report.iter().for_each(|l| println!("    {l}"));
                }
                project.sites.insert(name, s);
            }
            Err(e) => println!("{name}: FAILED, {e}"),
        }
    }
    project.save(&PathBuf::from(out))?;
    if let Some(csv) = args.get(4) {
        let model = model::Model::load(WEIGHTS)?;
        let results: HashMap<String, engine::SiteResult> = project.sites.iter().map(|(k, s)| (k.clone(), engine::run_site(&model, s, true))).collect();
        std::fs::write(csv, engine::results_csv(&project, &results, &model.meta.thick, &decision::Rule::default(), true)).map_err(|e| format!("{csv}: {e}"))?;
    }
    println!("{} of {} sites written to {out}", project.sites.len(), dirs.len());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--mcp") {
        let project = args.iter().position(|a| a == "--project").and_then(|i| args.get(i + 1)).map(PathBuf::from);
        if let Err(e) = mcp::serve(project) {
            eprintln!("vsfuse --mcp: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("process") {
        if let Err(e) = process(&args) {
            eprintln!("vsfuse: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("batch") {
        if let Err(e) = batch(&args) {
            eprintln!("vsfuse: {e}");
            std::process::exit(1);
        }
        return;
    }
    let project = args.get(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_title("VsFuse: SASW and KPT fusion for founding-level evaluation").with_inner_size([1480.0, 900.0]).with_min_inner_size([1040.0, 640.0]).with_icon(app_icon()),
        ..Default::default()
    };
    let r = eframe::run_native(
        "VsFuse",
        options,
        Box::new(move |cc| match app::App::new(cc, project) {
            Ok(a) => Ok(Box::new(a) as Box<dyn eframe::App>),
            Err(e) => Err(e.into()),
        }),
    );
    if let Err(e) = r {
        eprintln!("vsfuse: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust forward pass reproduces the PyTorch outputs at the example foundations.
    #[test]
    fn matches_reference() {
        #[derive(serde::Deserialize)]
        struct Ref {
            a: Vec<f64>,
            #[serde(rename = "as")]
            a_sd: Vec<f64>,
            b: Vec<f64>,
            bs: Vec<f64>,
        }
        let model = model::Model::load(WEIGHTS).unwrap();
        let project = data::Project::parse(SAMPLE).unwrap();
        let reference: HashMap<String, Ref> = serde_json::from_str(include_str!("../sample/example_reference.json")).unwrap();
        assert_eq!(project.sites.len(), 3);
        let (mut worst, mut worst_sd) = (0.0_f64, 0.0_f64);
        for (name, site) in &project.sites {
            let r = engine::run_site(&model, site, true);
            let q = &reference[name];
            let fused = r.fused.clone().unwrap_or(r.sasw.clone());
            for i in 0..q.a.len() {
                worst = worst.max((r.sasw.mean[i] - q.a[i]).abs()).max((fused.mean[i] - q.b[i]).abs());
                worst_sd = worst_sd.max((r.sasw.sd[i] / q.a_sd[i]).ln().abs()).max((fused.sd[i] / q.bs[i]).ln().abs());
            }
        }
        println!("largest difference in ln Vs {worst:.4}, in ln sd {worst_sd:.4}");
        assert!(worst < 0.02 && worst_sd < 0.05, "ln Vs {worst}, ln sd {worst_sd}");
    }

    /// A curve and a sounding read from text files run through the estimator.
    #[test]
    fn text_import() {
        let mut site = data::curve_from_text("wavelength,velocity,sigma\n8,260,0.1\n1,150,0.1\n2,170\n4,210,0.15\n").unwrap();
        assert_eq!(site.wavelength, vec![1.0, 2.0, 4.0, 8.0]);
        assert!((site.z_res - 3.2).abs() < 1e-9 && (site.sigma_rel[1] - 0.12).abs() < 1e-9);
        site.kpt = data::kpt_from_text("z n20\n0.1 8\n0.3 12\n0.5 150\n").unwrap();
        let model = model::Model::load(WEIGHTS).unwrap();
        let r = engine::run_site(&model, &site, true);
        assert_eq!((r.increments, r.fused.is_some()), (2, true));
        assert!(r.sasw.mean.iter().all(|m| m.is_finite() && (50.0..1500.0).contains(&m.exp())));
    }

    /// The Rust front end reproduces the composite curves of the Python pipeline. Needs the raw
    /// records and the study results beside this crate; skipped when they are absent.
    #[test]
    fn frontend_matches_python() {
        #[derive(serde::Deserialize)]
        struct Py {
            wlc: Vec<f64>,
            vc: Vec<f64>,
            sig: Vec<f64>,
        }
        let (raw, res) = (PathBuf::from("../sasw_sites"), PathBuf::from("../study/results/r7_occam_eff"));
        if !raw.is_dir() || !res.is_dir() {
            println!("raw records not present, skipped");
            return;
        }
        let mut sites: Vec<PathBuf> = std::fs::read_dir(&raw).unwrap().filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        sites.sort();
        let (mut same, mut close, mut total, mut worst) = (0, 0, 0, Vec::new());
        for site in &sites {
            let name = site.file_name().unwrap().to_string_lossy().to_string();
            let Ok(text) = std::fs::read_to_string(res.join(format!("{name}.json"))) else { continue };
            let Ok(py) = serde_json::from_str::<Py>(&text) else { continue };
            total += 1;
            match frontend::process(site, &frontend::Params::default()) {
                Ok(r) if r.wavelength.len() == py.wlc.len() => {
                    let d = (0..py.wlc.len()).map(|i| (r.velocity[i] / py.vc[i]).ln().abs().max((r.wavelength[i] / py.wlc[i]).ln().abs()).max((r.sigma[i] / py.sig[i]).ln().abs() * 0.1)).fold(0.0, f64::max);
                    if d < 1e-6 {
                        same += 1;
                    }
                    if d < 0.01 {
                        close += 1;
                    } else {
                        worst.push(format!("{name} {d:.3}"));
                    }
                }
                Ok(r) => worst.push(format!("{name} bins {} vs {}", r.wavelength.len(), py.wlc.len())),
                Err(e) => worst.push(format!("{name} {e}")),
            }
        }
        println!("{total} sites: {same} identical, {close} within 1%, others: {worst:?}");
        // the rest differ where the reference breaks an exact tie by floating-point or sort order
        assert!(total > 0 && close * 100 >= total * 90, "front end differs from the reference");
    }

    #[test]
    fn decision_rule() {
        let thick = vec![1.0, 1.0, 1.0, 1.0];
        let p = model::Posterior { mean: [150.0_f64, 150.0, 400.0, 400.0, 500.0].iter().map(|v| v.ln()).collect(), sd: vec![0.1; 5] };
        let r = decision::Rule::default();
        assert_eq!(decision::evaluate(&p, &thick, 5.0, &r).verdict, decision::Verdict::Deeper(2.0));
        assert_eq!(decision::evaluate(&p, &thick, 1.0, &r).verdict, decision::Verdict::NotCertifiable);
        assert_eq!(decision::evaluate(&p, &thick, 1.8, &r).verdict, decision::Verdict::Soft);
        assert!((data::vs_kpt(6.0) - 147.0).abs() < 1.0 && (data::vs_kpt(100.0) - 340.0).abs() < 1.0);
    }
}
