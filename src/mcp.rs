//! `vsfuse --mcp`: the program's functions as a Model Context Protocol server on stdio, so an
//! agent can process raw SASW records, attach soundings, read profiles and decisions, and
//! export tables. One project lives in memory for the length of the session.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::data::{self, KptRow, Project, Site};
use crate::decision::{evaluate, Rule};
use crate::engine::{self, SiteResult};
use crate::model::Model;

struct State {
    model: Model,
    project: Project,
    rule: Rule,
    use_kpt: bool,
    results: HashMap<String, SiteResult>,
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"name": name, "description": description, "inputSchema": {"type": "object", "properties": properties, "required": required}})
}

fn tools() -> Value {
    let num = |d: &str| json!({"type": "number", "description": d});
    let text = |d: &str| json!({"type": "string", "description": d});
    let arr = |d: &str| json!({"type": "array", "items": {"type": "number"}, "description": d});
    json!([
        tool("new_project", "Start an empty project in memory. Units everywhere: m, m/s, kPa, MPa.", json!({"name": text("Project name")}), &[]),
        tool("open_project", "Open a VsFuse project file (JSON) and replace the project in memory.", json!({"path": text("Path of the project file")}), &["path"]),
        tool("save_project", "Save the project in memory as a VsFuse project file (JSON).", json!({"path": text("Path to write")}), &["path"]),
        tool("process_raw", "Process raw SASW records into dispersion curves and add the sites to the project. `path` is the folder of one site (record folders named like H4mF_HM.trf, each with Run*/ASCII/Hxysv00000.txt and Cxysv00000.txt) or a folder of many site folders. Returns, per site, the number of runs, the accepted spacings and the resolved depth.", json!({"path": text("Site folder or folder of site folders")}), &["path"]),
        tool("add_curve", "Add a foundation from a composite dispersion curve given as arrays.", json!({"site": text("Name of the foundation"), "wavelength": arr("Wavelengths (m)"), "velocity": arr("Phase velocities (m/s)"), "sigma_rel": arr("Relative uncertainties, optional (default 0.12)")}), &["site", "wavelength", "velocity"]),
        tool("attach_kpt", "Attach a Kunzelstab (KPT) sounding to a foundation. Only full increments with 6 <= N20 <= 100 enter the fusion.", json!({"site": text("Name of the foundation"), "depth": arr("Mid-depth of each 200 mm increment (m)"), "n20": arr("Blows per 200 mm")}), &["site", "depth", "n20"]),
        tool("remove_site", "Remove a foundation from the project.", json!({"site": text("Name of the foundation")}), &["site"]),
        tool("list_sites", "List the foundations with Vs at the founding depth, the probability that it reaches the soft limit, the verdict and whether a sounding is fused.", json!({}), &[]),
        tool("get_site", "Full result of one foundation: decision, posterior of Vs per layer (mean and 95 % range) from SASW alone and fused with the KPT, the dispersion curve, and the processing report of its raw records.", json!({"site": text("Name of the foundation")}), &["site"]),
        tool("get_rule", "The founding-level rule in use.", json!({}), &[]),
        tool("set_rule", "Change the founding-level rule; every foundation is classified again. Give only the values to change.", json!({"founding_depth": num("Planned founding depth (m), default 1.5"), "soft_below": num("Velocity below which the ground is soft (m/s), default 180"), "bearing_from": num("Velocity of the bearing layer (m/s), default 360"), "deepest_option": num("A bearing layer deeper than this is not a founding option (m), default 6"), "unit_weight": num("Unit weight for the screening pressure (kN/m3), default 18.1"), "use_kpt": {"type": "boolean", "description": "Fuse soundings where they exist, default true"}}), &[]),
        tool("summary", "Counts of the verdicts over the project and the foundations whose decision is uncertain (probability between 0.16 and 0.84), which are the ones to test again.", json!({}), &[]),
        tool("export_table", "Write the founding-level table of every foundation as CSV.", json!({"path": text("Path to write")}), &["path"]),
        tool("export_profile", "Write the Vs profile of one foundation (per layer: mean, 95 % range, sd of ln Vs) as CSV.", json!({"site": text("Name of the foundation"), "path": text("Path to write")}), &["site", "path"]),
        tool("screenshot", "Render the program window for one foundation to a PNG (dispersion curve, Vs profile, decision). Needs a display.", json!({"site": text("Name of the foundation"), "path": text("PNG path to write"), "theme": text("light or dark, default light")}), &["site", "path"]),
    ])
}

impl State {
    fn result(&mut self, name: &str) -> Result<&SiteResult, String> {
        let site = self.project.sites.get(name).ok_or_else(|| format!("no foundation named {name}; call list_sites"))?;
        if !self.results.contains_key(name) {
            let r = engine::run_site(&self.model, site, true);
            self.results.insert(name.to_string(), r);
        }
        Ok(&self.results[name])
    }

    fn row(&mut self, name: &str) -> Result<Value, String> {
        let (rule, use_kpt, thick) = (self.rule, self.use_kpt, self.model.meta.thick.clone());
        let z_res = self.project.sites.get(name).map(|s| s.z_res).unwrap_or(0.0);
        let r = self.result(name)?;
        let e = evaluate(r.best(use_kpt), &thick, z_res, &rule);
        Ok(json!({"site": name, "vs_at_founding_depth": e.vs.round(), "probability_firm": (e.p_firm * 100.0).round() / 100.0, "confident": e.p_firm >= 0.84 || e.p_firm <= 0.16, "verdict": e.verdict.text(), "resolved_depth_m": (z_res * 10.0).round() / 10.0, "sounding_fused": use_kpt && r.fused.is_some()}))
    }

    fn call(&mut self, name: &str, a: &Value) -> Result<Value, String> {
        let s = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string).ok_or(format!("missing argument {k}"));
        let nums = |k: &str| -> Result<Vec<f64>, String> { a.get(k).and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_f64).collect()).ok_or(format!("missing argument {k}")) };
        let r1 = |x: f64| (x * 10.0).round() / 10.0;
        match name {
            "new_project" => {
                self.project = Project { name: a.get("name").and_then(Value::as_str).unwrap_or("").to_string(), ..Default::default() };
                self.results.clear();
                Ok(json!({"ok": true}))
            }
            "open_project" => {
                self.project = Project::load(&PathBuf::from(s("path")?))?;
                self.results.clear();
                Ok(json!({"name": self.project.name, "sites": self.project.sites.len()}))
            }
            "save_project" => {
                let p = s("path")?;
                self.project.save(&PathBuf::from(&p))?;
                Ok(json!({"written": p, "sites": self.project.sites.len()}))
            }
            "process_raw" => {
                let dir = PathBuf::from(s("path")?);
                let dirs = data::raw_sites(&dir);
                if dirs.is_empty() {
                    return Err(format!("{}: no SASW records found (record folders like H4mF_HM.trf with Run*/ASCII exports)", dir.display()));
                }
                let mut out = Vec::new();
                for d in dirs {
                    let n = d.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
                    match Site::from_raw(&d) {
                        Ok(site) => {
                            out.push(json!({"site": n, "runs": site.records, "accepted_spacings_m": site.accepted.iter().map(|c| c.spacing).collect::<Vec<_>>(), "resolved_depth_m": r1(site.z_res), "coherent_share": site.coherent_share.map(|c| (c * 100.0).round() / 100.0), "low_coherence": site.coherent_share.is_some_and(|c| c < 0.30)}));
                            self.results.remove(&n);
                            self.project.sites.insert(n, site);
                        }
                        Err(e) => out.push(json!({"site": n, "failed": e})),
                    }
                }
                Ok(json!({"processed": out}))
            }
            "add_curve" => {
                let (n, w, v) = (s("site")?, nums("wavelength")?, nums("velocity")?);
                if w.len() < 3 || w.len() != v.len() {
                    return Err("wavelength and velocity need the same length, at least three points".into());
                }
                let mut site = Site { sigma_rel: nums("sigma_rel").unwrap_or_else(|_| vec![0.12; w.len()]), wavelength: w, velocity: v, ..Default::default() };
                site.normalise();
                let z = site.z_res;
                self.results.remove(&n);
                self.project.sites.insert(n.clone(), site);
                Ok(json!({"site": n, "resolved_depth_m": r1(z)}))
            }
            "attach_kpt" => {
                let (n, z, b) = (s("site")?, nums("depth")?, nums("n20")?);
                if z.is_empty() || z.len() != b.len() {
                    return Err("depth and n20 need the same length".into());
                }
                let site = self.project.sites.get_mut(&n).ok_or_else(|| format!("no foundation named {n}"))?;
                site.kpt = z.iter().zip(&b).map(|(z, b)| KptRow { z: *z, n20: *b, full: true }).collect();
                site.normalise();
                let used = site.kpt.iter().filter(|r| data::admissible(r)).count();
                self.results.remove(&n);
                Ok(json!({"site": n, "increments": z.len(), "inside_conversion_range": used}))
            }
            "remove_site" => {
                let n = s("site")?;
                self.results.remove(&n);
                Ok(json!({"removed": self.project.sites.remove(&n).is_some()}))
            }
            "list_sites" => {
                let names: Vec<String> = self.project.sites.keys().cloned().collect();
                Ok(json!({"project": self.project.name, "sites": names.iter().map(|n| self.row(n)).collect::<Result<Vec<_>, _>>()?}))
            }
            "get_site" => {
                let n = s("site")?;
                let (rule, use_kpt, thick) = (self.rule, self.use_kpt, self.model.meta.thick.clone());
                let site = self.project.sites.get(&n).cloned().ok_or_else(|| format!("no foundation named {n}; call list_sites"))?;
                let r = self.result(&n)?.clone();
                let e = evaluate(r.best(use_kpt), &thick, site.z_res, &rule);
                let layers = |p: &crate::model::Posterior| -> Vec<Value> {
                    let mut top = 0.0;
                    (0..p.mean.len())
                        .map(|i| {
                            let bot = thick.get(i).map(|h| top + h);
                            let v = json!({"top_m": r1(top), "bottom_m": bot.map(r1), "vs": p.mean[i].exp().round(), "vs_lo95": (p.mean[i] - 1.96 * p.sd[i]).exp().round(), "vs_hi95": (p.mean[i] + 1.96 * p.sd[i]).exp().round(), "resolved": top < site.z_res});
                            top = bot.unwrap_or(top);
                            v
                        })
                        .collect()
                };
                Ok(json!({
                    "site": n,
                    "decision": {"verdict": e.verdict.text(), "vs_at_founding_depth": e.vs.round(), "sd_ln_vs": (e.sd * 1000.0).round() / 1000.0, "probability_firm": (e.p_firm * 100.0).round() / 100.0, "confident": e.p_firm >= 0.84 || e.p_firm <= 0.16, "site_class": e.class.to_string(), "screening_pressure_kpa": e.qa.round(), "g0_mpa": e.g0.round(), "certifiable_width_m": e.bmax.map(r1), "bearing_layer_depth_m": e.bearing_depth.map(r1)},
                    "resolved_depth_m": r1(site.z_res),
                    "sounding": {"increments": site.kpt.len(), "fused": r.increments, "used_in_decision": use_kpt && r.fused.is_some()},
                    "profile_sasw_only": layers(&r.sasw),
                    "profile_fused": r.fused.as_ref().map(layers),
                    "dispersion_curve": {"wavelength_m": site.wavelength.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), "velocity_mps": site.velocity.iter().map(|x| x.round()).collect::<Vec<_>>()},
                    "accepted_spacings_m": site.accepted.iter().map(|c| c.spacing).collect::<Vec<_>>(),
                    "coherent_share": site.coherent_share,
                    "raw_record_report": site.report,
                    "note": "Layers with resolved = false lie below the resolved depth and are extrapolations."
                }))
            }
            "get_rule" => Ok(json!({"founding_depth": self.rule.df, "soft_below": self.rule.soft, "bearing_from": self.rule.bearing, "deepest_option": self.rule.max_deeper, "unit_weight": self.rule.gamma, "use_kpt": self.use_kpt})),
            "set_rule" => {
                let f = |k: &str, v: &mut f64| {
                    if let Some(x) = a.get(k).and_then(Value::as_f64) {
                        *v = x;
                    }
                };
                f("founding_depth", &mut self.rule.df);
                f("soft_below", &mut self.rule.soft);
                f("bearing_from", &mut self.rule.bearing);
                f("deepest_option", &mut self.rule.max_deeper);
                f("unit_weight", &mut self.rule.gamma);
                if let Some(b) = a.get("use_kpt").and_then(Value::as_bool) {
                    self.use_kpt = b;
                }
                self.call("get_rule", &json!({}))
            }
            "summary" => {
                let names: Vec<String> = self.project.sites.keys().cloned().collect();
                let rows = names.iter().map(|n| self.row(n)).collect::<Result<Vec<_>, _>>()?;
                let mut counts: HashMap<String, usize> = HashMap::new();
                for r in &rows {
                    let v = r["verdict"].as_str().unwrap_or("");
                    let key = if v.starts_with("found") { "found at the planned level" } else if v.starts_with("soft at") { "found deeper" } else if v.starts_with("soft") { "soft within the resolved depth" } else { "not certifiable" };
                    *counts.entry(key.to_string()).or_default() += 1;
                }
                let uncertain: Vec<&Value> = rows.iter().filter(|r| r["confident"] == json!(false) && !r["verdict"].as_str().unwrap_or("").starts_with("not")).map(|r| &r["site"]).collect();
                Ok(json!({"project": self.project.name, "foundations": rows.len(), "verdicts": counts, "uncertain_test_again": uncertain, "rule": self.call("get_rule", &json!({}))?}))
            }
            "export_table" => {
                let p = s("path")?;
                let names: Vec<String> = self.project.sites.keys().cloned().collect();
                for n in &names {
                    self.result(n)?;
                }
                std::fs::write(&p, engine::results_csv(&self.project, &self.results, &self.model.meta.thick, &self.rule, self.use_kpt)).map_err(|e| format!("{p}: {e}"))?;
                Ok(json!({"written": p, "foundations": names.len()}))
            }
            "export_profile" => {
                let (n, p) = (s("site")?, s("path")?);
                let (use_kpt, thick) = (self.use_kpt, self.model.meta.thick.clone());
                let csv = engine::profile_csv(self.result(&n)?.best(use_kpt), &thick);
                std::fs::write(&p, csv).map_err(|e| format!("{p}: {e}"))?;
                Ok(json!({"written": p}))
            }
            "screenshot" => {
                let (n, p) = (s("site")?, s("path")?);
                if !self.project.sites.contains_key(&n) {
                    return Err(format!("no foundation named {n}"));
                }
                let tmp = std::env::temp_dir().join(format!("vsfuse_mcp_{}.json", std::process::id()));
                self.project.save(&tmp)?;
                let exe = std::env::current_exe().map_err(|e| e.to_string())?;
                let st = std::process::Command::new(exe).arg(&tmp).env("VSFUSE_SHOT", &p).env("VSFUSE_SITE", &n).env("VSFUSE_THEME", a.get("theme").and_then(Value::as_str).unwrap_or("light")).stdout(std::process::Stdio::null()).status().map_err(|e| e.to_string())?;
                let _ = std::fs::remove_file(&tmp);
                if st.success() && std::path::Path::new(&p).exists() {
                    Ok(json!({"written": p}))
                } else {
                    Err("the window could not be rendered (no display?)".into())
                }
            }
            _ => Err(format!("unknown tool {name}")),
        }
    }
}

pub fn serve(project: Option<PathBuf>) -> Result<(), String> {
    let mut st = State { model: Model::load(crate::WEIGHTS)?, project: Project::default(), rule: Rule::default(), use_kpt: true, results: HashMap::new() };
    if let Some(p) = project {
        st.project = Project::load(&p)?;
    }
    let (stdin, mut out) = (std::io::stdin(), std::io::stdout());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            let _ = writeln!(out, "{}", json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error"}}));
            continue;
        };
        let Some(id) = msg.get("id").cloned() else { continue }; // notifications need no reply
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        let reply = match method {
            "initialize" => json!({"result": {
                "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "vsfuse", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "VsFuse turns SASW surface-wave records and Kunzelstab (KPT) soundings into shear-wave velocity profiles and a founding-level decision for shallow foundations. Typical flow: process_raw (or open_project / add_curve) -> attach_kpt where a sounding exists -> list_sites or summary -> get_site for the profile and its uncertainty -> set_rule if the founding depth or limits differ -> export_table. Velocities are screening values from a neural estimator trained on simulated physics; report the 95 % range and the resolved depth with them, and report only numbers the tools returned."
            }}),
            "ping" => json!({"result": {}}),
            "tools/list" => json!({"result": {"tools": tools()}}),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match st.call(name, &args) {
                    Ok(v) => json!({"result": {"content": [{"type": "text", "text": serde_json::to_string_pretty(&v).unwrap_or_default()}], "isError": false}}),
                    Err(e) => json!({"result": {"content": [{"type": "text", "text": e}], "isError": true}}),
                }
            }
            _ => json!({"error": {"code": -32601, "message": format!("method not found: {method}")}}),
        };
        let mut full = json!({"jsonrpc": "2.0", "id": id});
        full.as_object_mut().unwrap().extend(reply.as_object().unwrap().clone());
        if writeln!(out, "{full}").is_err() || out.flush().is_err() {
            break;
        }
    }
    Ok(())
}
