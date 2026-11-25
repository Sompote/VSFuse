//! The window: foundations on the left, the dispersion curve and the Vs posterior on the
//! canvas, the founding-level decision and its rule on the right.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};

use crate::chrome::{self, Plot};
use crate::data::{self, admissible, vs_kpt, Project, Site};
use crate::decision::{evaluate, Evaluation, Rule, Verdict};
use crate::engine::{self, SiteResult};
use crate::model::{Model, Posterior};
use crate::theme::{self, medium_font, Tokens};

pub struct App {
    model: Arc<Model>,
    project: Project,
    path: Option<PathBuf>,
    results: HashMap<String, SiteResult>,
    rx: Option<Receiver<(u64, String, SiteResult)>>,
    raw_rx: Option<Receiver<Option<(String, Result<Site, String>)>>>,
    raw_left: usize,
    generation: u64,
    selected: Option<String>,
    filter: String,
    rule: Rule,
    use_kpt: bool,
    corrected: bool,
    deep: bool,
    dark: bool,
    message: String,
    shot: Option<PathBuf>,
    frames: u32,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> Result<App, String> {
        theme::install_fonts(&cc.egui_ctx);
        let dark = std::env::var("VSFUSE_THEME").map(|v| v != "light").unwrap_or(true);
        theme::apply(&cc.egui_ctx, if dark { Tokens::DARK } else { Tokens::LIGHT });
        let model = Arc::new(Model::load(crate::WEIGHTS)?);
        let (project, message) = match &path {
            Some(p) => (Project::load(p)?, format!("opened {}", p.display())),
            None => (Project::parse(crate::SAMPLE)?, "example project: three foundations".to_string()),
        };
        let mut app = App {
            model,
            selected: std::env::var("VSFUSE_SITE").ok().filter(|s| project.sites.contains_key(s)).or_else(|| project.sites.keys().next().cloned()),
            project,
            path,
            results: HashMap::new(),
            rx: None,
            raw_rx: None,
            raw_left: 0,
            generation: 0,
            filter: String::new(),
            rule: Rule::default(),
            use_kpt: true,
            corrected: true,
            deep: false,
            dark,
            message,
            shot: std::env::var("VSFUSE_SHOT").ok().map(PathBuf::from),
            frames: 0,
        };
        // VSFUSE_RAW=folder starts from the raw records of that folder
        if let Some(dir) = std::env::var("VSFUSE_RAW").ok().map(PathBuf::from) {
            app.project = Project { name: dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), ..Default::default() };
            app.selected = None;
            let dirs = data::raw_sites(&dir);
            app.start_raw(&cc.egui_ctx, dirs);
        }
        app.recompute(&cc.egui_ctx);
        Ok(app)
    }

    /// Runs the estimator on every foundation in a worker thread.
    fn recompute(&mut self, ctx: &egui::Context) {
        self.generation += 1;
        self.results.clear();
        let (tx, rx) = channel();
        self.rx = Some(rx);
        let (model, sites, generation, corrected, ctx) = (self.model.clone(), self.project.sites.clone(), self.generation, self.corrected, ctx.clone());
        std::thread::spawn(move || {
            for (name, site) in sites {
                if tx.send((generation, name, engine::run_site(&model, &site, corrected))).is_err() {
                    return;
                }
                ctx.request_repaint();
            }
        });
    }

    /// Processes the raw SASW records of a site folder, or of every site folder inside it.
    fn import_raw(&mut self, ctx: &egui::Context) {
        let Some(dir) = rfd::FileDialog::new().set_title("Folder of one site (records like H4mF_HM.trf) or of many sites").pick_folder() else { return };
        let dirs = data::raw_sites(&dir);
        if dirs.is_empty() {
            self.message = format!("{}: no SASW records found (folders like H4mF_HM.trf with Run*/ASCII exports)", dir.display());
            return;
        }
        self.start_raw(ctx, dirs);
    }

    fn start_raw(&mut self, ctx: &egui::Context, dirs: Vec<PathBuf>) {
        let (tx, rx) = channel();
        self.raw_rx = Some(rx);
        self.raw_left = dirs.len();
        self.message = format!("processing the raw records of {} site(s)", dirs.len());
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for d in dirs {
                let name = d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let _ = tx.send(Some((name, Site::from_raw(&d))));
                ctx.request_repaint();
            }
            let _ = tx.send(None);
            ctx.request_repaint();
        });
    }

    fn poll_raw(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.raw_rx else { return };
        let (mut done, mut added, mut failed) = (false, 0, Vec::new());
        while let Ok(item) = rx.try_recv() {
            match item {
                Some((name, Ok(site))) => {
                    self.selected = Some(name.clone());
                    self.project.sites.insert(name, site);
                    self.raw_left = self.raw_left.saturating_sub(1);
                    added += 1;
                }
                Some((name, Err(e))) => {
                    self.raw_left = self.raw_left.saturating_sub(1);
                    failed.push(format!("{name}: {e}"));
                }
                None => done = true,
            }
        }
        if let Some(f) = failed.last() {
            self.message = f.clone();
        }
        if done {
            self.raw_rx = None;
            if failed.is_empty() {
                self.message = "raw records processed: dispersion curves and profiles are ready".into();
            }
        }
        if added > 0 || done {
            self.recompute(ctx);
        }
    }

    fn poll(&mut self) {
        if let Some(rx) = &self.rx {
            while let Ok((g, name, r)) = rx.try_recv() {
                if g == self.generation {
                    self.results.insert(name, r);
                }
            }
        }
    }

    fn eval(&self, name: &str) -> Option<Evaluation> {
        let (site, r) = (self.project.sites.get(name)?, self.results.get(name)?);
        Some(evaluate(r.best(self.use_kpt), &self.model.meta.thick, site.z_res, &self.rule))
    }

    fn verdict_colour(t: &Tokens, v: &Verdict) -> Color32 {
        match v {
            Verdict::Found => t.ok,
            Verdict::Deeper(_) => t.accent_text,
            Verdict::Soft => t.warn,
            Verdict::NotCertifiable => t.text_faint,
        }
    }

    fn open_project(&mut self, ctx: &egui::Context) {
        if let Some(p) = rfd::FileDialog::new().add_filter("VsFuse project", &["json"]).pick_file() {
            match Project::load(&p) {
                Ok(project) => {
                    self.selected = project.sites.keys().next().cloned();
                    self.project = project;
                    self.message = format!("opened {}", p.display());
                    self.path = Some(p);
                    self.recompute(ctx);
                }
                Err(e) => self.message = e,
            }
        }
    }

    fn import_curve(&mut self, ctx: &egui::Context) {
        let Some(p) = rfd::FileDialog::new().add_filter("dispersion curve", &["csv", "txt", "dat"]).pick_file() else { return };
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| data::curve_from_text(&t)) {
            Ok(site) => {
                let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "new site".into());
                self.project.sites.insert(name.clone(), site);
                self.message = format!("curve {name} added: {} points", self.project.sites[&name].wavelength.len());
                self.selected = Some(name);
                self.recompute(ctx);
            }
            Err(e) => self.message = format!("{}: {e}", p.display()),
        }
    }

    fn import_kpt(&mut self, ctx: &egui::Context) {
        let Some(name) = self.selected.clone() else { return };
        let Some(p) = rfd::FileDialog::new().add_filter("KPT sounding", &["csv", "txt", "dat"]).pick_file() else { return };
        match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| data::kpt_from_text(&t)) {
            Ok(rows) => {
                self.message = format!("sounding of {} increments attached to {name}", rows.len());
                if let Some(s) = self.project.sites.get_mut(&name) {
                    s.kpt = rows;
                    s.normalise();
                }
                self.recompute(ctx);
            }
            Err(e) => self.message = format!("{}: {e}", p.display()),
        }
    }

    fn export(&mut self) {
        let Some(p) = rfd::FileDialog::new().set_file_name("founding_level.csv").save_file() else { return };
        let csv = engine::results_csv(&self.project, &self.results, &self.model.meta.thick, &self.rule, self.use_kpt);
        self.message = match std::fs::write(&p, csv) {
            Ok(()) => format!("table of {} foundations written to {}", self.results.len(), p.display()),
            Err(e) => format!("{}: {e}", p.display()),
        };
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").frame(chrome::dock_frame(ctx)).show(ctx, |ui| {
            let t = chrome::tokens(ui);
            ui.horizontal(|ui| {
                let icon = chrome::app_icon(ctx);
                ui.add(egui::Image::new((icon.id(), Vec2::splat(22.0))));
                ui.label(egui::RichText::new("VsFuse").font(medium_font(15.0)));
                ui.label(egui::RichText::new(if self.project.name.is_empty() { "untitled project" } else { &self.project.name }).color(t.text_dim));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(if self.dark { "Light" } else { "Dark" }).on_hover_text("Switch the theme").clicked() {
                        self.dark = !self.dark;
                        theme::apply(ctx, if self.dark { Tokens::DARK } else { Tokens::LIGHT });
                    }
                    if ui.add(chrome::primary_button("Export table")).on_hover_text("Founding-level table of every foundation, as CSV").clicked() {
                        self.export();
                    }
                    if ui.button("Save project").clicked() {
                        if let Some(p) = rfd::FileDialog::new().set_file_name("project.json").save_file() {
                            self.message = match self.project.save(&p) {
                                Ok(()) => format!("project saved to {}", p.display()),
                                Err(e) => e,
                            };
                        }
                    }
                    if ui.add_enabled(self.selected.is_some(), egui::Button::new("Attach KPT")).on_hover_text("Text file of: mid-depth (m), N20. It is attached to the selected foundation").clicked() {
                        self.import_kpt(ctx);
                    }
                    if ui.button("Add curve").on_hover_text("Text file of: wavelength (m), phase velocity (m/s), relative uncertainty (optional)").clicked() {
                        self.import_curve(ctx);
                    }
                    if ui.button("Open project").clicked() {
                        self.open_project(ctx);
                    }
                    if ui.add_enabled(self.raw_rx.is_none(), chrome::primary_button("Import raw SASW")).on_hover_text("Pick the folder of one site (records like H4mF_HM.trf) or a folder of many sites. The records are unwrapped across spacings, checked for consistency and merged into a dispersion curve, and the profile follows").clicked() {
                        self.import_raw(ctx);
                    }
                    if ui.button("New").on_hover_text("Start an empty project").clicked() {
                        self.project = Project::default();
                        self.results.clear();
                        self.selected = None;
                        self.path = None;
                        self.message = "empty project: import raw SASW records or add a curve".into();
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        let t = Tokens::get(ctx);
        let frame = egui::Frame::NONE.fill(t.dock).inner_margin(egui::Margin::symmetric(10, 4)).stroke(Stroke::new(1.0_f32, t.separator));
        egui::TopBottomPanel::bottom("status").exact_height(28.0).frame(frame).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let (done, all) = (self.results.len(), self.project.sites.len());
                if self.raw_rx.is_some() {
                    chrome::status_dot(ui, t.warn, format!("processing raw records, {} site(s) left", self.raw_left));
                } else if done < all {
                    chrome::status_dot(ui, t.warn, format!("estimating {done} of {all}"));
                } else {
                    let ms = self.results.values().map(|r| r.ms).sum::<f64>() / done.max(1) as f64;
                    chrome::status_dot(ui, t.ok, format!("{all} foundations, {ms:.1} ms per estimate"));
                }
                ui.label(egui::RichText::new(format!("network: {:.1} M parameters, {} layers", self.model.params as f64 / 1e6, self.model.meta.nout)).size(12.0).color(t.text_faint));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.message).size(12.0).color(t.text_dim));
                });
            });
        });
    }

    fn site_list(&mut self, ctx: &egui::Context) {
        let t = Tokens::get(ctx);
        let edge = Stroke::new(1.0_f32, t.separator);
        egui::SidePanel::left("sites").resizable(true).default_width(292.0).min_width(240.0).frame(egui::Frame::NONE.fill(t.chrome).stroke(edge).inner_margin(12)).show(ctx, |ui| {
            let mut counts = [0usize; 4];
            for name in self.project.sites.keys() {
                if let Some(e) = self.eval(name) {
                    counts[match e.verdict {
                        Verdict::Found => 0,
                        Verdict::Deeper(_) => 1,
                        Verdict::Soft => 2,
                        Verdict::NotCertifiable => 3,
                    }] += 1;
                }
            }
            chrome::header(ui, "Foundations", &self.project.sites.len().to_string());
            chrome::card(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(5.0, 5.0);
                    chrome::chip(ui, &format!("{} found", counts[0]), t.ok);
                    chrome::chip(ui, &format!("{} deeper", counts[1]), t.accent_text);
                    chrome::chip(ui, &format!("{} soft", counts[2]), t.warn);
                    chrome::chip(ui, &format!("{} n/c", counts[3]), t.text_faint);
                });
                ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("Filter by name").desired_width(f32::INFINITY));
                let filter = self.filter.to_lowercase();
                let names: Vec<String> = self.project.sites.keys().filter(|n| filter.is_empty() || n.to_lowercase().contains(&filter)).cloned().collect();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, 26.0, names.len(), |ui, range| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    for name in &names[range] {
                        let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 26.0), Sense::click());
                        let selected = self.selected.as_deref() == Some(name.as_str());
                        let p = ui.painter();
                        if selected {
                            p.rect_filled(rect, 5.0, t.accent_soft);
                        } else if resp.hovered() {
                            p.rect_filled(rect, 5.0, t.hover);
                        }
                        p.text(rect.left_center() + Vec2::new(8.0, 0.0), Align2::LEFT_CENTER, name, FontId::proportional(13.0), if selected { t.accent_text } else { t.text });
                        let mut x = rect.right() - 8.0;
                        if let Some(e) = self.eval(name) {
                            let c = Self::verdict_colour(&t, &e.verdict);
                            let r = p.text(Pos2::new(x, rect.center().y), Align2::RIGHT_CENTER, e.verdict.short(), FontId::proportional(11.0), c);
                            x = r.left() - 8.0;
                            p.text(Pos2::new(x, rect.center().y), Align2::RIGHT_CENTER, format!("{:.0}", e.vs), FontId::monospace(11.0), t.text_dim);
                            x -= 34.0;
                        }
                        if self.project.sites[name].has_kpt() {
                            p.circle_filled(Pos2::new(x, rect.center().y), 3.0, t.tools.run);
                        }
                        if resp.clicked() {
                            self.selected = Some(name.clone());
                        }
                    }
                });
            });
        });
    }

    fn decision_panel(&mut self, ctx: &egui::Context) {
        let t = Tokens::get(ctx);
        let edge = Stroke::new(1.0_f32, t.separator);
        let mut changed = false;
        egui::SidePanel::right("decision").resizable(true).default_width(332.0).min_width(290.0).frame(egui::Frame::NONE.fill(t.chrome).stroke(edge).inner_margin(12)).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let name = self.selected.clone().unwrap_or_default();
                chrome::header(ui, "Founding-level decision", &name);
                chrome::card(ui, |ui| match (self.project.sites.get(&name), self.results.get(&name), self.eval(&name)) {
                    (Some(site), Some(r), Some(e)) => {
                        let c = Self::verdict_colour(&t, &e.verdict);
                        ui.label(egui::RichText::new(e.verdict.text()).font(medium_font(15.0)).color(c));
                        let firm = if e.p_firm >= 0.84 || e.p_firm <= 0.16 { ("confident", t.ok) } else { ("uncertain: test again here", t.warn) };
                        ui.horizontal(|ui| {
                            chrome::chip(ui, &format!("P(Vs \u{2265} {:.0}) = {:.2}", self.rule.soft, e.p_firm), c);
                            chrome::chip(ui, firm.0, firm.1);
                        });
                        ui.add_space(2.0);
                        chrome::hairline(ui);
                        chrome::kv(ui, &format!("Vs at {:.1} m", self.rule.df), &format!("{:.0} m/s", e.vs), None);
                        chrome::kv(ui, "95% range", &format!("{:.0} to {:.0} m/s", e.vs * (-1.96 * e.sd).exp(), e.vs * (1.96 * e.sd).exp()), None);
                        chrome::kv(ui, "Site class at that level", &e.class.to_string(), None);
                        chrome::kv(ui, "Screening pressure q_a", &format!("{:.0} kPa", e.qa), None);
                        chrome::kv(ui, "Shear modulus G0", &format!("{:.0} MPa", e.g0), None);
                        chrome::kv(ui, "Resolved depth", &format!("{:.1} m", site.z_res), (site.z_res <= self.rule.df).then_some(t.warn));
                        chrome::kv(ui, "Certifiable width", &e.bmax.map(|b| format!("{b:.1} m")).unwrap_or_else(|| "none".into()), None);
                        chrome::kv(ui, "Bearing layer", &e.bearing_depth.map(|d| format!("at {d:.1} m")).unwrap_or_else(|| "not resolved".into()), None);
                        chrome::kv(ui, "Sounding", &if r.fused.is_some() && self.use_kpt { format!("fused, {} increments", r.increments) } else if r.fused.is_some() { "available, not used".into() } else { "none".into() }, None);
                    }
                    (Some(_), None, _) => {
                        ui.label(egui::RichText::new("estimating\u{2026}").color(t.text_dim));
                    }
                    _ => {
                        ui.label(egui::RichText::new("Select a foundation, or add a dispersion curve.").color(t.text_dim));
                    }
                });
                ui.add_space(10.0);
                chrome::header(ui, "Rule", "");
                chrome::card(ui, |ui| {
                    egui::Grid::new("rule").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                        let mut row = |ui: &mut egui::Ui, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f64>, speed: f64, unit: &str| {
                            ui.label(egui::RichText::new(label).color(t.text_dim));
                            ui.add(egui::DragValue::new(v).range(range).speed(speed).suffix(unit));
                            ui.end_row();
                        };
                        row(ui, "Founding depth", &mut self.rule.df, 0.3..=8.0, 0.05, " m");
                        row(ui, "Soft below", &mut self.rule.soft, 100.0..=300.0, 1.0, " m/s");
                        row(ui, "Bearing layer from", &mut self.rule.bearing, 200.0..=800.0, 1.0, " m/s");
                        row(ui, "Deepest founding option", &mut self.rule.max_deeper, 1.0..=15.0, 0.1, " m");
                        row(ui, "Unit weight", &mut self.rule.gamma, 14.0..=24.0, 0.1, " kN/m\u{00b3}");
                    });
                });
                ui.add_space(10.0);
                chrome::header(ui, "Model inputs", "");
                chrome::card(ui, |ui| {
                    ui.checkbox(&mut self.use_kpt, "Fuse the KPT sounding where one exists");
                    changed |= ui.checkbox(&mut self.corrected, "Use the mode-corrected curve").on_hover_text("Divides the curve by the apparent / fundamental ratio of a reference inversion, where the project holds one").changed();
                    ui.checkbox(&mut self.deep, "Show the profile to 24 m");
                    if let Some(r) = self.results.get(&name) {
                        ui.label(egui::RichText::new(format!("{} of {} wavelengths and {} of {} increments in use", r.wavelengths, self.model.meta.g, if self.use_kpt { r.increments } else { 0 }, self.model.meta.k)).small().color(t.text_faint));
                    }
                });
                if let Some(site) = self.project.sites.get(&name).filter(|s| !s.report.is_empty()) {
                    ui.add_space(10.0);
                    chrome::header(ui, "Raw records", &format!("{} runs, {} spacings accepted", site.records.unwrap_or(0), site.accepted.len()));
                    chrome::card(ui, |ui| {
                        if let Some(c) = site.coherent_share {
                            ui.horizontal(|ui| {
                                chrome::chip(ui, &format!("coherent share {c:.2}"), if c < 0.30 { t.warn } else { t.ok });
                                if c < 0.30 {
                                    chrome::chip(ui, "low coherence: repeat the records", t.warn);
                                }
                            });
                        }
                        for line in &site.report {
                            let c = if line.contains("accepted") { t.text } else if line.contains("ambiguous") { t.text_faint } else { t.warn };
                            ui.label(egui::RichText::new(line).small().color(c));
                        }
                    });
                }
                if let (Some(r), true) = (self.results.get(&name), self.project.sites.contains_key(&name)) {
                    ui.add_space(10.0);
                    chrome::header(ui, "Layers", "mean and 95% range of Vs");
                    let post = r.best(self.use_kpt).clone();
                    let thick = self.model.meta.thick.clone();
                    chrome::card(ui, |ui| {
                        egui::Grid::new("layers").num_columns(3).striped(true).spacing([14.0, 3.0]).show(ui, |ui| {
                            let mut top = 0.0;
                            for i in 0..post.mean.len() {
                                let depth = match thick.get(i) {
                                    Some(h) => format!("{top:.1} to {:.1} m", top + h),
                                    None => format!("below {top:.1} m"),
                                };
                                ui.label(egui::RichText::new(depth).color(t.text_dim).small());
                                ui.label(egui::RichText::new(format!("{:.0}", post.mean[i].exp())).monospace());
                                ui.label(egui::RichText::new(format!("{:.0} to {:.0}", (post.mean[i] - 1.96 * post.sd[i]).exp(), (post.mean[i] + 1.96 * post.sd[i]).exp())).monospace().color(t.text_dim));
                                ui.end_row();
                                top += thick.get(i).copied().unwrap_or(0.0);
                            }
                        });
                        if ui.button("Save this profile").clicked() {
                            if let Some(p) = rfd::FileDialog::new().set_file_name(format!("{name}_profile.csv")).save_file() {
                                self.message = match std::fs::write(&p, engine::profile_csv(&post, &thick)) {
                                    Ok(()) => format!("profile written to {}", p.display()),
                                    Err(e) => format!("{}: {e}", p.display()),
                                };
                            }
                        }
                    });
                }
            });
        });
        if changed {
            self.recompute(ctx);
        }
    }

    fn canvas(&mut self, ctx: &egui::Context) {
        let t = Tokens::get(ctx);
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(t.canvas)).show(ctx, |ui| {
            let rect = ui.max_rect();
            let p = ui.painter_at(rect);
            chrome::canvas_background(&p, rect, &t);
            let Some((name, site)) = self.selected.as_ref().and_then(|n| self.project.sites.get(n).map(|s| (n.clone(), s))) else {
                p.text(rect.center(), Align2::CENTER_CENTER, "Open a project or add a dispersion curve", medium_font(15.0), t.text_dim);
                return;
            };
            let inner = rect.shrink(16.0);
            let split = inner.left() + inner.width() * 0.47;
            let left = Rect::from_min_max(inner.min, Pos2::new(split - 8.0, inner.bottom()));
            let right = Rect::from_min_max(Pos2::new(split + 8.0, inner.top()), inner.max);
            self.curve_plot(&p, left, &t, &name, site);
            self.profile_plot(&p, right, &t, &name, site);
        });
    }

    fn curve_plot(&self, p: &egui::Painter, sheet: Rect, t: &Tokens, name: &str, site: &Site) {
        let n = site.wavelength.len();
        let vmax = site.velocity.iter().cloned().fold(100.0, f64::max) * 1.25;
        let step = if vmax > 800.0 { 200.0 } else if vmax > 400.0 { 100.0 } else { 50.0 };
        let yt: Vec<f64> = (0..=(vmax / step) as usize).map(|i| i as f64 * step).collect();
        let plot = Plot::frame(p, sheet, t, &format!("Dispersion curve, {name}"), "wavelength (m)", "phase velocity (m/s)", (0.3, 80.0), (0.0, vmax), true, false, &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], &yt);
        let pc = p.with_clip_rect(plot.area);
        // the wavelengths the network reads: ticks along the bottom, bright where measured
        let inp = site.inputs(&self.model.meta, self.corrected, false);
        for (j, lam) in self.model.meta.lam.iter().enumerate() {
            let s = plot.to_screen(*lam, 0.0);
            pc.line_segment([s, s - Vec2::new(0.0, if inp.mask[j] { 9.0 } else { 4.0 })], Stroke::new(1.5_f32, if inp.mask[j] { t.accent } else { t.field_border }));
        }
        let corrected = self.corrected && site.mode_ratio.len() == n;
        // accepted points of each spacing, when the curve came from raw records
        let hues = [t.tools.soil, t.tools.water, t.tools.mesh, t.tools.run, t.tools.view, t.tools.fixity, t.tools.geogrid, t.tools.spring];
        for (k, a) in site.accepted.iter().enumerate() {
            let c = hues[k % hues.len()];
            for (w, v) in a.wavelength.iter().zip(&a.velocity) {
                pc.circle_filled(plot.to_screen(*w, *v), 1.6, c.gamma_multiply(0.75));
            }
            p.text(Pos2::new(plot.area.left() + 8.0, plot.area.top() + 56.0 + 14.0 * k as f32), Align2::LEFT_TOP, format!("\u{25cf} d = {} m", a.spacing), FontId::proportional(11.0), c);
        }
        if !site.accepted.is_empty() {
            let pts: Vec<Pos2> = (0..n).map(|i| plot.to_screen(site.wavelength[i], site.velocity[i])).collect();
            pc.add(egui::Shape::line(pts, Stroke::new(1.8_f32, t.accent)));
        }
        for i in 0..n {
            let (lam, v, s) = (site.wavelength[i], site.velocity[i], site.sigma_rel.get(i).copied().unwrap_or(0.12));
            pc.line_segment([plot.to_screen(lam, v * (1.0 - s)), plot.to_screen(lam, v * (1.0 + s))], Stroke::new(1.0_f32, t.accent.gamma_multiply(0.35)));
        }
        for i in 0..n {
            pc.circle_filled(plot.to_screen(site.wavelength[i], site.velocity[i]), 2.2, if corrected { t.text_faint } else { t.accent });
        }
        if corrected {
            let pts: Vec<Pos2> = (0..n).map(|i| plot.to_screen(site.wavelength[i], site.velocity[i] / site.mode_ratio[i])).collect();
            pc.add(egui::Shape::line(pts, Stroke::new(1.8_f32, t.accent)));
        }
        let zres = plot.to_screen(site.z_res * 2.5, 0.0).x;
        pc.vline(zres, plot.area.y_range(), Stroke::new(1.0_f32, t.warn));
        p.text(Pos2::new(zres - 5.0, plot.area.bottom() - 14.0), Align2::RIGHT_BOTTOM, format!("longest wavelength\nresolved depth {:.1} m", site.z_res), FontId::proportional(11.0), t.warn);
        let legend = if !site.accepted.is_empty() {
            "coloured points: accepted records by receiver spacing\nblue line and points: composite curve read by the network\nblue ticks: wavelengths in use"
        } else if corrected { "grey points: measured apparent velocity\nblue line: mode-corrected curve read by the network\nblue ticks: wavelengths in use" } else { "points: measured curve read by the network\nblue ticks: wavelengths in use" };
        p.text(plot.area.left_top() + Vec2::new(8.0, 8.0), Align2::LEFT_TOP, legend, FontId::proportional(11.0), t.text_faint);
    }

    fn band(pc: &egui::Painter, plot: &Plot, post: &Posterior, thick: &[f64], zmax: f64, colour: Color32, alpha: f32, width: f32) {
        let mut top = 0.0;
        let mut line = Vec::new();
        for i in 0..post.mean.len() {
            let bot = thick.get(i).map(|h| top + h).unwrap_or(zmax.max(top + 2.0)).min(zmax);
            if top >= zmax {
                break;
            }
            let (lo, hi) = ((post.mean[i] - 1.96 * post.sd[i]).exp(), (post.mean[i] + 1.96 * post.sd[i]).exp());
            pc.rect_filled(Rect::from_two_pos(plot.to_screen(lo, top), plot.to_screen(hi, bot)), 0.0, colour.gamma_multiply(alpha));
            line.push(plot.to_screen(post.mean[i].exp(), top));
            line.push(plot.to_screen(post.mean[i].exp(), bot));
            top = bot;
        }
        pc.add(egui::Shape::line(line, Stroke::new(width, colour)));
    }

    fn profile_plot(&self, p: &egui::Painter, sheet: Rect, t: &Tokens, name: &str, site: &Site) {
        let zmax = if self.deep { 24.0 } else { 10.0 };
        let yt: Vec<f64> = (0..=(zmax / if self.deep { 4.0 } else { 2.0 }) as usize).map(|i| i as f64 * if self.deep { 4.0 } else { 2.0 }).collect();
        let plot = Plot::frame(p, sheet, t, &format!("Shear-wave velocity profile, {name}"), "Vs (m/s)", "depth (m)", (50.0, 1200.0), (0.0, zmax), true, true, &[100.0, 200.0, 400.0, 800.0], &yt);
        let pc = p.with_clip_rect(plot.area);
        let thick = &self.model.meta.thick;
        // below the resolved depth the profile is an extrapolation
        if site.z_res < zmax {
            pc.rect_filled(Rect::from_min_max(plot.to_screen(50.0, site.z_res), plot.area.max), 0.0, t.canvas.gamma_multiply(0.55));
            let y = plot.to_screen(50.0, site.z_res).y;
            pc.hline(plot.area.x_range(), y, Stroke::new(1.0_f32, t.warn));
            pc.text(Pos2::new(plot.area.right() - 6.0, y + 4.0), Align2::RIGHT_TOP, "below the resolved depth", FontId::proportional(11.0), t.warn);
        }
        for (v, label, c) in [(self.rule.soft, "soft", t.text_faint), (self.rule.bearing, "bearing", t.text_faint)] {
            let x = plot.to_screen(v, 0.0).x;
            pc.add(egui::Shape::dashed_line(&[Pos2::new(x, plot.area.top()), Pos2::new(x, plot.area.bottom())], Stroke::new(1.0_f32, c), 4.0, 4.0));
            pc.text(Pos2::new(x + 4.0, plot.area.top() + 4.0), Align2::LEFT_TOP, label, FontId::proportional(10.5), c);
        }
        let yf = plot.to_screen(50.0, self.rule.df).y;
        pc.hline(plot.area.x_range(), yf, Stroke::new(1.2_f32, t.tools.view));
        pc.text(Pos2::new(plot.area.left() + 6.0, yf - 3.0), Align2::LEFT_BOTTOM, format!("founding depth {:.1} m", self.rule.df), FontId::proportional(11.0), t.tools.view);
        let Some(r) = self.results.get(name) else {
            p.text(plot.area.center(), Align2::CENTER_CENTER, "estimating\u{2026}", medium_font(13.0), t.text_dim);
            return;
        };
        let fused = r.fused.as_ref().filter(|_| self.use_kpt);
        Self::band(&pc, &plot, &r.sasw, thick, zmax, t.accent, if fused.is_some() { 0.10 } else { 0.22 }, if fused.is_some() { 1.2 } else { 2.0 });
        if let Some(f) = fused {
            Self::band(&pc, &plot, f, thick, zmax, t.tools.view, 0.22, 2.0);
        }
        for row in &site.kpt {
            let pos = plot.to_screen(vs_kpt(row.n20.max(0.5)), row.z);
            if admissible(row) {
                pc.circle_filled(pos, 3.0, t.tools.run);
            } else {
                pc.circle_stroke(pos, 2.6, Stroke::new(1.0_f32, t.tools.run.gamma_multiply(0.6)));
            }
        }
        // legend, on a backing so it stays readable over the bands
        let mut entries = vec![("SASW only, mean and 95% band", t.accent)];
        if fused.is_some() {
            entries.push(("SASW + KPT, mean and 95% band", t.tools.view));
        }
        if !site.kpt.is_empty() {
            entries.push(("KPT (open: outside the conversion)", t.tools.run));
        }
        let back = Rect::from_min_size(Pos2::new(plot.area.right() - 236.0, plot.area.bottom() - 14.0 - 16.0 * entries.len() as f32), Vec2::new(228.0, 8.0 + 16.0 * entries.len() as f32));
        p.rect_filled(back, 5.0, t.card.gamma_multiply(0.92));
        p.rect_stroke(back, 5.0, Stroke::new(1.0_f32, t.separator), egui::StrokeKind::Inside);
        for (i, (text, c)) in entries.iter().enumerate() {
            let y = back.top() + 12.0 + 16.0 * i as f32;
            p.line_segment([Pos2::new(back.left() + 8.0, y), Pos2::new(back.left() + 24.0, y)], Stroke::new(2.0_f32, *c));
            p.text(Pos2::new(back.left() + 30.0, y), Align2::LEFT_CENTER, *text, FontId::proportional(11.0), t.text_dim);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_raw(ctx);
        self.poll();
        self.top_bar(ctx);
        self.status_bar(ctx);
        self.site_list(ctx);
        self.decision_panel(ctx);
        self.canvas(ctx);
        // VSFUSE_SHOT=file.png saves one frame once every foundation is estimated, then quits
        if let Some(path) = self.shot.clone() {
            if self.raw_rx.is_none() && !self.project.sites.is_empty() && self.results.len() == self.project.sites.len() {
                self.frames += 1;
            }
            if self.frames == 4 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            let image = ctx.input(|i| i.raw.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
            if let Some(img) = image {
                let _ = image::save_buffer(&path, img.as_raw(), img.width() as u32, img.height() as u32, image::ColorType::Rgba8);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ctx.request_repaint();
        }
    }
}
