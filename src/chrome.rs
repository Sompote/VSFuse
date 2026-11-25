//! Shared widgets, drawn with the theme tokens (the TellusHive look: explorer headers on a
//! steel-blue tint, cards, chips, a dotted canvas).

use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Margin, Pos2, Rect, Sense, Stroke, Vec2};

use crate::theme::{medium_font, Tokens, RADIUS, TAB_STRIP_H};

pub fn tokens(ui: &egui::Ui) -> Tokens {
    Tokens::get(ui.ctx())
}

pub fn dock_frame(ctx: &egui::Context) -> egui::Frame {
    let t = Tokens::get(ctx);
    egui::Frame::NONE.fill(t.dock).inner_margin(Margin::symmetric(10, 6)).stroke(Stroke::new(1.0_f32, t.separator))
}

/// Title bar of an explorer; the `card` under it joins on.
pub fn header(ui: &mut egui::Ui, title: &str, context: &str) {
    let t = tokens(ui);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), TAB_STRIP_H + 2.0), Sense::hover());
    let corners = CornerRadius { nw: RADIUS, ne: RADIUS, sw: 0, se: 0 };
    let p = ui.painter();
    p.rect_filled(rect, corners, t.header);
    p.rect_stroke(rect, corners, Stroke::new(1.0_f32, t.header_edge), egui::StrokeKind::Inside);
    let title_rect = p.text(rect.left_center() + Vec2::new(10.0, 0.0), Align2::LEFT_CENTER, title, medium_font(12.5), t.text);
    if !context.is_empty() {
        p.text(Pos2::new(title_rect.right() + 7.0, rect.center().y), Align2::LEFT_CENTER, context, FontId::proportional(12.0), t.accent_text);
    }
    ui.add_space(-ui.spacing().item_spacing.y - 1.0);
}

pub fn card(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let t = tokens(ui);
    let corners = CornerRadius { nw: 0, ne: 0, sw: RADIUS, se: RADIUS };
    egui::Frame::NONE.fill(t.card).stroke(Stroke::new(1.0_f32, t.separator)).corner_radius(corners).inner_margin(10).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        add(ui);
    });
}

pub fn chip(ui: &mut egui::Ui, text: &str, colour: Color32) {
    let t = tokens(ui);
    let font = FontId::proportional(11.0);
    let w = ui.fonts(|f| f.layout_no_wrap(text.to_string(), font.clone(), colour).size().x) + 14.0;
    let (r, _) = ui.allocate_exact_size(Vec2::new(w, 20.0), Sense::hover());
    ui.painter().rect_filled(r, CornerRadius::same(10), colour.gamma_multiply(if t.dark { 0.18 } else { 0.14 }));
    ui.painter().text(r.center(), Align2::CENTER_CENTER, text, font, colour);
}

pub fn primary_button(text: &str) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.to_string()).color(Color32::WHITE).font(medium_font(13.0))).fill(Color32::from_rgb(47, 116, 208)).corner_radius(CornerRadius::same(7)).min_size(Vec2::new(0.0, 28.0))
}

pub fn status_dot(ui: &mut egui::Ui, colour: Color32, text: impl Into<String>) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
        ui.painter().circle_filled(r.center(), 3.5, colour);
        ui.label(egui::RichText::new(text.into()).size(12.0));
    });
}

pub fn hairline(ui: &mut egui::Ui) {
    let t = tokens(ui);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, t.separator);
}

/// A label on the left and a value on the right, on one row.
pub fn kv(ui: &mut egui::Ui, label: &str, value: &str, colour: Option<Color32>) {
    let t = tokens(ui);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(t.text_dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(value).monospace().color(colour.unwrap_or(t.text)));
        });
    });
}

pub fn canvas_background(painter: &egui::Painter, rect: Rect, t: &Tokens) {
    painter.rect_filled(rect, 0.0, t.canvas);
    let pitch = 22.0;
    let mut y = rect.top() + pitch / 2.0;
    while y < rect.bottom() {
        let mut x = rect.left() + pitch / 2.0;
        while x < rect.right() {
            painter.circle_filled(Pos2::new(x, y), 0.9, t.canvas_dot);
            x += pitch;
        }
        y += pitch;
    }
}

/// A plot area with a title, drawn as a sheet on the canvas. Maps data to screen with a log
/// or linear x axis and a linear y axis (y grows downward when `y_down`).
pub struct Plot {
    pub area: Rect,
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub log_x: bool,
    pub y_down: bool,
}

impl Plot {
    pub fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        let fx = if self.log_x { (x.max(1e-9).ln() - self.x.0.ln()) / (self.x.1.ln() - self.x.0.ln()) } else { (x - self.x.0) / (self.x.1 - self.x.0) };
        let fy = (y - self.y.0) / (self.y.1 - self.y.0);
        let fy = if self.y_down { fy } else { 1.0 - fy };
        Pos2::new(self.area.left() + fx as f32 * self.area.width(), self.area.top() + fy as f32 * self.area.height())
    }

    /// Sheet, grid, ticks and axis titles. Returns the plot ready for drawing data.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(p: &egui::Painter, sheet: Rect, t: &Tokens, title: &str, xl: &str, yl: &str, x: (f64, f64), y: (f64, f64), log_x: bool, y_down: bool, xt: &[f64], yt: &[f64]) -> Plot {
        let shadow = egui::epaint::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(if t.dark { 90 } else { 30 }) };
        p.add(shadow.as_shape(sheet, CornerRadius::same(8)));
        p.rect_filled(sheet, CornerRadius::same(8), t.card);
        p.rect_stroke(sheet, CornerRadius::same(8), Stroke::new(1.0_f32, t.separator), egui::StrokeKind::Inside);
        p.text(sheet.left_top() + Vec2::new(14.0, 12.0), Align2::LEFT_TOP, title, medium_font(13.0), t.text);
        let area = Rect::from_min_max(sheet.left_top() + Vec2::new(58.0, 40.0), sheet.right_bottom() - Vec2::new(18.0, 44.0));
        let plot = Plot { area, x, y, log_x, y_down };
        let grid = Stroke::new(1.0_f32, t.separator);
        let small = FontId::proportional(11.0);
        for v in xt {
            let s = plot.to_screen(*v, y.0);
            p.vline(s.x, area.y_range(), grid);
            p.text(Pos2::new(s.x, area.bottom() + 5.0), Align2::CENTER_TOP, fmt(*v), small.clone(), t.text_dim);
        }
        for v in yt {
            let s = plot.to_screen(x.0, *v);
            p.hline(area.x_range(), s.y, grid);
            p.text(Pos2::new(area.left() - 7.0, s.y), Align2::RIGHT_CENTER, fmt(*v), small.clone(), t.text_dim);
        }
        p.rect_stroke(area, 0.0, Stroke::new(1.0_f32, t.field_border), egui::StrokeKind::Inside);
        p.text(Pos2::new(area.center().x, sheet.bottom() - 10.0), Align2::CENTER_BOTTOM, xl, FontId::proportional(12.0), t.text_dim);
        let g = p.layout_no_wrap(yl.to_string(), FontId::proportional(12.0), t.text_dim);
        let pos = Pos2::new(sheet.left() + 10.0, area.center().y + g.size().x / 2.0);
        p.add(egui::epaint::TextShape::new(pos, g, t.text_dim).with_angle(-std::f32::consts::FRAC_PI_2));
        plot
    }
}

pub fn fmt(v: f64) -> String {
    if v.abs() >= 100.0 || (v - v.round()).abs() < 1e-9 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// The app's mark, as a texture for the top bar.
pub fn app_icon(ctx: &egui::Context) -> egui::TextureHandle {
    let id = egui::Id::new("app_icon_texture");
    if let Some(t) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return t;
    }
    let img = image::load_from_memory(include_bytes!("../assets/app_icon.png")).expect("app icon is a valid PNG").resize(96, 96, image::imageops::FilterType::Lanczos3).to_rgba8();
    let tex = ctx.load_texture("app_icon", egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw()), egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    tex
}
