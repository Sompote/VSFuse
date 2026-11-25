# VsFuse

Desktop program for the neural fusion estimator of the paper *Physics-trained neural fusion of SASW
and Kunzelstab soundings for railway bridge foundations*. It reads a composite SASW dispersion curve
and an optional Kunzelstab (KPT) sounding for each foundation, returns the mean and 95 % band of the
shear-wave velocity in 20 layers, and classifies the founding level with a probability.

The network (residual MLP, 9.6 million parameters) runs in pure Rust and is built into the binary.
The interface uses the TellusHive theme (colour tokens, Inter and JetBrains Mono fonts, explorer
headers, cards and chips).

## Run

```text
cargo build --release
target/release/vsfuse                              the window, with three example foundations
target/release/vsfuse project.json                 the window, with your project
target/release/vsfuse batch project.json out.csv   founding-level table of every foundation, no window
target/release/vsfuse process RAW_DIR project.json [table.csv]   raw records to a project (and table), no window
cargo test --release                               checks the Rust forward pass against the PyTorch outputs
```

## Inputs

* **Project** (`Open project`): JSON, `{"name": "...", "sites": {"P1": {"wavelength": [...], "velocity": [...],
  "sigma_rel": [...], "mode_ratio": [...], "z_res": 6.7, "kpt": [{"z": 0.1, "n20": 12}]}}}`.
  `sigma_rel`, `mode_ratio`, `z_res` and `kpt` are optional. `sample/example_project.json` is an example.
* **Add curve**: text file with rows of wavelength (m), phase velocity (m/s) and, optionally, relative uncertainty.
* **Attach KPT**: text file with rows of mid-depth (m) and N20; it is attached to the selected foundation.

* **Import raw SASW**: the folder of one site, or a folder of many site folders. A site folder holds
  one folder per record, named with spacing and direction (`H4mF_HM.trf`, `H16mF_DW.trf`), each with
  `Run*/ASCII/Hxysv00000.txt` (transfer function) and `Cxysv00000.txt` (coherence). The program reads
  every valid run, unwraps the phase across spacings, accepts only data confirmed by another shot
  direction or spacing, merges them into a composite curve and runs the estimator. The accepted points
  of each spacing and the processing report are shown.

`src/frontend.rs` is a Rust port of the GateSASW pipeline (V5 front end). Against the Python reference
it gives identical composite curves at 144 of the 164 sites of the case study (data not included here) and curves within 1 % at
153. The other 11 differ because the reference breaks exact ties (two records with the same rank
correlation, or two spacings at the same frequency) by floating-point and sort order.
A curve made from raw records is not mode-corrected, because that correction needs an inversion.

## What the program does

* KPT conversion: N = 0.539 (N20 + 0.954), Vs = 96.9 N^0.314, for full increments with 6 <= N20 <= 100.
* Resolved depth: z_res = longest wavelength / 2.5 unless the project gives it.
* Decision at the founding depth D_f (default 1.5 m): not certifiable when z_res <= D_f, found at D_f when
  Vs >= 180 m/s, found deeper when a layer with Vs >= 360 m/s lies within 6 m and z_res, soft otherwise.
  P(Vs >= 180 m/s) comes from the stated standard deviation. All limits can be changed in the window.
* `Use the mode-corrected curve` divides the curve by `mode_ratio` where the project holds one. Without it
  the estimator runs with no inversion at all and is slightly less accurate (see the paper).

## Files

`src/model.rs` forward pass, `src/data.rs` project and text import, `src/decision.rs` founding rule,
`src/engine.rs` batch and tables, `src/app.rs` window, `src/theme.rs` and `src/chrome.rs` look,
`tools/export_weights.py` converts a trained `model.pt` to `assets/model.vsf`,
`sample/` holds three anonymised example foundations and their reference outputs.
Screenshots for checking: `VSFUSE_SHOT=out.png [VSFUSE_SITE=name] [VSFUSE_THEME=light] vsfuse`.
