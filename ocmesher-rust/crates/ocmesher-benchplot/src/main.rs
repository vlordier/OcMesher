use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use plotters::prelude::*;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
enum PlotError {
    #[error("failed to read input JSON: {0}")]
    Read(#[from] std::io::Error),
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid benchmark schema: {0}")]
    Schema(String),
}

#[derive(Debug, Clone)]
struct SceneSpeedup {
    scene: String,
    current_avg_ms: f64,
    upstream_avg_ms: f64,
    speedup: f64,
}

fn usage() {
    eprintln!("Usage: ocmesher-benchplot --input <bench_api_parity.json> --output <speedups.png>");
}

fn get_avg(scene: &Value) -> Option<f64> {
    scene.get("timing")?.get("avg_ms")?.as_f64()
}

fn collect_speedups(v: &Value) -> Result<Vec<SceneSpeedup>, PlotError> {
    let current_scenes = v
        .get("current")
        .and_then(|x| x.get("behavior"))
        .and_then(|x| x.get("scenes"))
        .and_then(Value::as_object)
        .ok_or_else(|| PlotError::Schema("missing current.behavior.scenes".to_string()))?;

    let upstream_scenes = v
        .get("upstream")
        .and_then(|x| x.get("behavior"))
        .and_then(|x| x.get("scenes"))
        .and_then(Value::as_object)
        .ok_or_else(|| PlotError::Schema("missing upstream.behavior.scenes".to_string()))?;

    let mut current_by_suffix: BTreeMap<String, f64> = BTreeMap::new();
    for (name, scene) in current_scenes {
        if let Some(suffix) = name.strip_prefix("rust_wrapper_") {
            if let Some(avg) = get_avg(scene) {
                current_by_suffix.insert(suffix.to_string(), avg);
            }
        }
    }

    let mut upstream_by_suffix: BTreeMap<String, f64> = BTreeMap::new();
    for (name, scene) in upstream_scenes {
        if let Some(suffix) = name.strip_prefix("ocmesher_") {
            if let Some(avg) = get_avg(scene) {
                upstream_by_suffix.insert(suffix.to_string(), avg);
            }
        }
    }

    let mut speedups = Vec::new();
    for (suffix, current_avg_ms) in current_by_suffix {
        if let Some(upstream_avg_ms) = upstream_by_suffix.get(&suffix) {
            speedups.push(SceneSpeedup {
                scene: suffix,
                current_avg_ms,
                upstream_avg_ms: *upstream_avg_ms,
                speedup: upstream_avg_ms / current_avg_ms,
            });
        }
    }

    if speedups.is_empty() {
        return Err(PlotError::Schema(
            "no overlapping scene metrics between rust_wrapper_* and ocmesher_*".to_string(),
        ));
    }

    Ok(speedups)
}

fn render_plot(output: &Path, speedups: &[SceneSpeedup]) -> Result<(), PlotError> {
    let root = BitMapBackend::new(output, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| PlotError::Schema(format!("plot fill failed: {e}")))?;

    let max_speedup = speedups
        .iter()
        .map(|s| s.speedup)
        .fold(1.0f64, |a, b| a.max(b));
    let y_max = (max_speedup * 1.25).max(1.4);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Rust Wrapper Speedup vs Upstream Python+C++ (avg ms)",
            ("sans-serif", 34),
        )
        .margin(25)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(0..speedups.len(), 0f64..y_max)
        .map_err(|e| PlotError::Schema(format!("chart build failed: {e}")))?;

    chart
        .configure_mesh()
        .x_labels(speedups.len())
        .x_label_formatter(&|i| {
            speedups
                .get(*i)
                .map(|s| s.scene.clone())
                .unwrap_or_else(String::new)
        })
        .y_desc("Speedup (upstream_avg_ms / current_avg_ms)")
        .x_desc("Scene")
        .light_line_style(RGBColor(220, 220, 220))
        .draw()
        .map_err(|e| PlotError::Schema(format!("mesh draw failed: {e}")))?;

    chart
        .draw_series(LineSeries::new(
            vec![(0, 1.0), (speedups.len().saturating_sub(1), 1.0)],
            &BLACK.mix(0.35),
        ))
        .map_err(|e| PlotError::Schema(format!("baseline draw failed: {e}")))?;

    for (idx, s) in speedups.iter().enumerate() {
        let color = if s.speedup >= 1.0 {
            RGBColor(54, 162, 96)
        } else {
            RGBColor(201, 67, 67)
        };
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(idx, 0.0), (idx + 1, s.speedup)],
                color.filled(),
            )))
            .map_err(|e| PlotError::Schema(format!("bar draw failed: {e}")))?;

        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.2}x", s.speedup),
                (idx, (s.speedup + 0.04).min(y_max - 0.02)),
                ("sans-serif", 18).into_font().color(&BLACK),
            )))
            .map_err(|e| PlotError::Schema(format!("label draw failed: {e}")))?;
    }

    root.present()
        .map_err(|e| PlotError::Schema(format!("present failed: {e}")))?;
    Ok(())
}

fn parse_args() -> Option<(PathBuf, PathBuf)> {
    let mut args = env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                usage();
                return None;
            }
            _ => {
                eprintln!("unknown arg: {arg}");
                usage();
                return None;
            }
        }
    }

    match (input, output) {
        (Some(i), Some(o)) => Some((i, o)),
        _ => {
            usage();
            None
        }
    }
}

fn main() {
    let Some((input, output)) = parse_args() else {
        return;
    };

    if let Err(err) = (|| -> Result<(), PlotError> {
        let raw = fs::read_to_string(&input)?;
        let json: Value = serde_json::from_str(&raw)?;
        let speedups = collect_speedups(&json)?;
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        render_plot(&output, &speedups)?;
        println!("wrote {}", output.display());
        for s in speedups {
            println!(
                "scene={} current_avg_ms={:.3} upstream_avg_ms={:.3} speedup={:.3}x",
                s.scene, s.current_avg_ms, s.upstream_avg_ms, s.speedup
            );
        }
        Ok(())
    })() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
