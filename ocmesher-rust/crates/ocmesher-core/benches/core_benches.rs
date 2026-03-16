use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ocmesher_core::{pack_cameras, validate_bounds};

fn synthetic_camera_inputs(n_cams: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut poses = vec![0.0f64; n_cams * 16];
    let mut ks = vec![0.0f64; n_cams * 9];
    let hs = vec![1080.0f64; n_cams];
    let ws = vec![1920.0f64; n_cams];

    for cam in 0..n_cams {
        let p = cam * 16;
        poses[p] = 1.0;
        poses[p + 5] = 1.0;
        poses[p + 10] = 1.0;
        poses[p + 15] = 1.0;
        poses[p + 3] = cam as f64 * 0.1;
        poses[p + 7] = cam as f64 * -0.05;
        poses[p + 11] = cam as f64 * 0.02;

        let k = cam * 9;
        ks[k] = 1200.0;
        ks[k + 4] = 1200.0;
        ks[k + 2] = 960.0;
        ks[k + 5] = 540.0;
        ks[k + 8] = 1.0;
    }

    (poses, ks, hs, ws)
}

fn bench_pack_cameras(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_cameras");
    for &n_cams in &[1usize, 8, 32, 128] {
        let (poses, ks, hs, ws) = synthetic_camera_inputs(n_cams);
        group.throughput(Throughput::Elements(n_cams as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_cams), &n_cams, |b, _| {
            b.iter(|| {
                let packed = pack_cameras(
                    black_box(&poses),
                    black_box(&ks),
                    black_box(&hs),
                    black_box(&ws),
                )
                .expect("pack_cameras should succeed");
                black_box(packed);
            });
        });
    }
    group.finish();
}

fn bench_validate_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_bounds");
    let bounds = [-128.0, 128.0, -64.0, 64.0, -32.0, 32.0];
    group.bench_function("valid_bounds", |b| {
        b.iter(|| {
            let out = validate_bounds(black_box(&bounds)).expect("bounds should be valid");
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(core, bench_pack_cameras, bench_validate_bounds);
criterion_main!(core);
