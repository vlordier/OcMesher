// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

// ARM64 NEON SIMD benchmarks for OcMesher
// Compares scalar vs. NEON implementations of hot-path kernels
// Targeting Apple M-series (M4) with NEON intrinsics

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

using hrc = std::chrono::high_resolution_clock;

static double elapsed_ms(hrc::time_point start, hrc::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct BenchStats {
    double min_ms, max_ms, mean_ms, median_ms, stddev_ms;
    int samples;
};

static BenchStats compute_stats(std::vector<double> &times) {
    BenchStats s{};
    if (times.empty()) return s;
    std::sort(times.begin(), times.end());
    s.samples   = static_cast<int>(times.size());
    s.min_ms    = times.front();
    s.max_ms    = times.back();
    s.median_ms = times[times.size() / 2];
    double sum  = std::accumulate(times.begin(), times.end(), 0.0);
    s.mean_ms   = sum / s.samples;
    double sq   = 0;
    for (double t : times) sq += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(sq / s.samples);
    return s;
}

static void print_stats(const char *name, BenchStats &s) {
    std::printf("  %-48s  samples=%3d  mean=%10.3f ms  median=%10.3f ms  "
                "min=%10.3f ms  max=%10.3f ms  stddev=%8.3f ms\n",
                name, s.samples, s.mean_ms, s.median_ms, s.min_ms, s.max_ms,
                s.stddev_ms);
}

static void print_speedup(const char *name, double scalar_ms, double neon_ms) {
    double speedup = scalar_ms / neon_ms;
    std::printf("  %-48s  speedup = %.2fx  (scalar=%.3f ms, neon=%.3f ms)\n",
                name, speedup, scalar_ms, neon_ms);
}

// ===========================================================================
// 1. 3D coordinate transform (compute_center equivalent)
//    Scalar: 3 adds, 3 muls, 3 divs per vertex
// ===========================================================================
static void coord_transform_scalar(const int *icoords, const int *levels,
                                   double *out, int N, const double *center,
                                   double size) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 3; j++) {
            out[i * 3 + j] = center[j] - size / 2.0 +
                             size * (icoords[i * 3 + j] + 0.5) /
                                 static_cast<double>(1 << levels[i]);
        }
    }
}

#if HAS_NEON
static void coord_transform_neon(const int *icoords, const int *levels,
                                 double *out, int N, const double *center,
                                 double size) {
    // NEON processes 2 doubles at a time (float64x2_t)
    float64x2_t half_size = vdupq_n_f64(size / 2.0);
    float64x2_t vsize = vdupq_n_f64(size);
    float64x2_t vhalf = vdupq_n_f64(0.5);

    for (int i = 0; i < N; i++) {
        double denom = static_cast<double>(1 << levels[i]);
        float64x2_t vdenom = vdupq_n_f64(denom);

        // Process x,y together (2 lanes)
        float64x2_t cxy = vld1q_f64(center);  // center[0], center[1]
        int32x2_t ixy = {icoords[i * 3 + 0], icoords[i * 3 + 1]};
        float64x2_t fxy = vcvtq_f64_s64(vmovl_s32(ixy));
        fxy = vaddq_f64(fxy, vhalf);
        fxy = vmulq_f64(fxy, vsize);
        fxy = vdivq_f64(fxy, vdenom);
        fxy = vaddq_f64(vsubq_f64(cxy, half_size), fxy);
        vst1q_f64(&out[i * 3], fxy);

        // Process z scalar (only 1 value remaining)
        out[i * 3 + 2] = center[2] - size / 2.0 +
                         size * (icoords[i * 3 + 2] + 0.5) / denom;
    }
}
#endif

static void bench_coord_transform(int iterations, int warmup) {
    const int N = 500000;

    std::vector<int> icoords(N * 3);
    std::vector<int> levels(N);
    std::vector<double> out(N * 3);
    double center[3] = {0.0, 0.0, 0.0};
    double size = 10.0;

    for (int i = 0; i < N * 3; i++) icoords[i] = rand() % 1024;
    for (int i = 0; i < N; i++) levels[i] = 4 + (rand() % 8);

    // Warmup + scalar benchmark
    for (int w = 0; w < warmup; w++)
        coord_transform_scalar(icoords.data(), levels.data(), out.data(), N,
                               center, size);

    std::vector<double> scalar_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        coord_transform_scalar(icoords.data(), levels.data(), out.data(), N,
                               center, size);
        auto t1 = hrc::now();
        scalar_times.push_back(elapsed_ms(t0, t1));
    }
    auto ss = compute_stats(scalar_times);
    print_stats("Coord transform scalar (500k)", ss);

#if HAS_NEON
    for (int w = 0; w < warmup; w++)
        coord_transform_neon(icoords.data(), levels.data(), out.data(), N,
                             center, size);

    std::vector<double> neon_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        coord_transform_neon(icoords.data(), levels.data(), out.data(), N,
                             center, size);
        auto t1 = hrc::now();
        neon_times.push_back(elapsed_ms(t0, t1));
    }
    auto sn = compute_stats(neon_times);
    print_stats("Coord transform NEON (500k)", sn);
    print_speedup("Coord transform NEON vs scalar", ss.mean_ms, sn.mean_ms);
#else
    std::printf("  %-48s  SKIPPED (NEON not available)\n",
                "Coord transform NEON (500k)");
#endif
}

// ===========================================================================
// 2. 3x3 determinant (tri-seg intersection kernel)
//    Used heavily in construct_faces
// ===========================================================================
static double det3x3_scalar(const double *m) {
    // m stored row-major: m[0..2] = row0, m[3..5] = row1, m[6..8] = row2
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

#if HAS_NEON
static double det3x3_neon(const double *m) {
    // Use NEON to parallelise cofactor multiplications
    // cofactors: (e*i - f*h), (d*i - f*g), (d*h - e*g)
    float64x2_t ei = {m[4], m[3]};
    float64x2_t fh = {m[5], m[5]};
    float64x2_t ig = {m[8], m[8]};
    float64x2_t hg = {m[7], m[6]};

    float64x2_t prod1 = vmulq_f64(ei, ig);
    float64x2_t prod2 = vmulq_f64(fh, hg);
    float64x2_t cofactors_01 = vsubq_f64(prod1, prod2);

    // Third cofactor: d*h - e*g
    double cof2 = m[3] * m[7] - m[4] * m[6];

    // a * cof0 - b * cof1 + c * cof2
    double result = m[0] * vgetq_lane_f64(cofactors_01, 0) -
                    m[1] * vgetq_lane_f64(cofactors_01, 1) + m[2] * cof2;
    return result;
}
#endif

static void bench_det3x3(int iterations, int warmup) {
    const int N = 200000;
    std::vector<double> data(N * 9);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;

    volatile double sink = 0;

    // Scalar
    for (int w = 0; w < warmup; w++) {
        double acc = 0;
        for (int i = 0; i < N; i++) acc += det3x3_scalar(&data[i * 9]);
        sink += acc;
    }
    std::vector<double> scalar_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        double acc = 0;
        for (int i = 0; i < N; i++) acc += det3x3_scalar(&data[i * 9]);
        auto t1 = hrc::now();
        sink += acc;
        scalar_times.push_back(elapsed_ms(t0, t1));
    }
    auto ss = compute_stats(scalar_times);
    print_stats("det3x3 scalar (200k)", ss);

#if HAS_NEON
    for (int w = 0; w < warmup; w++) {
        double acc = 0;
        for (int i = 0; i < N; i++) acc += det3x3_neon(&data[i * 9]);
        sink += acc;
    }
    std::vector<double> neon_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        double acc = 0;
        for (int i = 0; i < N; i++) acc += det3x3_neon(&data[i * 9]);
        auto t1 = hrc::now();
        sink += acc;
        neon_times.push_back(elapsed_ms(t0, t1));
    }
    auto sn = compute_stats(neon_times);
    print_stats("det3x3 NEON (200k)", sn);
    print_speedup("det3x3 NEON vs scalar", ss.mean_ms, sn.mean_ms);
#else
    std::printf("  %-48s  SKIPPED (NEON not available)\n",
                "det3x3 NEON (200k)");
#endif
    (void)sink;
}

// ===========================================================================
// 3. SDF sign comparison (bipolar detection – inner loop of update_verts)
//    Checks 8 corner signs against a centre sign
// ===========================================================================
static int sdf_bipolar_scalar(const float *sdf, const float *center,
                              int N_VERTS) {
    int count = 0;
    for (int i = 0; i < N_VERTS; i++) {
        bool bipolar = false;
        bool cs = (center[i] >= 0);
        for (int j = 0; j < 8; j++) {
            if ((sdf[i * 8 + j] >= 0) != cs) {
                bipolar = true;
                break;
            }
        }
        if (bipolar) count++;
    }
    return count;
}

#if HAS_NEON
static int sdf_bipolar_neon(const float *sdf, const float *center,
                            int N_VERTS) {
    int count = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);

    for (int i = 0; i < N_VERTS; i++) {
        // Load center sign
        float cs = center[i];
        uint32_t center_sign = (cs >= 0.0f) ? 0xFFFFFFFF : 0;
        uint32x4_t vcenter_sign = vdupq_n_u32(center_sign);

        // Load 8 SDF values in two groups of 4
        float32x4_t s0 = vld1q_f32(&sdf[i * 8]);
        float32x4_t s1 = vld1q_f32(&sdf[i * 8 + 4]);

        // Compare >= 0
        uint32x4_t sign0 = vcgeq_f32(s0, vzero);
        uint32x4_t sign1 = vcgeq_f32(s1, vzero);

        // XOR with center sign to find mismatches
        uint32x4_t diff0 = veorq_u32(sign0, vcenter_sign);
        uint32x4_t diff1 = veorq_u32(sign1, vcenter_sign);

        // OR all lanes together
        uint32x4_t any = vorrq_u32(diff0, diff1);
        uint32_t result = vgetq_lane_u32(any, 0) | vgetq_lane_u32(any, 1) |
                          vgetq_lane_u32(any, 2) | vgetq_lane_u32(any, 3);

        if (result) count++;
    }
    return count;
}
#endif

static void bench_sdf_bipolar(int iterations, int warmup) {
    const int N = 500000;
    std::vector<float> sdf(N * 8);
    std::vector<float> center(N);
    for (size_t i = 0; i < sdf.size(); i++)
        sdf[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < N; i++)
        center[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    volatile int sink = 0;

    // Scalar
    for (int w = 0; w < warmup; w++)
        sink += sdf_bipolar_scalar(sdf.data(), center.data(), N);

    std::vector<double> scalar_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        sink += sdf_bipolar_scalar(sdf.data(), center.data(), N);
        auto t1 = hrc::now();
        scalar_times.push_back(elapsed_ms(t0, t1));
    }
    auto ss = compute_stats(scalar_times);
    print_stats("SDF bipolar scalar (500k)", ss);

#if HAS_NEON
    for (int w = 0; w < warmup; w++)
        sink += sdf_bipolar_neon(sdf.data(), center.data(), N);

    std::vector<double> neon_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        sink += sdf_bipolar_neon(sdf.data(), center.data(), N);
        auto t1 = hrc::now();
        neon_times.push_back(elapsed_ms(t0, t1));
    }
    auto sn = compute_stats(neon_times);
    print_stats("SDF bipolar NEON (500k)", sn);
    print_speedup("SDF bipolar NEON vs scalar", ss.mean_ms, sn.mean_ms);
#else
    std::printf("  %-48s  SKIPPED (NEON not available)\n",
                "SDF bipolar NEON (500k)");
#endif
    (void)sink;
}

// ===========================================================================
// 4. Vertex position generation (update_verts output kernel)
//    For each vertex, compute 8 corner positions from center + half-extent
// ===========================================================================
static void vertex_pos_scalar(const double *centers, const double *mids,
                              double *out, int N) {
    for (int i = 0; i < N; i++) {
        double mid = mids[i];
        for (int j = 0; j < 8; j++) {
            int cid = i * 8 + j;
            for (int k = 0; k < 3; k++) {
                out[cid * 3 + k] =
                    centers[i * 3 + k] + (((j >> k) & 1) * 2 - 1) * mid;
            }
        }
    }
}

#if HAS_NEON
static void vertex_pos_neon(const double *centers, const double *mids,
                            double *out, int N) {
    for (int i = 0; i < N; i++) {
        float64x2_t cxy = vld1q_f64(&centers[i * 3]);  // cx, cy
        double cz = centers[i * 3 + 2];
        float64x2_t vmid = vdupq_n_f64(mids[i]);
        float64x2_t vneg_mid = vnegq_f64(vmid);

        for (int j = 0; j < 8; j++) {
            int cid = i * 8 + j;
            // signs: -1 or +1 for each axis
            float64x2_t sx = (j & 1) ? vmid : vneg_mid;
            float64x2_t sy = (j & 2) ? vmid : vneg_mid;

            // x = cx + sx, y = cy + sy
            double ox = vgetq_lane_f64(cxy, 0) + vgetq_lane_f64(sx, 0);
            double oy = vgetq_lane_f64(cxy, 1) + vgetq_lane_f64(sy, 0);
            double oz = cz + ((j & 4) ? mids[i] : -mids[i]);

            out[cid * 3 + 0] = ox;
            out[cid * 3 + 1] = oy;
            out[cid * 3 + 2] = oz;
        }
    }
}
#endif

static void bench_vertex_pos(int iterations, int warmup) {
    const int N = 200000;
    std::vector<double> centers(N * 3);
    std::vector<double> mids(N);
    std::vector<double> out(N * 8 * 3);

    for (size_t i = 0; i < centers.size(); i++)
        centers[i] = static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < N; i++)
        mids[i] = static_cast<double>(rand()) / RAND_MAX * 0.5;

    // Scalar
    for (int w = 0; w < warmup; w++)
        vertex_pos_scalar(centers.data(), mids.data(), out.data(), N);

    std::vector<double> scalar_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        vertex_pos_scalar(centers.data(), mids.data(), out.data(), N);
        auto t1 = hrc::now();
        scalar_times.push_back(elapsed_ms(t0, t1));
    }
    auto ss = compute_stats(scalar_times);
    print_stats("Vertex pos scalar (200k)", ss);

#if HAS_NEON
    for (int w = 0; w < warmup; w++)
        vertex_pos_neon(centers.data(), mids.data(), out.data(), N);

    std::vector<double> neon_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        vertex_pos_neon(centers.data(), mids.data(), out.data(), N);
        auto t1 = hrc::now();
        neon_times.push_back(elapsed_ms(t0, t1));
    }
    auto sn = compute_stats(neon_times);
    print_stats("Vertex pos NEON (200k)", sn);
    print_speedup("Vertex pos NEON vs scalar", ss.mean_ms, sn.mean_ms);
#else
    std::printf("  %-48s  SKIPPED (NEON not available)\n",
                "Vertex pos NEON (200k)");
#endif
}

// ===========================================================================
// 5. Dot product throughput (3D – used in projection)
// ===========================================================================
static double dot3_scalar(const double *a, const double *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

#if HAS_NEON
static double dot3_neon(const double *a, const double *b) {
    float64x2_t va01 = vld1q_f64(a);
    float64x2_t vb01 = vld1q_f64(b);
    float64x2_t prod = vmulq_f64(va01, vb01);
    double sum01 = vaddvq_f64(prod);
    return sum01 + a[2] * b[2];
}
#endif

static void bench_dot3(int iterations, int warmup) {
    const int N = 1000000;
    std::vector<double> a(N * 3), b(N * 3);
    for (size_t i = 0; i < a.size(); i++) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    volatile double sink = 0;

    // Scalar
    for (int w = 0; w < warmup; w++) {
        double acc = 0;
        for (int i = 0; i < N; i++)
            acc += dot3_scalar(&a[i * 3], &b[i * 3]);
        sink += acc;
    }
    std::vector<double> scalar_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        double acc = 0;
        for (int i = 0; i < N; i++)
            acc += dot3_scalar(&a[i * 3], &b[i * 3]);
        auto t1 = hrc::now();
        sink += acc;
        scalar_times.push_back(elapsed_ms(t0, t1));
    }
    auto ss = compute_stats(scalar_times);
    print_stats("3D dot product scalar (1M)", ss);

#if HAS_NEON
    for (int w = 0; w < warmup; w++) {
        double acc = 0;
        for (int i = 0; i < N; i++)
            acc += dot3_neon(&a[i * 3], &b[i * 3]);
        sink += acc;
    }
    std::vector<double> neon_times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        double acc = 0;
        for (int i = 0; i < N; i++)
            acc += dot3_neon(&a[i * 3], &b[i * 3]);
        auto t1 = hrc::now();
        sink += acc;
        neon_times.push_back(elapsed_ms(t0, t1));
    }
    auto sn = compute_stats(neon_times);
    print_stats("3D dot product NEON (1M)", sn);
    print_speedup("3D dot product NEON vs scalar", ss.mean_ms, sn.mean_ms);
#else
    std::printf("  %-48s  SKIPPED (NEON not available)\n",
                "3D dot product NEON (1M)");
#endif
    (void)sink;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char *argv[]) {
    int iterations = 20;
    int warmup = 3;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--iterations N] [--warmup N]\n", argv[0]);
            return 0;
        }
    }

    std::printf("=== OcMesher ARM64 NEON Benchmark ===\n\n");

#if defined(__aarch64__) || defined(_M_ARM64)
    std::printf("Architecture : arm64\n");
#else
    std::printf("Architecture : non-arm64 (NEON benchmarks will be skipped)\n");
#endif

#if HAS_NEON
    std::printf("NEON         : enabled\n");
#else
    std::printf("NEON         : not available\n");
#endif

#if defined(__APPLE__)
    std::printf("Platform     : macOS\n");
#elif defined(__linux__)
    std::printf("Platform     : Linux\n");
#endif

    std::printf("Configuration: iterations=%d  warmup=%d\n\n", iterations,
                warmup);

    srand(42);

    std::printf("--- Scalar vs NEON comparison ---\n\n");
    bench_dot3(iterations, warmup);
    std::printf("\n");
    bench_det3x3(iterations, warmup);
    std::printf("\n");
    bench_coord_transform(iterations, warmup);
    std::printf("\n");
    bench_sdf_bipolar(iterations, warmup);
    std::printf("\n");
    bench_vertex_pos(iterations, warmup);

    std::printf("\nAll ARM64 NEON benchmarks complete.\n");
    return 0;
}
