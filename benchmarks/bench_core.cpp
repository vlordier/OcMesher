// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

// Performance benchmarks for OcMesher core algorithms
// Targets arm64 Apple M-series (M4) with robust timing and statistics

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Portable high-resolution clock
// ---------------------------------------------------------------------------
using hrc = std::chrono::high_resolution_clock;

static double elapsed_ms(hrc::time_point start, hrc::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ---------------------------------------------------------------------------
// Statistics helper
// ---------------------------------------------------------------------------
struct BenchStats {
    double min_ms;
    double max_ms;
    double mean_ms;
    double median_ms;
    double stddev_ms;
    int    samples;
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
    double sq   = 0.0;
    for (double t : times) sq += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(sq / s.samples);
    return s;
}

static void print_stats(const char *name, BenchStats &s) {
    std::printf("  %-40s  samples=%3d  mean=%10.3f ms  median=%10.3f ms  "
                "min=%10.3f ms  max=%10.3f ms  stddev=%8.3f ms\n",
                name, s.samples, s.mean_ms, s.median_ms, s.min_ms, s.max_ms,
                s.stddev_ms);
}

// ---------------------------------------------------------------------------
// Platform / architecture info
// ---------------------------------------------------------------------------
static void print_platform_info() {
    std::printf("=== OcMesher Performance Benchmark ===\n\n");

#if defined(__aarch64__) || defined(_M_ARM64)
    std::printf("Architecture : arm64\n");
#elif defined(__x86_64__) || defined(_M_X64)
    std::printf("Architecture : x86_64\n");
#else
    std::printf("Architecture : unknown\n");
#endif

#if defined(__APPLE__)
    std::printf("Platform     : macOS (Apple Silicon target)\n");
#elif defined(__linux__)
    std::printf("Platform     : Linux\n");
#else
    std::printf("Platform     : other\n");
#endif

#ifdef _OPENMP
    std::printf("OpenMP       : enabled  (max threads = %d)\n",
                omp_get_max_threads());
#else
    std::printf("OpenMP       : disabled\n");
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::printf("NEON SIMD    : available\n");
#else
    std::printf("NEON SIMD    : not available\n");
#endif

    std::printf("sizeof(double) = %zu, sizeof(float) = %zu\n\n",
                sizeof(double), sizeof(float));
}

// ===========================================================================
// Core data types (mirrored from ocmesher/source/core.h)
// ===========================================================================
typedef double T;
typedef float  sdfT;

struct computed_vertex {
    T c[3], l, r;
};

struct cube {
    int coords[3], L;
};
typedef cube vertex;

struct node {
    cube c;
    int  nxts[8];
};

// Convenience macros (from core.h) used in benchmarks
#define cubex(x) ((x) * (x) * (x))
#define cube_index(x, y, z, s) ((x) * (s) * (s) + (y) * (s) + (z))

// ===========================================================================
// 1. Determinant computation (tri-segment intersection kernel)
// ===========================================================================
inline T det3x3(T m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

static void bench_determinant(int iterations, int warmup) {
    const int N = 100000;
    std::vector<T> data(N * 9);
    for (int i = 0; i < N * 9; i++)
        data[i] = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;

    // Warmup
    volatile T sink = 0;
    for (int w = 0; w < warmup; w++) {
        for (int i = 0; i < N; i++) {
            T m[3][3];
            std::memcpy(m, &data[i * 9], 9 * sizeof(T));
            sink += det3x3(m);
        }
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        T acc = 0;
        for (int i = 0; i < N; i++) {
            T m[3][3];
            std::memcpy(m, &data[i * 9], 9 * sizeof(T));
            acc += det3x3(m);
        }
        auto t1 = hrc::now();
        sink += acc;  // prevent optimising away
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    print_stats("3x3 determinant (100k evals)", st);
    (void)sink;
}

// ===========================================================================
// 2. Octree expansion (node allocation + child creation)
// ===========================================================================
static void expand_octree_bench(std::vector<node> &nodes, int index) {
    for (int i = 0; i < 8; i++) {
        node *current = &nodes[index];
        current->nxts[i] = static_cast<int>(nodes.size());
        node n0;
        n0.nxts[0] = -2;  // mark leaf
        n0.c.L = current->c.L + 1;
        for (int k = 0; k < 3; k++) {
            int offset = (i >> k) & 1;
            n0.c.coords[k] = current->c.coords[k] * 2 + offset;
        }
        nodes.push_back(n0);
    }
}

static void bench_octree_expansion(int iterations, int warmup) {
    // Warmup
    for (int w = 0; w < warmup; w++) {
        std::vector<node> nodes;
        node root;
        root.nxts[0] = -2;
        std::memset(root.c.coords, 0, sizeof(root.c.coords));
        root.c.L = 0;
        nodes.push_back(root);
        std::queue<int> q;
        q.push(0);
        while (nodes.size() < 5000) {
            int idx = q.front(); q.pop();
            int base = static_cast<int>(nodes.size());
            expand_octree_bench(nodes, idx);
            for (int i = 0; i < 8; i++) q.push(base + i);
        }
    }

    const int TARGET_NODES = 50000;
    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<node> nodes;
        nodes.reserve(TARGET_NODES + 8);
        node root;
        root.nxts[0] = -2;
        std::memset(root.c.coords, 0, sizeof(root.c.coords));
        root.c.L = 0;
        nodes.push_back(root);
        std::queue<int> q;
        q.push(0);

        auto t0 = hrc::now();
        while (static_cast<int>(nodes.size()) < TARGET_NODES) {
            int idx = q.front(); q.pop();
            int base = static_cast<int>(nodes.size());
            expand_octree_bench(nodes, idx);
            for (int i = 0; i < 8; i++) q.push(base + i);
        }
        auto t1 = hrc::now();

        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    char label[128];
    std::snprintf(label, sizeof(label), "Octree expansion (%dk nodes)", TARGET_NODES / 1000);
    print_stats(label, st);
}

// ===========================================================================
// 3. Vertex enumeration (grid vertex generation per node)
// ===========================================================================
static void enumerate_vertices_bench(vertex *v, node n, int grid_level) {
    int ss = 1 << grid_level;
    for (int i = 0; i <= ss; i++)
        for (int j = 0; j <= ss; j++)
            for (int k = 0; k <= ss; k++) {
                int vid = i + (ss + 1) * j + (ss + 1) * (ss + 1) * k;
                v[vid].coords[0] = n.c.coords[0] * ss + i;
                v[vid].coords[1] = n.c.coords[1] * ss + j;
                v[vid].coords[2] = n.c.coords[2] * ss + k;
                v[vid].L = n.c.L + grid_level;
                // Simplification pass
                while (v[vid].L > 0) {
                    bool can_coarsen = true;
                    for (int p = 0; p < 3; p++) {
                        if (v[vid].coords[p] & 1) {
                            can_coarsen = false;
                            break;
                        }
                    }
                    if (!can_coarsen) break;
                    for (int p = 0; p < 3; p++) v[vid].coords[p] >>= 1;
                    v[vid].L--;
                }
            }
}

static void bench_vertex_enumeration(int iterations, int warmup) {
    const int GRID_LEVEL = 4;
    const int SS = 1 << GRID_LEVEL;
    const int VERT_COUNT = cubex(SS + 1);
    const int NUM_NODES = 200;

    std::vector<node> test_nodes(NUM_NODES);
    for (int i = 0; i < NUM_NODES; i++) {
        test_nodes[i].c.coords[0] = rand() % 64;
        test_nodes[i].c.coords[1] = rand() % 64;
        test_nodes[i].c.coords[2] = rand() % 64;
        test_nodes[i].c.L = 3;
        test_nodes[i].nxts[0] = -2 - GRID_LEVEL;
    }

    std::vector<vertex> verts(VERT_COUNT);

    // Warmup
    for (int w = 0; w < warmup; w++) {
        for (int i = 0; i < NUM_NODES; i++) {
            enumerate_vertices_bench(verts.data(), test_nodes[i], GRID_LEVEL);
        }
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        for (int i = 0; i < NUM_NODES; i++) {
            enumerate_vertices_bench(verts.data(), test_nodes[i], GRID_LEVEL);
        }
        auto t1 = hrc::now();
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    char label[128];
    std::snprintf(label, sizeof(label), "Vertex enum (%d nodes, level %d)",
                  NUM_NODES, GRID_LEVEL);
    print_stats(label, st);
}

// ===========================================================================
// 4. Projected size calculation (camera projection math)
// ===========================================================================
static void bench_projection(int iterations, int warmup) {
    const int N_CUBES = 50000;
    const int N_CAMS = 4;

    // Synthetic camera data: 12 (extrinsic) + 9 (intrinsic) + 2 (H,W) = 23
    std::vector<T> cams(N_CAMS * 23);
    for (size_t i = 0; i < cams.size(); i++)
        cams[i] = (static_cast<T>(rand()) / RAND_MAX) * 2.0 - 1.0;
    // Set H, W to reasonable values
    for (int k = 0; k < N_CAMS; k++) {
        cams[k * 23 + 21] = 1080.0;
        cams[k * 23 + 22] = 1920.0;
        cams[k * 23 + 12] = 500.0; // focal length proxy
    }

    T center[3] = {0.0, 0.0, 0.0};
    T size = 10.0;
    T pixels_per_cube = 8.0;
    T min_dist = 0.01;

    std::vector<cube> cubes(N_CUBES);
    for (int i = 0; i < N_CUBES; i++) {
        cubes[i].coords[0] = rand() % 256;
        cubes[i].coords[1] = rand() % 256;
        cubes[i].coords[2] = rand() % 256;
        cubes[i].L = 4 + (rand() % 6);
    }

    // Inline projection kernel matching core.cpp logic
    auto projected_size_k = [&](const cube &c, int k) -> T {
        T *cam = &cams[k * 23];
        T Pw[3];
        for (int j = 0; j < 3; j++)
            Pw[j] = center[j] - size / 2.0 + size * (c.coords[j] + 0.5) / (1 << c.L);

        T Pc[3];
        for (int i = 0; i < 3; i++) {
            Pc[i] = cam[i * 4 + 3];
            for (int j = 0; j < 3; j++)
                Pc[i] += Pw[j] * cam[i * 4 + j];
        }
        T r = std::sqrt(Pc[0] * Pc[0] + Pc[1] * Pc[1] + Pc[2] * Pc[2]);
        r = std::max(r, min_dist);

        T W = cam[22];
        T pix_ang = std::atan(W / 2.0 / cam[12]) * 2.0 / W;
        T ang = pix_ang * pixels_per_cube;
        return size / (1 << c.L) / r / ang;
    };

    // Warmup
    volatile T sink = 0;
    for (int w = 0; w < warmup; w++) {
        for (int i = 0; i < N_CUBES; i++) {
            T max_s = 0;
            for (int k = 0; k < N_CAMS; k++)
                max_s = std::max(max_s, projected_size_k(cubes[i], k));
            sink += max_s;
        }
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        T acc = 0;
        for (int i = 0; i < N_CUBES; i++) {
            T max_s = 0;
            for (int k = 0; k < N_CAMS; k++)
                max_s = std::max(max_s, projected_size_k(cubes[i], k));
            acc += max_s;
        }
        auto t1 = hrc::now();
        sink += acc;
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    char label[128];
    std::snprintf(label, sizeof(label), "Projection (%dk cubes, %d cams)",
                  N_CUBES / 1000, N_CAMS);
    print_stats(label, st);
    (void)sink;
}

// ===========================================================================
// 5. SDF bisection convergence (update_verts kernel)
// ===========================================================================
static void bench_sdf_bisection(int iterations, int warmup) {
    const int N_VERTS = 100000;
    const int BISECTION_ITERS = 10;

    std::vector<computed_vertex> verts(N_VERTS);
    for (int i = 0; i < N_VERTS; i++) {
        verts[i].c[0] = static_cast<T>(rand()) / RAND_MAX;
        verts[i].c[1] = static_cast<T>(rand()) / RAND_MAX;
        verts[i].c[2] = static_cast<T>(rand()) / RAND_MAX;
        verts[i].l = 0.0;
        verts[i].r = 0.5;
    }

    // Synthetic SDF data (8 corner samples + 1 centre per vertex)
    std::vector<sdfT> sdf(N_VERTS * 8);
    std::vector<sdfT> center_sdf(N_VERTS);
    for (int i = 0; i < N_VERTS * 8; i++)
        sdf[i] = static_cast<sdfT>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < N_VERTS; i++)
        center_sdf[i] = static_cast<sdfT>(rand()) / RAND_MAX * 2.0f - 1.0f;

    std::vector<T> positions(N_VERTS * 8 * 3);

    // Warmup
    for (int w = 0; w < warmup; w++) {
        for (int b = 0; b < BISECTION_ITERS; b++) {
            for (int i = 0; i < N_VERTS; i++) {
                T mid = (verts[i].l + verts[i].r) / 2.0;
                bool bipolar = false;
                for (int j = 0; j < 8; j++) {
                    if ((sdf[i * 8 + j] >= 0) != (center_sdf[i] >= 0)) {
                        bipolar = true;
                        break;
                    }
                }
                if (bipolar) verts[i].r = mid;
                else verts[i].l = mid;
            }
        }
    }

    // Reset
    for (int i = 0; i < N_VERTS; i++) {
        verts[i].l = 0.0;
        verts[i].r = 0.5;
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < N_VERTS; i++) {
            verts[i].l = 0.0;
            verts[i].r = 0.5;
        }

        auto t0 = hrc::now();
        for (int b = 0; b < BISECTION_ITERS; b++) {
            for (int i = 0; i < N_VERTS; i++) {
                T mid = (verts[i].l + verts[i].r) / 2.0;
                bool bipolar = false;
                for (int j = 0; j < 8; j++) {
                    if ((sdf[i * 8 + j] >= 0) != (center_sdf[i] >= 0)) {
                        bipolar = true;
                        break;
                    }
                }
                if (bipolar) verts[i].r = mid;
                else verts[i].l = mid;

                // Also produce output positions
                mid = (verts[i].l + verts[i].r) / 2.0;
                for (int j = 0; j < 8; j++) {
                    int cid = i * 8 + j;
                    for (int k = 0; k < 3; k++)
                        positions[cid * 3 + k] =
                            verts[i].c[k] + (((j >> k) & 1) * 2 - 1) * mid;
                }
            }
        }
        auto t1 = hrc::now();
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    char label[128];
    std::snprintf(label, sizeof(label),
                  "SDF bisection (%dk verts, %d iters)", N_VERTS / 1000,
                  BISECTION_ITERS);
    print_stats(label, st);
}

// ===========================================================================
// 6. Triangle–segment intersection throughput
// ===========================================================================
static bool tri_seg_intersect_bench(T *t1, T *t2, T *t3, T *s1, T *s2) {
    T m[3][3];
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = t3[i] - s1[i];
    }
    T det1 = det3x3(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s2[i];
        m[1][i] = t2[i] - s2[i];
        m[2][i] = t3[i] - s2[i];
    }
    T det2 = det3x3(m);
    if (!((det1 > 0 && det2 < 0) || (det1 < 0 && det2 > 0))) return false;

    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = s2[i] - s1[i];
    }
    det1 = det3x3(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t2[i] - s1[i];
        m[1][i] = t3[i] - s1[i];
    }
    det2 = det3x3(m);
    if (!((det1 > 0 && det2 > 0) || (det1 < 0 && det2 < 0))) return false;
    for (int i = 0; i < 3; i++) {
        m[0][i] = t3[i] - s1[i];
        m[1][i] = t1[i] - s1[i];
    }
    det1 = det3x3(m);
    if (!((det1 >= 0 && det2 > 0) || (det1 <= 0 && det2 < 0))) return false;
    return true;
}

static void bench_tri_seg_intersect(int iterations, int warmup) {
    const int N = 200000;
    std::vector<T> tris(N * 9);
    std::vector<T> segs(N * 6);
    for (size_t i = 0; i < tris.size(); i++)
        tris[i] = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;
    for (size_t i = 0; i < segs.size(); i++)
        segs[i] = static_cast<T>(rand()) / RAND_MAX * 2.0 - 1.0;

    volatile int sink = 0;
    for (int w = 0; w < warmup; w++) {
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            cnt += tri_seg_intersect_bench(&tris[i * 9], &tris[i * 9 + 3],
                                           &tris[i * 9 + 6], &segs[i * 6],
                                           &segs[i * 6 + 3]);
        }
        sink += cnt;
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            cnt += tri_seg_intersect_bench(&tris[i * 9], &tris[i * 9 + 3],
                                           &tris[i * 9 + 6], &segs[i * 6],
                                           &segs[i * 6 + 3]);
        }
        auto t1 = hrc::now();
        sink += cnt;
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    print_stats("Tri-seg intersection (200k tests)", st);
    (void)sink;
}

// ===========================================================================
// 7. Map/set operations (key_cube insertion & lookup)
// ===========================================================================
typedef std::pair<int, std::pair<int, std::pair<int, int>>> key_cube_t;

static void bench_map_operations(int iterations, int warmup) {
    const int N = 100000;

    std::vector<key_cube_t> keys(N);
    for (int i = 0; i < N; i++) {
        keys[i] = std::make_pair(
            rand() % 1024,
            std::make_pair(rand() % 1024,
                           std::make_pair(rand() % 1024, rand() % 12)));
    }

    // Warmup
    for (int w = 0; w < warmup; w++) {
        std::map<key_cube_t, int> m;
        for (int i = 0; i < N; i++) m[keys[i]] = i;
    }

    // Insert benchmark
    std::vector<double> ins_times;
    for (int iter = 0; iter < iterations; iter++) {
        std::map<key_cube_t, int> m;
        auto t0 = hrc::now();
        for (int i = 0; i < N; i++) m[keys[i]] = i;
        auto t1 = hrc::now();
        ins_times.push_back(elapsed_ms(t0, t1));
    }

    // Lookup benchmark
    std::vector<double> lookup_times;
    std::map<key_cube_t, int> lookup_map;
    for (int i = 0; i < N; i++) lookup_map[keys[i]] = i;
    volatile int sink = 0;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        int acc = 0;
        for (int i = 0; i < N; i++) {
            auto it = lookup_map.find(keys[i]);
            if (it != lookup_map.end()) acc += it->second;
        }
        auto t1 = hrc::now();
        sink += acc;
        lookup_times.push_back(elapsed_ms(t0, t1));
    }

    auto si = compute_stats(ins_times);
    auto sl = compute_stats(lookup_times);
    print_stats("Map insert (100k key_cube)", si);
    print_stats("Map lookup (100k key_cube)", sl);
    (void)sink;
}

// ===========================================================================
// 8. Priority queue operations (coarse octree scheduling)
// ===========================================================================
static void bench_priority_queue(int iterations, int warmup) {
    const int N = 100000;

    std::vector<std::pair<T, int>> data(N);
    for (int i = 0; i < N; i++)
        data[i] = std::make_pair(static_cast<T>(rand()) / RAND_MAX, i);

    // Warmup
    for (int w = 0; w < warmup; w++) {
        std::priority_queue<std::pair<T, int>> pq;
        for (int i = 0; i < N; i++) pq.push(data[i]);
        while (!pq.empty()) pq.pop();
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        std::priority_queue<std::pair<T, int>> pq;
        for (int i = 0; i < N; i++) pq.push(data[i]);
        while (!pq.empty()) pq.pop();
        auto t1 = hrc::now();
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    print_stats("Priority queue push+pop (100k)", st);
}

// ===========================================================================
// 9. Memory bandwidth (sequential read/write – important for M-series
//    unified memory architecture)
// ===========================================================================
static void bench_memory_bandwidth(int iterations, int warmup) {
    const size_t SIZE = 64 * 1024 * 1024;  // 64 MB
    const size_t N = SIZE / sizeof(T);

    std::vector<T> src(N);
    std::vector<T> dst(N);
    for (size_t i = 0; i < N; i++) src[i] = static_cast<T>(i);

    // Warmup
    for (int w = 0; w < warmup; w++) {
        std::memcpy(dst.data(), src.data(), SIZE);
    }

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        std::memcpy(dst.data(), src.data(), SIZE);
        auto t1 = hrc::now();
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    double gbps = (SIZE / 1e9) / (st.mean_ms / 1e3);
    char label[128];
    std::snprintf(label, sizeof(label),
                  "Memcpy 64 MB (%.1f GB/s mean)", gbps);
    print_stats(label, st);
}

// ===========================================================================
// 10. OpenMP parallel scaling (visibility-filter–like workload)
// ===========================================================================
static void bench_omp_scaling(int iterations, int warmup) {
#ifndef _OPENMP
    std::printf("  %-40s  SKIPPED (OpenMP not available)\n",
                "OpenMP parallel scaling");
    return;
#else
    const int N = 500000;
    std::vector<T> xs(N), ys(N), zs(N);
    std::vector<int> results(N);
    for (int i = 0; i < N; i++) {
        xs[i] = static_cast<T>(rand()) / RAND_MAX * 1920.0;
        ys[i] = static_cast<T>(rand()) / RAND_MAX * 1080.0;
        zs[i] = static_cast<T>(rand()) / RAND_MAX * 100.0;
    }

    auto run_workload = [&](int num_threads) {
        omp_set_num_threads(num_threads);
        int total = 0;
        #pragma omp parallel for reduction(+:total)
        for (int i = 0; i < N; i++) {
            // Simulate projection + visibility check
            T px = xs[i] / zs[i];
            T py = ys[i] / zs[i];
            if (px >= 0 && px < 1920 && py >= 0 && py < 1080) {
                results[i] = 1;
                total++;
            } else {
                results[i] = 0;
            }
        }
        return total;
    };

    // Warmup
    for (int w = 0; w < warmup; w++) {
        run_workload(omp_get_max_threads());
    }

    int max_threads = omp_get_max_threads();
    for (int nt = 1; nt <= max_threads; nt *= 2) {
        std::vector<double> times;
        for (int iter = 0; iter < iterations; iter++) {
            auto t0 = hrc::now();
            run_workload(nt);
            auto t1 = hrc::now();
            times.push_back(elapsed_ms(t0, t1));
        }
        auto st = compute_stats(times);
        char label[128];
        std::snprintf(label, sizeof(label),
                      "OMP visibility (%d threads, 500k)", nt);
        print_stats(label, st);
    }
    // Also test max threads if not power of 2
    if ((max_threads & (max_threads - 1)) != 0) {
        std::vector<double> times;
        for (int iter = 0; iter < iterations; iter++) {
            auto t0 = hrc::now();
            run_workload(max_threads);
            auto t1 = hrc::now();
            times.push_back(elapsed_ms(t0, t1));
        }
        auto st = compute_stats(times);
        char label[128];
        std::snprintf(label, sizeof(label),
                      "OMP visibility (%d threads, 500k)", max_threads);
        print_stats(label, st);
    }
    // Restore default
    omp_set_num_threads(max_threads);
#endif
}

// ===========================================================================
// 11. Coordinate computation throughput
// ===========================================================================
static void bench_compute_coords(int iterations, int warmup) {
    const int N = 500000;
    T center[3] = {0.0, 0.0, 0.0};
    T size = 10.0;

    std::vector<int> icoords(N * 3);
    std::vector<int> levels(N);
    std::vector<T> out(N * 3);
    for (int i = 0; i < N; i++) {
        icoords[i * 3 + 0] = rand() % 1024;
        icoords[i * 3 + 1] = rand() % 1024;
        icoords[i * 3 + 2] = rand() % 1024;
        levels[i] = 4 + (rand() % 8);
    }

    auto compute = [&]() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < 3; j++) {
                out[i * 3 + j] = center[j] - size / 2.0 +
                                 size * icoords[i * 3 + j] /
                                     static_cast<T>(1 << levels[i]);
            }
        }
    };

    // Warmup
    for (int w = 0; w < warmup; w++) compute();

    std::vector<double> times;
    for (int iter = 0; iter < iterations; iter++) {
        auto t0 = hrc::now();
        compute();
        auto t1 = hrc::now();
        times.push_back(elapsed_ms(t0, t1));
    }

    auto st = compute_stats(times);
    print_stats("Coordinate compute (500k verts)", st);
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char *argv[]) {
    int iterations = 20;
    int warmup = 3;

    // Simple argument parsing
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

    print_platform_info();
    std::printf("Configuration: iterations=%d  warmup=%d\n\n", iterations,
                warmup);

    srand(42);  // Reproducible results

    std::printf("--- Micro-benchmarks ---\n");
    bench_determinant(iterations, warmup);
    bench_tri_seg_intersect(iterations, warmup);
    bench_compute_coords(iterations, warmup);
    bench_sdf_bisection(iterations, warmup);

    std::printf("\n--- Data structure benchmarks ---\n");
    bench_map_operations(iterations, warmup);
    bench_priority_queue(iterations, warmup);

    std::printf("\n--- Algorithm benchmarks ---\n");
    bench_octree_expansion(iterations, warmup);
    bench_vertex_enumeration(iterations, warmup);
    bench_projection(iterations, warmup);

    std::printf("\n--- System benchmarks ---\n");
    bench_memory_bandwidth(iterations, warmup);
    bench_omp_scaling(iterations, warmup);

    std::printf("\nAll benchmarks complete.\n");
    return 0;
}
