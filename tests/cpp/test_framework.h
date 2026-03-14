// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

// Shared lightweight test framework and helpers for OcMesher C++ tests.

#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "../../ocmesher/source/core.h"

// ============================================================
// Lightweight test framework
// ============================================================

static int g_total = 0;
static int g_passed = 0;
static int g_failed = 0;

#define ASSERT_TRUE(cond)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "    FAIL: " << #cond << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                             \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b)                                                       \
    do {                                                                      \
        if ((a) != (b)) {                                                     \
            std::cerr << "    FAIL: " << #a << " == " << #b << " (" << (a)    \
                      << " vs " << (b) << ") at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_NE(a, b)                                                       \
    do {                                                                      \
        if ((a) == (b)) {                                                     \
            std::cerr << "    FAIL: " << #a << " != " << #b << " (" << (a)    \
                      << " vs " << (b) << ") at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_GT(a, b)                                                       \
    do {                                                                      \
        if (!((a) > (b))) {                                                   \
            std::cerr << "    FAIL: " << #a << " > " << #b << " (" << (a)     \
                      << " vs " << (b) << ") at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_GE(a, b)                                                       \
    do {                                                                      \
        if (!((a) >= (b))) {                                                  \
            std::cerr << "    FAIL: " << #a << " >= " << #b << " (" << (a)    \
                      << " vs " << (b) << ") at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_LT(a, b)                                                       \
    do {                                                                      \
        if (!((a) < (b))) {                                                   \
            std::cerr << "    FAIL: " << #a << " < " << #b << " (" << (a)     \
                      << " vs " << (b) << ") at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                               \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                      \
        if (std::fabs((double)(a) - (double)(b)) > (double)(tol)) {           \
            std::cerr << "    FAIL: |" << #a << " - " << #b                   \
                      << "| <= " << #tol << " (" << (a) << " vs " << (b)      \
                      << ", diff=" << std::fabs((double)(a) - (double)(b))     \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define TEST_SUITE(name)                                \
    do {                                                \
        std::cout << "\n[" << name << "]" << std::endl; \
    } while (0)

#define RUN_TEST(func)                                           \
    do {                                                         \
        g_total++;                                               \
        std::cout << "  " << #func << " ... ";                   \
        std::cout.flush();                                       \
        try {                                                    \
            if (func()) {                                        \
                g_passed++;                                      \
                std::cout << "PASSED" << std::endl;              \
            } else {                                             \
                g_failed++;                                      \
                std::cout << "FAILED" << std::endl;              \
            }                                                    \
        } catch (...) {                                          \
            g_failed++;                                          \
            std::cout << "EXCEPTION" << std::endl;               \
        }                                                        \
    } while (0)

#define TEST_RESULTS()                                                        \
    do {                                                                      \
        std::cout << "\n============================" << std::endl;           \
        std::cout << "Results: " << g_passed << "/" << g_total << " passed";  \
        if (g_failed > 0) std::cout << ", " << g_failed << " FAILED";         \
        std::cout << std::endl;                                               \
    } while (0)

// ============================================================
// Shared test helpers
// ============================================================

static T s_center[3];
static T s_cams[23 * 10];

inline void setup_default_params() {
    s_center[0] = 0.0;
    s_center[1] = 0.0;
    s_center[2] = 0.0;
    params::center = s_center;
    params::size = 2.0;
    params::n_cams = 0;
    params::cams = NULL;
    params::pixels_per_cube = 8.0;
    params::occ_scale = 10.0;
    params::min_dist = 1.0;
    params::memory_limit_mb = 1000;
    params::coarse_count = 500000;
    params::n_elements = 1;
}

inline void setup_simple_camera() {
    setup_default_params();
    params::n_cams = 1;
    params::cams = s_cams;
    memset(s_cams, 0, sizeof(s_cams));
    // inv_pose: identity rotation, zero translation (3x4)
    s_cams[0] = 1.0;
    s_cams[5] = 1.0;
    s_cams[10] = 1.0;
    // K: [500 0 320; 0 500 240; 0 0 1]
    s_cams[12] = 500.0;
    s_cams[14] = 320.0;
    s_cams[16] = 500.0;
    s_cams[17] = 240.0;
    s_cams[20] = 1.0;
    // H=480, W=640
    s_cams[21] = 480.0;
    s_cams[22] = 640.0;
}

inline node make_root_leaf() {
    node root;
    mark_leaf_node(root);
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    return root;
}

inline node make_partial_root() {
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    return root;
}

inline cube make_cube(int x, int y, int z, int L) {
    cube c;
    c.coords[0] = x;
    c.coords[1] = y;
    c.coords[2] = z;
    c.L = L;
    return c;
}

#endif // TEST_FRAMEWORK_H
