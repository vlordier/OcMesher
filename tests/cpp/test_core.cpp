// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the
// LICENSE file in the root directory of this source tree.

// Comprehensive unit tests for ocmesher C++ core functionality.
// Build: g++ -std=c++11 -O2 -o test_core test_core.cpp -lm
// Run:   ./test_core

#include <iostream>
#include <string>
#include <limits>

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
            std::cerr << "    FAIL: " << #a << " == " << #b << " at "         \
                      << __FILE__ << ":" << __LINE__ << std::endl;            \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                      \
        if (std::fabs((double)(a) - (double)(b)) > (double)(tol)) {           \
            std::cerr << "    FAIL: |" << #a << " - " << #b << "| <= " << #tol \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define TEST_SUITE(name)                           \
    do {                                           \
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

// ============================================================
// Helper functions for setting up test state
// ============================================================

static T s_center[3];
static T s_cams[23 * 10];

void setup_default_params() {
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

void setup_simple_camera() {
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

node make_root_leaf() {
    node root;
    mark_leaf_node(root);
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    return root;
}

// ============================================================
// Test: Macros
// ============================================================

bool test_cubex() {
    ASSERT_EQ(cubex(0), 0);
    ASSERT_EQ(cubex(1), 1);
    ASSERT_EQ(cubex(2), 8);
    ASSERT_EQ(cubex(3), 27);
    ASSERT_EQ(cubex(4), 64);
    ASSERT_EQ(cubex(5), 125);
    return true;
}

bool test_cube_index() {
    // cube_index(x, y, z, s) = x*s*s + y*s + z
    ASSERT_EQ(cube_index(0, 0, 0, 2), 0);
    ASSERT_EQ(cube_index(1, 0, 0, 2), 4);
    ASSERT_EQ(cube_index(0, 1, 0, 2), 2);
    ASSERT_EQ(cube_index(0, 0, 1, 2), 1);
    ASSERT_EQ(cube_index(1, 1, 1, 2), 7);
    ASSERT_EQ(cube_index(0, 0, 0, 3), 0);
    ASSERT_EQ(cube_index(2, 2, 2, 3), 26);
    ASSERT_EQ(cube_index(1, 0, 0, 4), 16);
    return true;
}

bool test_first_second_digit() {
    ASSERT_EQ(first_digit(0), 0);
    ASSERT_EQ(first_digit(1), 1);
    ASSERT_EQ(first_digit(2), 0);
    ASSERT_EQ(first_digit(3), 1);
    ASSERT_EQ(first_digit(4), 0);
    ASSERT_EQ(first_digit(7), 1);

    ASSERT_EQ(second_digit(0), 0);
    ASSERT_EQ(second_digit(1), 0);
    ASSERT_EQ(second_digit(2), 1);
    ASSERT_EQ(second_digit(3), 1);
    ASSERT_EQ(second_digit(4), 0);
    ASSERT_EQ(second_digit(5), 0);
    ASSERT_EQ(second_digit(6), 1);
    ASSERT_EQ(second_digit(7), 1);
    return true;
}

bool test_make_int3() {
    int3 v = make_int3(1, 2, 3);
    ASSERT_EQ(xpp(v), 1);
    ASSERT_EQ(ypp(v), 2);
    ASSERT_EQ(zpp(v), 3);

    v = make_int3(0, 0, 0);
    ASSERT_EQ(xpp(v), 0);
    ASSERT_EQ(ypp(v), 0);
    ASSERT_EQ(zpp(v), 0);

    v = make_int3(-5, 10, 100);
    ASSERT_EQ(xpp(v), -5);
    ASSERT_EQ(ypp(v), 10);
    ASSERT_EQ(zpp(v), 100);
    return true;
}

// ============================================================
// Test: Node and Cube Marking
// ============================================================

bool test_leaf_node_marking() {
    node n;
    mark_leaf_node(n);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(n.nxts[0], -2);

    n.nxts[0] = 0;
    ASSERT_FALSE(leaf_node(n));

    n.nxts[0] = 5;
    ASSERT_FALSE(leaf_node(n));

    n.nxts[0] = -1;
    ASSERT_FALSE(leaf_node(n));
    return true;
}

bool test_grid_node_marking() {
    node n;

    mark_grid_node(n, 0);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(grid_node_level(n), 0);
    ASSERT_EQ(n.nxts[0], -2);

    mark_grid_node(n, 1);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(grid_node_level(n), 1);
    ASSERT_EQ(n.nxts[0], -3);

    mark_grid_node(n, 2);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(grid_node_level(n), 2);

    mark_grid_node(n, 5);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(grid_node_level(n), 5);

    mark_grid_node(n, 10);
    ASSERT_TRUE(leaf_node(n));
    ASSERT_EQ(grid_node_level(n), 10);
    return true;
}

bool test_cube_status_regular() {
    cube c;
    c.L = 0;
    ASSERT_TRUE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));

    c.L = 5;
    ASSERT_TRUE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));
    return true;
}

bool test_cube_status_boundary() {
    cube c;
    c.L = 3;
    mark_boundary(c);
    ASSERT_FALSE(is_regular(c));
    ASSERT_TRUE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));
    ASSERT_EQ(c.L, -1);
    return true;
}

bool test_cube_status_exterior() {
    cube c;
    c.L = 3;
    mark_exterior(c);
    ASSERT_FALSE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_TRUE(is_exterior(c));
    ASSERT_EQ(c.L, -2);
    return true;
}

bool test_cube_key_conversion() {
    cube c;
    c.coords[0] = 1;
    c.coords[1] = 2;
    c.coords[2] = 3;
    c.L = 4;
    key_cube key = cube_to_key(c);
    cube c2;
    key_to_cube(c2, key);
    ASSERT_EQ(c2.coords[0], 1);
    ASSERT_EQ(c2.coords[1], 2);
    ASSERT_EQ(c2.coords[2], 3);
    ASSERT_EQ(c2.L, 4);
    return true;
}

bool test_cube_key_roundtrip_various() {
    int test_coords[][4] = {
        {0, 0, 0, 0},   {1, 0, 0, 1},    {0, 1, 0, 1},
        {0, 0, 1, 1},   {1, 1, 1, 1},    {3, 2, 1, 2},
        {7, 5, 3, 3},   {15, 14, 13, 4}, {31, 0, 31, 5},
        {100, 200, 50, 10},
    };
    for (int t = 0; t < 10; t++) {
        cube c;
        c.coords[0] = test_coords[t][0];
        c.coords[1] = test_coords[t][1];
        c.coords[2] = test_coords[t][2];
        c.L = test_coords[t][3];
        key_cube key = cube_to_key(c);
        cube c2;
        key_to_cube(c2, key);
        ASSERT_EQ(c2.coords[0], c.coords[0]);
        ASSERT_EQ(c2.coords[1], c.coords[1]);
        ASSERT_EQ(c2.coords[2], c.coords[2]);
        ASSERT_EQ(c2.L, c.L);
    }
    return true;
}

bool test_cube_key_uniqueness() {
    cube c1, c2;
    c1.coords[0] = 0; c1.coords[1] = 0; c1.coords[2] = 0; c1.L = 1;
    c2.coords[0] = 1; c2.coords[1] = 0; c2.coords[2] = 0; c2.L = 1;
    key_cube k1 = cube_to_key(c1);
    key_cube k2 = cube_to_key(c2);
    ASSERT_TRUE(k1 != k2);

    // Same coords different level
    c2.coords[0] = 0; c2.L = 2;
    k2 = cube_to_key(c2);
    ASSERT_TRUE(k1 != k2);
    return true;
}

// ============================================================
// Test: Struct initialization
// ============================================================

bool test_computed_vertex_init() {
    computed_vertex cv;
    cv.c[0] = 1.0;
    cv.c[1] = 2.0;
    cv.c[2] = 3.0;
    cv.l = 0.0;
    cv.r = 0.5;
    ASSERT_NEAR(cv.c[0], 1.0, 1e-10);
    ASSERT_NEAR(cv.c[1], 2.0, 1e-10);
    ASSERT_NEAR(cv.c[2], 3.0, 1e-10);
    ASSERT_NEAR(cv.l, 0.0, 1e-10);
    ASSERT_NEAR(cv.r, 0.5, 1e-10);
    return true;
}

bool test_cube_struct_init() {
    cube c;
    c.coords[0] = 5;
    c.coords[1] = 10;
    c.coords[2] = 15;
    c.L = 3;
    ASSERT_EQ(c.coords[0], 5);
    ASSERT_EQ(c.coords[1], 10);
    ASSERT_EQ(c.coords[2], 15);
    ASSERT_EQ(c.L, 3);
    return true;
}

bool test_node_struct_init() {
    node n;
    n.c.coords[0] = 1;
    n.c.coords[1] = 2;
    n.c.coords[2] = 3;
    n.c.L = 2;
    for (int i = 0; i < 8; i++) n.nxts[i] = i * 10;
    ASSERT_EQ(n.c.coords[0], 1);
    ASSERT_EQ(n.c.L, 2);
    ASSERT_EQ(n.nxts[0], 0);
    ASSERT_EQ(n.nxts[7], 70);
    return true;
}

// ============================================================
// Test: int_log
// ============================================================

bool test_int_log_powers_of_two() {
    ASSERT_EQ(int_log(1.0), 0);
    ASSERT_EQ(int_log(2.0), 1);
    ASSERT_EQ(int_log(4.0), 2);
    ASSERT_EQ(int_log(8.0), 3);
    ASSERT_EQ(int_log(16.0), 4);
    ASSERT_EQ(int_log(32.0), 5);
    ASSERT_EQ(int_log(64.0), 6);
    ASSERT_EQ(int_log(128.0), 7);
    ASSERT_EQ(int_log(256.0), 8);
    ASSERT_EQ(int_log(1024.0), 10);
    return true;
}

bool test_int_log_non_powers() {
    ASSERT_EQ(int_log(3.0), 2);
    ASSERT_EQ(int_log(5.0), 3);
    ASSERT_EQ(int_log(6.0), 3);
    ASSERT_EQ(int_log(7.0), 3);
    ASSERT_EQ(int_log(9.0), 4);
    ASSERT_EQ(int_log(15.0), 4);
    ASSERT_EQ(int_log(17.0), 5);
    ASSERT_EQ(int_log(1025.0), 11);
    return true;
}

bool test_int_log_fractional() {
    ASSERT_EQ(int_log(0.5), 0);
    ASSERT_EQ(int_log(0.1), 0);
    ASSERT_EQ(int_log(0.001), 0);
    ASSERT_EQ(int_log(1.5), 1);
    return true;
}

bool test_int_log_zero() {
    ASSERT_EQ(int_log(0.0), 0);
    return true;
}

bool test_int_log_large_values() {
    ASSERT_EQ(int_log(1048576.0), 20);
    ASSERT_EQ(int_log(1048577.0), 21);
    ASSERT_EQ(int_log(1000000.0), 20);
    return true;
}

// ============================================================
// Test: det (3x3 determinant)
// ============================================================

bool test_det_identity() {
    T m[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    ASSERT_NEAR(det(m), 1.0, 1e-10);
    return true;
}

bool test_det_zero_matrix() {
    T m[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_singular() {
    // row3 = row1 + row2
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {5, 7, 9}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_known_value() {
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    // det = 1*(0-48) - 2*(0-42) + 3*(32-35) = -48+84-9 = 27
    ASSERT_NEAR(det(m), 27.0, 1e-10);
    return true;
}

bool test_det_negative() {
    T m[3][3] = {{2, 0, 0}, {0, 3, 0}, {0, 0, -1}};
    ASSERT_NEAR(det(m), -6.0, 1e-10);
    return true;
}

bool test_det_diagonal() {
    T m[3][3] = {{2, 0, 0}, {0, 3, 0}, {0, 0, 5}};
    ASSERT_NEAR(det(m), 30.0, 1e-10);
    return true;
}

bool test_det_antisymmetric() {
    T m[3][3] = {{0, 1, -1}, {-1, 0, 1}, {1, -1, 0}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_scale_invariance() {
    T m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    T d1 = det(m1);
    T k = 2.0;
    T m2[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) m2[i][j] = m1[i][j] * k;
    T d2 = det(m2);
    ASSERT_NEAR(d2, d1 * k * k * k, 1e-8);
    return true;
}

bool test_det_transpose_equality() {
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    T d1 = det(m);
    T mt[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) mt[i][j] = m[j][i];
    T d2 = det(mt);
    ASSERT_NEAR(d1, d2, 1e-10);
    return true;
}

bool test_det_row_swap_negation() {
    T m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    T d1 = det(m1);
    // Swap rows 0 and 1
    T m2[3][3] = {{4, 5, 6}, {1, 2, 3}, {7, 8, 0}};
    T d2 = det(m2);
    ASSERT_NEAR(d1, -d2, 1e-10);
    return true;
}

// ============================================================
// Test: tri_seg_intersect
// ============================================================

bool test_tri_seg_through_center() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, -1};
    T s2[] = {0.1, 0.1, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_no_intersection_parallel() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0, 0, 1};
    T s2[] = {1, 0, 1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_no_intersection_same_side() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, 1};
    T s2[] = {0.1, 0.1, 2};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_miss_outside_triangle() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {5, 5, -1};
    T s2[] = {5, 5, 1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_reverse_direction() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, 1};
    T s2[] = {0.1, 0.1, -1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_large_triangle() {
    T t1[] = {-100, -100, 0};
    T t2[] = {100, -100, 0};
    T t3[] = {0, 100, 0};
    T s1[] = {0, 0, -50};
    T s2[] = {0, 0, 50};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_near_edge() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.01, 0.01, -1};
    T s2[] = {0.01, 0.01, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_negative_z_plane() {
    T t1[] = {0, 0, -5};
    T t2[] = {1, 0, -5};
    T t3[] = {0, 1, -5};
    T s1[] = {0.2, 0.2, -10};
    T s2[] = {0.2, 0.2, 0};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_different_planes() {
    // Triangle in tilted plane
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 1};
    T t3[] = {0, 1, 0.5};
    T s1[] = {0.2, 0.2, -1};
    T s2[] = {0.2, 0.2, 2};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tri_seg_miss_behind() {
    // Segment entirely behind the triangle
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.2, 0.2, -3};
    T s2[] = {0.2, 0.2, -1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// Test: compute_coords
// ============================================================

bool test_compute_coords_origin() {
    setup_default_params();
    T coords[3];
    int icoords[3] = {0, 0, 0};
    compute_coords(coords, icoords, 1);
    // center=(0,0,0), size=2 → 0 - 1 + 2*0/2 = -1
    ASSERT_NEAR(coords[0], -1.0, 1e-10);
    ASSERT_NEAR(coords[1], -1.0, 1e-10);
    ASSERT_NEAR(coords[2], -1.0, 1e-10);
    return true;
}

bool test_compute_coords_center() {
    setup_default_params();
    T coords[3];
    int icoords[3] = {1, 1, 1};
    compute_coords(coords, icoords, 1);
    ASSERT_NEAR(coords[0], 0.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 0.0, 1e-10);
    return true;
}

bool test_compute_coords_far_corner() {
    setup_default_params();
    T coords[3];
    int icoords[3] = {2, 2, 2};
    compute_coords(coords, icoords, 1);
    ASSERT_NEAR(coords[0], 1.0, 1e-10);
    ASSERT_NEAR(coords[1], 1.0, 1e-10);
    ASSERT_NEAR(coords[2], 1.0, 1e-10);
    return true;
}

bool test_compute_coords_level2() {
    setup_default_params();
    T coords[3];
    int icoords[3] = {1, 2, 3};
    compute_coords(coords, icoords, 2);
    // coords[j] = 0 - 1 + 2*icoords[j]/4
    ASSERT_NEAR(coords[0], -0.5, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 0.5, 1e-10);
    return true;
}

bool test_compute_coords_level3() {
    setup_default_params();
    T coords[3];
    int icoords[3] = {0, 4, 8};
    compute_coords(coords, icoords, 3);
    // coords[j] = 0 - 1 + 2*icoords[j]/8
    ASSERT_NEAR(coords[0], -1.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 1.0, 1e-10);
    return true;
}

bool test_compute_coords_offset_center() {
    s_center[0] = 1.0;
    s_center[1] = 2.0;
    s_center[2] = 3.0;
    params::center = s_center;
    params::size = 4.0;
    T coords[3];
    int icoords[3] = {0, 0, 0};
    compute_coords(coords, icoords, 1);
    // coords[0] = 1 - 2 + 4*0/2 = -1
    // coords[1] = 2 - 2 + 4*0/2 = 0
    // coords[2] = 3 - 2 + 4*0/2 = 1
    ASSERT_NEAR(coords[0], -1.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 1.0, 1e-10);
    setup_default_params();
    return true;
}

bool test_compute_coords_monotonic() {
    setup_default_params();
    for (int i = 0; i < 8; i++) {
        T c1[3], c2[3];
        int ic1[3] = {i, 0, 0};
        int ic2[3] = {i + 1, 0, 0};
        compute_coords(c1, ic1, 3);
        compute_coords(c2, ic2, 3);
        ASSERT_TRUE(c2[0] > c1[0]);
    }
    return true;
}

// ============================================================
// Test: compute_center
// ============================================================

bool test_compute_center_root() {
    setup_default_params();
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 0;
    T coords[3];
    compute_center(coords, c);
    // center[j] - size/2 + size*(0+0.5)/1 = 0 - 1 + 1 = 0
    ASSERT_NEAR(coords[0], 0.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 0.0, 1e-10);
    return true;
}

bool test_compute_center_level1_children() {
    setup_default_params();
    // Child (0,0,0) at level 1
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 1;
    T coords[3];
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], -0.5, 1e-10);
    ASSERT_NEAR(coords[1], -0.5, 1e-10);
    ASSERT_NEAR(coords[2], -0.5, 1e-10);

    // Child (1,1,1) at level 1
    c.coords[0] = 1;
    c.coords[1] = 1;
    c.coords[2] = 1;
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], 0.5, 1e-10);
    ASSERT_NEAR(coords[1], 0.5, 1e-10);
    ASSERT_NEAR(coords[2], 0.5, 1e-10);
    return true;
}

bool test_compute_center_level2() {
    setup_default_params();
    cube c;
    c.coords[0] = 1;
    c.coords[1] = 2;
    c.coords[2] = 3;
    c.L = 2;
    T coords[3];
    compute_center(coords, c);
    // center[j] - size/2 + size*(coords[j]+0.5)/(1<<L)
    // 0 - 1 + 2*(1.5)/4 = -0.25
    // 0 - 1 + 2*(2.5)/4 = 0.25
    // 0 - 1 + 2*(3.5)/4 = 0.75
    ASSERT_NEAR(coords[0], -0.25, 1e-10);
    ASSERT_NEAR(coords[1], 0.25, 1e-10);
    ASSERT_NEAR(coords[2], 0.75, 1e-10);
    return true;
}

bool test_compute_center_within_cube() {
    setup_default_params();
    cube c;
    c.coords[0] = 1;
    c.coords[1] = 2;
    c.coords[2] = 3;
    c.L = 3;
    T center[3];
    compute_center(center, c);
    T corner_lo[3], corner_hi[3];
    int ic_lo[3] = {c.coords[0], c.coords[1], c.coords[2]};
    int ic_hi[3] = {c.coords[0] + 1, c.coords[1] + 1, c.coords[2] + 1};
    compute_coords(corner_lo, ic_lo, c.L);
    compute_coords(corner_hi, ic_hi, c.L);
    for (int j = 0; j < 3; j++) {
        ASSERT_TRUE(center[j] > corner_lo[j]);
        ASSERT_TRUE(center[j] < corner_hi[j]);
    }
    return true;
}

// ============================================================
// Test: expand_octree
// ============================================================

bool test_expand_octree_root() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    ASSERT_EQ((int)nodes.size(), 1);
    ASSERT_TRUE(leaf_node(nodes[0]));

    expand_octree(nodes, 0);
    ASSERT_EQ((int)nodes.size(), 9);
    ASSERT_FALSE(leaf_node(nodes[0]));

    for (int i = 1; i <= 8; i++) {
        ASSERT_TRUE(leaf_node(nodes[i]));
        ASSERT_EQ(nodes[i].c.L, 1);
    }
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(nodes[0].nxts[i], 1 + i);
    }
    return true;
}

bool test_expand_octree_child_coords() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(nodes[1 + i].c.coords[0], (i >> 0) & 1);
        ASSERT_EQ(nodes[1 + i].c.coords[1], (i >> 1) & 1);
        ASSERT_EQ(nodes[1 + i].c.coords[2], (i >> 2) & 1);
    }
    return true;
}

bool test_expand_octree_grandchildren() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);
    expand_octree(nodes, 1);

    ASSERT_EQ((int)nodes.size(), 17);
    ASSERT_FALSE(leaf_node(nodes[1]));

    for (int i = 9; i < 17; i++) {
        ASSERT_TRUE(leaf_node(nodes[i]));
        ASSERT_EQ(nodes[i].c.L, 2);
    }
    // First grandchild should be at coords (0,0,0)
    ASSERT_EQ(nodes[9].c.coords[0], 0);
    ASSERT_EQ(nodes[9].c.coords[1], 0);
    ASSERT_EQ(nodes[9].c.coords[2], 0);
    return true;
}

bool test_expand_octree_preserves_siblings() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    int c7_coords[3] = {nodes[8].c.coords[0], nodes[8].c.coords[1],
                         nodes[8].c.coords[2]};
    int c7_L = nodes[8].c.L;

    expand_octree(nodes, 1);

    ASSERT_EQ(nodes[8].c.coords[0], c7_coords[0]);
    ASSERT_EQ(nodes[8].c.coords[1], c7_coords[1]);
    ASSERT_EQ(nodes[8].c.coords[2], c7_coords[2]);
    ASSERT_EQ(nodes[8].c.L, c7_L);
    return true;
}

bool test_expand_octree_multi_level() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    for (int i = 1; i <= 8; i++) expand_octree(nodes, i);

    ASSERT_EQ((int)nodes.size(), 9 + 64);
    for (int i = 9; i < 73; i++) {
        ASSERT_TRUE(leaf_node(nodes[i]));
        ASSERT_EQ(nodes[i].c.L, 2);
    }
    return true;
}

bool test_expand_octree_spatial_coverage() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    set<int3> child_coords;
    for (int i = 1; i <= 8; i++) {
        child_coords.insert(
            make_int3(nodes[i].c.coords[0], nodes[i].c.coords[1],
                      nodes[i].c.coords[2]));
    }
    ASSERT_EQ((int)child_coords.size(), 8);
    return true;
}

// ============================================================
// Test: partial_expand_octree
// ============================================================

bool test_partial_expand_single() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    partial_expand_octree(nodes, 0, 1);

    ASSERT_EQ((int)nodes.size(), 2);
    ASSERT_EQ(nodes[0].nxts[0], 1);
    for (int i = 1; i < 8; i++) ASSERT_EQ(nodes[0].nxts[i], -1);
    return true;
}

bool test_partial_expand_multiple() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    partial_expand_octree(nodes, 0, 7); // children 0,1,2
    ASSERT_EQ((int)nodes.size(), 4);
    ASSERT_EQ(nodes[0].nxts[0], 1);
    ASSERT_EQ(nodes[0].nxts[1], 2);
    ASSERT_EQ(nodes[0].nxts[2], 3);
    for (int i = 3; i < 8; i++) ASSERT_EQ(nodes[0].nxts[i], -1);
    return true;
}

bool test_partial_expand_idempotent() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    partial_expand_octree(nodes, 0, 1);
    int first_child_id = nodes[0].nxts[0];

    partial_expand_octree(nodes, 0, 1);
    ASSERT_EQ(nodes[0].nxts[0], first_child_id);
    ASSERT_EQ((int)nodes.size(), 2);
    return true;
}

bool test_partial_expand_all_children() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    partial_expand_octree(nodes, 0, 0xFF);
    ASSERT_EQ((int)nodes.size(), 9);
    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(nodes[0].nxts[i] >= 1);
        ASSERT_TRUE(nodes[0].nxts[i] <= 8);
    }
    return true;
}

bool test_partial_expand_child_coords() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    partial_expand_octree(nodes, 0, 0xFF);

    for (int i = 0; i < 8; i++) {
        int child = nodes[0].nxts[i];
        ASSERT_EQ(nodes[child].c.L, 1);
        ASSERT_EQ(nodes[child].c.coords[0], (i >> 0) & 1);
        ASSERT_EQ(nodes[child].c.coords[1], (i >> 1) & 1);
        ASSERT_EQ(nodes[child].c.coords[2], (i >> 2) & 1);
    }
    return true;
}

// ============================================================
// Test: enumerate_vertices
// ============================================================

bool test_enumerate_vertices_level0_count() {
    setup_default_params();
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    // Grid level 0 -> ss=1, vertices: 2^3 = 8
    vertex v[8];
    enumerate_vertices(v, n);

    // Check vertex 0 at (0,0,0) gets promoted to level 0
    ASSERT_EQ(v[0].coords[0], 0);
    ASSERT_EQ(v[0].coords[1], 0);
    ASSERT_EQ(v[0].coords[2], 0);
    ASSERT_EQ(v[0].L, 0);
    return true;
}

bool test_enumerate_vertices_level0_coords() {
    setup_default_params();
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    vertex v[8];
    enumerate_vertices(v, n);

    // vid=1: (1,0,0) at L=1, has odd coord so stays
    ASSERT_EQ(v[1].coords[0], 1);
    ASSERT_EQ(v[1].coords[1], 0);
    ASSERT_EQ(v[1].coords[2], 0);
    ASSERT_EQ(v[1].L, 1);
    return true;
}

bool test_enumerate_vertices_level1() {
    setup_default_params();
    node n;
    mark_grid_node(n, 1);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 0;

    // Grid level 1 -> ss=2, vertices: 3^3 = 27
    vertex v[27];
    enumerate_vertices(v, n);

    // vid 0: (0,0,0) at L=1 -> promoted to L=0
    ASSERT_EQ(v[0].L, 0);
    ASSERT_EQ(v[0].coords[0], 0);

    // vid 1: (1,0,0) at L=1 -> stays (odd coord)
    ASSERT_EQ(v[1].L, 1);
    ASSERT_EQ(v[1].coords[0], 1);
    return true;
}

bool test_enumerate_vertices_level1_all_valid() {
    setup_default_params();
    node n;
    mark_grid_node(n, 1);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 0;

    vertex v[27];
    enumerate_vertices(v, n);

    // All vertices should have non-negative levels
    for (int i = 0; i < 27; i++) {
        ASSERT_TRUE(v[i].L >= 0);
    }
    return true;
}

// ============================================================
// Test: search
// ============================================================

bool test_search_out_of_bounds_zero() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    int coords[3] = {0, 0, 0};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

bool test_search_out_of_bounds_max() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    int coords[3] = {2, 1, 1};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

bool test_search_valid_interior() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    for (int i = 1; i <= 8; i++) mark_grid_node(nodes[i], 0);

    int coords[3] = {1, 1, 1};
    cube result = search(&nodes[0], coords, 2);
    ASSERT_TRUE(is_regular(result));
    return true;
}

bool test_search_negative_coords() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    int coords[3] = {-1, 1, 1};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

// ============================================================
// Test: divide_to_cube
// ============================================================

bool test_divide_to_cube_level1() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    cube target;
    target.coords[0] = 0;
    target.coords[1] = 0;
    target.coords[2] = 0;
    target.L = 1;

    int node_id = divide_to_cube(nodes, target);
    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.coords[0], 0);
    ASSERT_EQ(nodes[node_id].c.coords[1], 0);
    ASSERT_EQ(nodes[node_id].c.coords[2], 0);
    ASSERT_EQ(nodes[node_id].c.L, 1);
    return true;
}

bool test_divide_to_cube_level2() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    cube target;
    target.coords[0] = 1;
    target.coords[1] = 1;
    target.coords[2] = 1;
    target.L = 2;

    int node_id = divide_to_cube(nodes, target);
    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.coords[0], 1);
    ASSERT_EQ(nodes[node_id].c.coords[1], 1);
    ASSERT_EQ(nodes[node_id].c.coords[2], 1);
    ASSERT_EQ(nodes[node_id].c.L, 2);
    return true;
}

bool test_divide_to_cube_creates_path() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    cube target;
    target.coords[0] = 3;
    target.coords[1] = 3;
    target.coords[2] = 3;
    target.L = 3;

    int node_id = divide_to_cube(nodes, target);
    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.L, 3);
    // Intermediate nodes should have been created
    ASSERT_TRUE((int)nodes.size() > 2);
    return true;
}

// ============================================================
// Test: projected_coords
// ============================================================

bool test_projected_coords_distance() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 0;
    T r;
    projected_coords(c, 0, NULL, &r);
    // Cube center at (0,0,0), camera at origin, distance = 0 clamped to min_dist
    ASSERT_NEAR(r, 1.0, 1e-10);
    return true;
}

bool test_projected_coords_image_coords() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 1;
    c.coords[1] = 1;
    c.coords[2] = 1;
    c.L = 1;

    T icoords[3], r;
    projected_coords(c, 0, icoords, &r);

    // Cube center = (0.5, 0.5, 0.5), identity inv_pose -> Pc = (0.5, 0.5, 0.5)
    // K = [500,0,320; 0,500,240; 0,0,1]
    // icoords[0] = (500*0.5 + 0 + 320*0.5) / 0.5 = 410/0.5 = 820
    // icoords[1] = (0 + 500*0.5 + 240*0.5) / 0.5 = 370/0.5 = 740
    ASSERT_NEAR(icoords[0], 820.0, 1e-5);
    ASSERT_NEAR(icoords[1], 740.0, 1e-5);
    return true;
}

bool test_projected_coords_null_icoords() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 0;
    T r;
    projected_coords(c, 0, NULL, &r);
    ASSERT_NEAR(r, 1.0, 1e-10);
    return true;
}

bool test_projected_coords_null_r() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 1;
    c.coords[1] = 1;
    c.coords[2] = 1;
    c.L = 1;
    T icoords[3];
    projected_coords(c, 0, icoords, NULL);
    // Should not crash and return valid icoords
    ASSERT_TRUE(std::isfinite(icoords[0]));
    ASSERT_TRUE(std::isfinite(icoords[1]));
    return true;
}

// ============================================================
// Test: projected_size
// ============================================================

bool test_projected_size_positive() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 0;
    T size = projected_size(c, 0);
    ASSERT_TRUE(size > 0);
    return true;
}

bool test_projected_size_deeper_smaller() {
    setup_simple_camera();
    cube c1, c2;
    c1.coords[0] = 0;
    c1.coords[1] = 0;
    c1.coords[2] = 0;
    c1.L = 1;
    c2.coords[0] = 0;
    c2.coords[1] = 0;
    c2.coords[2] = 0;
    c2.L = 2;
    T size1 = projected_size(c1, 0);
    T size2 = projected_size(c2, 0);
    ASSERT_TRUE(size1 > size2);
    return true;
}

bool test_projected_size_max_over_cameras() {
    setup_simple_camera();
    params::n_cams = 2;
    memcpy(s_cams + 23, s_cams, 23 * sizeof(T));
    s_cams[23 + 3] = 1.0;

    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 1;
    T max_size = projected_size(c);
    T size0 = projected_size(c, 0);
    T size1 = projected_size(c, 1);
    T expected_max = size0 > size1 ? size0 : size1;
    ASSERT_NEAR(max_size, expected_max, 1e-10);
    return true;
}

bool test_projected_size_single_camera_consistency() {
    setup_simple_camera();
    cube c;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 1;
    T single = projected_size(c, 0);
    T maximum = projected_size(c);
    ASSERT_NEAR(single, maximum, 1e-10);
    return true;
}

// ============================================================
// Test: compute_boundary
// ============================================================

bool test_compute_boundary_same_cube() {
    cube c, bound;
    c.coords[0] = 1;
    c.coords[1] = 1;
    c.coords[2] = 1;
    c.L = 2;
    bound.coords[0] = 1;
    bound.coords[1] = 1;
    bound.coords[2] = 1;
    bound.L = 2;
    int instance = compute_boundary(c, bound);
    ASSERT_EQ(instance, 255);
    return true;
}

bool test_compute_boundary_valid_range() {
    cube c, bound;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 2;
    bound.coords[0] = 0;
    bound.coords[1] = 0;
    bound.coords[2] = 0;
    bound.L = 1;
    int instance = compute_boundary(c, bound);
    ASSERT_TRUE(instance >= 0 && instance <= 255);
    return true;
}

bool test_compute_boundary_corner_cube() {
    cube c, bound;
    c.coords[0] = 0;
    c.coords[1] = 0;
    c.coords[2] = 0;
    c.L = 2;
    bound.coords[0] = 0;
    bound.coords[1] = 0;
    bound.coords[2] = 0;
    bound.L = 1;
    int instance = compute_boundary(c, bound);
    // Corner cube: the interior octant (7) should not be on boundary
    // All other octants touch at least one face
    ASSERT_TRUE(instance > 0);
    ASSERT_TRUE((instance & (1 << 7)) == 0);
    return true;
}

// ============================================================
// Test: find_edges
// ============================================================

bool test_find_edges_uniform_positive() {
    setup_default_params();
    params::n_elements = 1;
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    map<key_cube, int> vmap;
    vertex v[8];
    enumerate_vertices(v, n);
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = 1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    for (int i = 0; i <= params::n_elements; i++)
        ASSERT_EQ((int)bipolar_edges[i].size(), 0);
    return true;
}

bool test_find_edges_uniform_negative() {
    setup_default_params();
    params::n_elements = 1;
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    map<key_cube, int> vmap;
    vertex v[8];
    enumerate_vertices(v, n);
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    for (int i = 0; i <= params::n_elements; i++)
        ASSERT_EQ((int)bipolar_edges[i].size(), 0);
    return true;
}

bool test_find_edges_with_crossing() {
    setup_default_params();
    params::n_elements = 1;
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    map<key_cube, int> vmap;
    vertex v[8];
    enumerate_vertices(v, n);
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    // Split by z: indices 0-3 positive (k=0), 4-7 negative (k=1)
    sdfT sdf[8];
    for (int i = 0; i < 4; i++) sdf[i] = 1.0f;
    for (int i = 4; i < 8; i++) sdf[i] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    int total = 0;
    for (int i = 0; i <= params::n_elements; i++)
        total += bipolar_edges[i].size();
    ASSERT_TRUE(total > 0);
    // Expect exactly 4 bipolar edges per element along z-axis
    ASSERT_EQ((int)bipolar_edges[0].size(), 4);
    ASSERT_EQ((int)bipolar_edges[1].size(), 4);
    return true;
}

bool test_find_edges_single_negative_vertex() {
    setup_default_params();
    params::n_elements = 1;
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    map<key_cube, int> vmap;
    vertex v[8];
    enumerate_vertices(v, n);
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = 1.0f;
    sdf[0] = -1.0f;  // Only vertex 0 is negative

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // Vertex 0 connects to vertices 1,2,4 via edges, so 3 bipolar edges
    ASSERT_EQ((int)bipolar_edges[0].size(), 3);
    return true;
}

bool test_find_edges_multi_element() {
    setup_default_params();
    params::n_elements = 2;
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    map<key_cube, int> vmap;
    vertex v[8];
    enumerate_vertices(v, n);
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    // 2 elements, different sign patterns
    sdfT sdf[16];
    for (int i = 0; i < 8; i++) {
        sdf[i * 2 + 0] = 1.0f;      // element 0: all positive
        sdf[i * 2 + 1] = (i < 4) ? 1.0f : -1.0f;  // element 1: split by z
    }

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    ASSERT_EQ((int)bipolar_edges[0].size(), 0);  // element 0: no crossing
    ASSERT_EQ((int)bipolar_edges[1].size(), 4);  // element 1: 4 crossings
    return true;
}

// ============================================================
// Test: Integration - octree build and query
// ============================================================

bool test_octree_build_and_search() {
    setup_default_params();
    vector<node> nodes;
    node root = make_root_leaf();
    nodes.push_back(root);
    expand_octree(nodes, 0);

    // Mark all children as grid nodes level 0
    for (int i = 1; i <= 8; i++) mark_grid_node(nodes[i], 0);

    // Search for all valid internal points
    int found_count = 0;
    for (int x = 1; x < 4; x++)
        for (int y = 1; y < 4; y++)
            for (int z = 1; z < 4; z++) {
                int coords[3] = {x, y, z};
                cube result = search(&nodes[0], coords, 3);
                if (is_regular(result)) found_count++;
            }
    ASSERT_TRUE(found_count > 0);
    return true;
}

bool test_octree_divide_and_search() {
    setup_default_params();
    vector<node> nodes;
    node root;
    memset(root.nxts, -1, 8 * sizeof(int));
    memset(root.c.coords, 0, 3 * sizeof(int));
    root.c.L = 0;
    nodes.push_back(root);

    // Divide to create cube at (1,1,1) level 2
    cube target;
    target.coords[0] = 1;
    target.coords[1] = 1;
    target.coords[2] = 1;
    target.L = 2;
    int node_id = divide_to_cube(nodes, target);

    // Verify the node exists and is a leaf
    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.L, 2);
    return true;
}

// ============================================================
// Test: Coordinate system consistency
// ============================================================

bool test_center_is_midpoint_of_corners() {
    setup_default_params();
    cube c;
    c.coords[0] = 2;
    c.coords[1] = 3;
    c.coords[2] = 1;
    c.L = 3;

    T center[3];
    compute_center(center, c);

    T lo[3], hi[3];
    int ic_lo[3] = {c.coords[0], c.coords[1], c.coords[2]};
    int ic_hi[3] = {c.coords[0] + 1, c.coords[1] + 1, c.coords[2] + 1};
    compute_coords(lo, ic_lo, c.L);
    compute_coords(hi, ic_hi, c.L);

    for (int j = 0; j < 3; j++) {
        ASSERT_NEAR(center[j], (lo[j] + hi[j]) / 2.0, 1e-10);
    }
    return true;
}

bool test_adjacent_cubes_share_face() {
    setup_default_params();
    cube c1, c2;
    c1.coords[0] = 0;
    c1.coords[1] = 0;
    c1.coords[2] = 0;
    c1.L = 2;
    c2.coords[0] = 1;
    c2.coords[1] = 0;
    c2.coords[2] = 0;
    c2.L = 2;

    // The right face of c1 should coincide with the left face of c2
    T r1[3], l2[3];
    int ic_r1[3] = {c1.coords[0] + 1, c1.coords[1], c1.coords[2]};
    int ic_l2[3] = {c2.coords[0], c2.coords[1], c2.coords[2]};
    compute_coords(r1, ic_r1, c1.L);
    compute_coords(l2, ic_l2, c2.L);
    ASSERT_NEAR(r1[0], l2[0], 1e-10);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "OcMesher C++ Core Unit Tests" << std::endl;
    std::cout << "============================" << std::endl;

    TEST_SUITE("Macros");
    RUN_TEST(test_cubex);
    RUN_TEST(test_cube_index);
    RUN_TEST(test_first_second_digit);
    RUN_TEST(test_make_int3);

    TEST_SUITE("Node and Cube Marking");
    RUN_TEST(test_leaf_node_marking);
    RUN_TEST(test_grid_node_marking);
    RUN_TEST(test_cube_status_regular);
    RUN_TEST(test_cube_status_boundary);
    RUN_TEST(test_cube_status_exterior);
    RUN_TEST(test_cube_key_conversion);
    RUN_TEST(test_cube_key_roundtrip_various);
    RUN_TEST(test_cube_key_uniqueness);

    TEST_SUITE("Structs");
    RUN_TEST(test_computed_vertex_init);
    RUN_TEST(test_cube_struct_init);
    RUN_TEST(test_node_struct_init);

    TEST_SUITE("int_log");
    RUN_TEST(test_int_log_powers_of_two);
    RUN_TEST(test_int_log_non_powers);
    RUN_TEST(test_int_log_fractional);
    RUN_TEST(test_int_log_zero);
    RUN_TEST(test_int_log_large_values);

    TEST_SUITE("Determinant");
    RUN_TEST(test_det_identity);
    RUN_TEST(test_det_zero_matrix);
    RUN_TEST(test_det_singular);
    RUN_TEST(test_det_known_value);
    RUN_TEST(test_det_negative);
    RUN_TEST(test_det_diagonal);
    RUN_TEST(test_det_antisymmetric);
    RUN_TEST(test_det_scale_invariance);
    RUN_TEST(test_det_transpose_equality);
    RUN_TEST(test_det_row_swap_negation);

    TEST_SUITE("Triangle-Segment Intersection");
    RUN_TEST(test_tri_seg_through_center);
    RUN_TEST(test_tri_seg_no_intersection_parallel);
    RUN_TEST(test_tri_seg_no_intersection_same_side);
    RUN_TEST(test_tri_seg_miss_outside_triangle);
    RUN_TEST(test_tri_seg_reverse_direction);
    RUN_TEST(test_tri_seg_large_triangle);
    RUN_TEST(test_tri_seg_near_edge);
    RUN_TEST(test_tri_seg_negative_z_plane);
    RUN_TEST(test_tri_seg_different_planes);
    RUN_TEST(test_tri_seg_miss_behind);

    TEST_SUITE("Coordinate Computation");
    RUN_TEST(test_compute_coords_origin);
    RUN_TEST(test_compute_coords_center);
    RUN_TEST(test_compute_coords_far_corner);
    RUN_TEST(test_compute_coords_level2);
    RUN_TEST(test_compute_coords_level3);
    RUN_TEST(test_compute_coords_offset_center);
    RUN_TEST(test_compute_coords_monotonic);

    TEST_SUITE("Cube Center Computation");
    RUN_TEST(test_compute_center_root);
    RUN_TEST(test_compute_center_level1_children);
    RUN_TEST(test_compute_center_level2);
    RUN_TEST(test_compute_center_within_cube);

    TEST_SUITE("Projection");
    RUN_TEST(test_projected_coords_distance);
    RUN_TEST(test_projected_coords_image_coords);
    RUN_TEST(test_projected_coords_null_icoords);
    RUN_TEST(test_projected_coords_null_r);
    RUN_TEST(test_projected_size_positive);
    RUN_TEST(test_projected_size_deeper_smaller);
    RUN_TEST(test_projected_size_max_over_cameras);
    RUN_TEST(test_projected_size_single_camera_consistency);

    TEST_SUITE("Octree Expansion");
    RUN_TEST(test_expand_octree_root);
    RUN_TEST(test_expand_octree_child_coords);
    RUN_TEST(test_expand_octree_grandchildren);
    RUN_TEST(test_expand_octree_preserves_siblings);
    RUN_TEST(test_expand_octree_multi_level);
    RUN_TEST(test_expand_octree_spatial_coverage);

    TEST_SUITE("Partial Octree Expansion");
    RUN_TEST(test_partial_expand_single);
    RUN_TEST(test_partial_expand_multiple);
    RUN_TEST(test_partial_expand_idempotent);
    RUN_TEST(test_partial_expand_all_children);
    RUN_TEST(test_partial_expand_child_coords);

    TEST_SUITE("Enumerate Vertices");
    RUN_TEST(test_enumerate_vertices_level0_count);
    RUN_TEST(test_enumerate_vertices_level0_coords);
    RUN_TEST(test_enumerate_vertices_level1);
    RUN_TEST(test_enumerate_vertices_level1_all_valid);

    TEST_SUITE("Search");
    RUN_TEST(test_search_out_of_bounds_zero);
    RUN_TEST(test_search_out_of_bounds_max);
    RUN_TEST(test_search_valid_interior);
    RUN_TEST(test_search_negative_coords);

    TEST_SUITE("Divide to Cube");
    RUN_TEST(test_divide_to_cube_level1);
    RUN_TEST(test_divide_to_cube_level2);
    RUN_TEST(test_divide_to_cube_creates_path);

    TEST_SUITE("Boundary Computation");
    RUN_TEST(test_compute_boundary_same_cube);
    RUN_TEST(test_compute_boundary_valid_range);
    RUN_TEST(test_compute_boundary_corner_cube);

    TEST_SUITE("Find Edges");
    RUN_TEST(test_find_edges_uniform_positive);
    RUN_TEST(test_find_edges_uniform_negative);
    RUN_TEST(test_find_edges_with_crossing);
    RUN_TEST(test_find_edges_single_negative_vertex);
    RUN_TEST(test_find_edges_multi_element);

    TEST_SUITE("Integration");
    RUN_TEST(test_octree_build_and_search);
    RUN_TEST(test_octree_divide_and_search);
    RUN_TEST(test_center_is_midpoint_of_corners);
    RUN_TEST(test_adjacent_cubes_share_face);

    std::cout << "\n============================" << std::endl;
    std::cout << "Results: " << g_passed << "/" << g_total << " passed";
    if (g_failed > 0) std::cout << ", " << g_failed << " FAILED";
    std::cout << std::endl;

    return g_failed > 0 ? 1 : 0;
}
