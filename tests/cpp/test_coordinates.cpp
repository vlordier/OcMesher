// Tests for compute_coords and compute_center
// Build: g++ -std=c++11 -O2 -o test_coordinates test_coordinates.cpp -lm

#include "test_framework.h"

// ============================================================
// compute_coords
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
        ASSERT_GT(c2[0], c1[0]);
    }
    return true;
}

bool test_compute_coords_monotonic_all_axes() {
    setup_default_params();
    for (int axis = 0; axis < 3; axis++) {
        for (int i = 0; i < 8; i++) {
            T c1[3], c2[3];
            int ic1[3] = {0, 0, 0};
            int ic2[3] = {0, 0, 0};
            ic1[axis] = i;
            ic2[axis] = i + 1;
            compute_coords(c1, ic1, 3);
            compute_coords(c2, ic2, 3);
            ASSERT_GT(c2[axis], c1[axis]);
        }
    }
    return true;
}

bool test_compute_coords_uniform_spacing() {
    setup_default_params();
    // At level L, spacing between consecutive coords is size/(1<<L)
    for (int L = 1; L <= 5; L++) {
        T expected_spacing = params::size / (T)(1 << L);
        int ic1[3] = {0, 0, 0};
        int ic2[3] = {1, 0, 0};
        T c1[3], c2[3];
        compute_coords(c1, ic1, L);
        compute_coords(c2, ic2, L);
        ASSERT_NEAR(c2[0] - c1[0], expected_spacing, 1e-10);
    }
    return true;
}

bool test_compute_coords_level4() {
    setup_default_params();
    // Level 4 -> 16 divisions
    T coords[3];
    int icoords[3] = {8, 8, 8};
    compute_coords(coords, icoords, 4);
    // 0 - 1 + 2*8/16 = 0
    ASSERT_NEAR(coords[0], 0.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 0.0, 1e-10);
    return true;
}

bool test_compute_coords_asymmetric_size() {
    s_center[0] = 0.0;
    s_center[1] = 0.0;
    s_center[2] = 0.0;
    params::center = s_center;
    params::size = 10.0;
    T coords[3];
    int icoords[3] = {0, 0, 0};
    compute_coords(coords, icoords, 1);
    ASSERT_NEAR(coords[0], -5.0, 1e-10);
    int icoords2[3] = {2, 2, 2};
    compute_coords(coords, icoords2, 1);
    ASSERT_NEAR(coords[0], 5.0, 1e-10);
    setup_default_params();
    return true;
}

// ============================================================
// compute_center
// ============================================================

bool test_compute_center_root() {
    setup_default_params();
    cube c = make_cube(0, 0, 0, 0);
    T coords[3];
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], 0.0, 1e-10);
    ASSERT_NEAR(coords[1], 0.0, 1e-10);
    ASSERT_NEAR(coords[2], 0.0, 1e-10);
    return true;
}

bool test_compute_center_level1_children() {
    setup_default_params();
    cube c = make_cube(0, 0, 0, 1);
    T coords[3];
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], -0.5, 1e-10);
    ASSERT_NEAR(coords[1], -0.5, 1e-10);
    ASSERT_NEAR(coords[2], -0.5, 1e-10);

    c = make_cube(1, 1, 1, 1);
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], 0.5, 1e-10);
    ASSERT_NEAR(coords[1], 0.5, 1e-10);
    ASSERT_NEAR(coords[2], 0.5, 1e-10);
    return true;
}

bool test_compute_center_level2() {
    setup_default_params();
    cube c = make_cube(1, 2, 3, 2);
    T coords[3];
    compute_center(coords, c);
    ASSERT_NEAR(coords[0], -0.25, 1e-10);
    ASSERT_NEAR(coords[1], 0.25, 1e-10);
    ASSERT_NEAR(coords[2], 0.75, 1e-10);
    return true;
}

bool test_compute_center_within_cube() {
    setup_default_params();
    cube c = make_cube(1, 2, 3, 3);
    T center[3];
    compute_center(center, c);
    T corner_lo[3], corner_hi[3];
    int ic_lo[3] = {c.coords[0], c.coords[1], c.coords[2]};
    int ic_hi[3] = {c.coords[0] + 1, c.coords[1] + 1, c.coords[2] + 1};
    compute_coords(corner_lo, ic_lo, c.L);
    compute_coords(corner_hi, ic_hi, c.L);
    for (int j = 0; j < 3; j++) {
        ASSERT_GT(center[j], corner_lo[j]);
        ASSERT_LT(center[j], corner_hi[j]);
    }
    return true;
}

bool test_compute_center_is_midpoint() {
    setup_default_params();
    cube c = make_cube(2, 3, 1, 3);
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

bool test_compute_center_many_cubes() {
    setup_default_params();
    // All cubes at level 2 should have distinct centers
    set<int3> centers;
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
            for (int z = 0; z < 4; z++) {
                cube c = make_cube(x, y, z, 2);
                T coords[3];
                compute_center(coords, c);
                // Store as int3 (scaled) to check uniqueness
                int ix = (int)(coords[0] * 1000);
                int iy = (int)(coords[1] * 1000);
                int iz = (int)(coords[2] * 1000);
                centers.insert(make_int3(ix, iy, iz));
            }
    ASSERT_EQ((int)centers.size(), 64);
    return true;
}

bool test_compute_center_offset_params() {
    s_center[0] = 5.0;
    s_center[1] = -3.0;
    s_center[2] = 10.0;
    params::center = s_center;
    params::size = 8.0;
    cube c = make_cube(0, 0, 0, 1);
    T coords[3];
    compute_center(coords, c);
    // center[j] - size/2 + size*(0+0.5)/2
    // 5 - 4 + 8*0.5/2 = 5 - 4 + 2 = 3
    ASSERT_NEAR(coords[0], 3.0, 1e-10);
    // -3 - 4 + 2 = -5
    ASSERT_NEAR(coords[1], -5.0, 1e-10);
    // 10 - 4 + 2 = 8
    ASSERT_NEAR(coords[2], 8.0, 1e-10);
    setup_default_params();
    return true;
}

// ============================================================
// Adjacent cubes consistency
// ============================================================

bool test_adjacent_cubes_share_face() {
    setup_default_params();
    cube c1 = make_cube(0, 0, 0, 2);
    cube c2 = make_cube(1, 0, 0, 2);
    T r1[3], l2[3];
    int ic_r1[3] = {c1.coords[0] + 1, c1.coords[1], c1.coords[2]};
    int ic_l2[3] = {c2.coords[0], c2.coords[1], c2.coords[2]};
    compute_coords(r1, ic_r1, c1.L);
    compute_coords(l2, ic_l2, c2.L);
    ASSERT_NEAR(r1[0], l2[0], 1e-10);
    return true;
}

bool test_adjacent_cubes_all_axes() {
    setup_default_params();
    for (int axis = 0; axis < 3; axis++) {
        cube c1 = make_cube(0, 0, 0, 2);
        cube c2 = make_cube(0, 0, 0, 2);
        c2.coords[axis] = 1;

        int ic_face1[3] = {c1.coords[0], c1.coords[1], c1.coords[2]};
        ic_face1[axis] += 1;
        int ic_face2[3] = {c2.coords[0], c2.coords[1], c2.coords[2]};

        T f1[3], f2[3];
        compute_coords(f1, ic_face1, c1.L);
        compute_coords(f2, ic_face2, c2.L);
        ASSERT_NEAR(f1[axis], f2[axis], 1e-10);
    }
    return true;
}

bool test_cube_size_decreases_with_level() {
    setup_default_params();
    for (int L = 1; L <= 5; L++) {
        T c1[3], c2[3];
        int ic1[3] = {0, 0, 0};
        int ic2[3] = {1, 0, 0};
        compute_coords(c1, ic1, L);
        compute_coords(c2, ic2, L);
        T size_at_L = c2[0] - c1[0];
        T expected = params::size / (T)(1 << L);
        ASSERT_NEAR(size_at_L, expected, 1e-10);
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Coordinate Tests ===" << std::endl;

    TEST_SUITE("compute_coords");
    RUN_TEST(test_compute_coords_origin);
    RUN_TEST(test_compute_coords_center);
    RUN_TEST(test_compute_coords_far_corner);
    RUN_TEST(test_compute_coords_level2);
    RUN_TEST(test_compute_coords_level3);
    RUN_TEST(test_compute_coords_offset_center);
    RUN_TEST(test_compute_coords_monotonic);
    RUN_TEST(test_compute_coords_monotonic_all_axes);
    RUN_TEST(test_compute_coords_uniform_spacing);
    RUN_TEST(test_compute_coords_level4);
    RUN_TEST(test_compute_coords_asymmetric_size);

    TEST_SUITE("compute_center");
    RUN_TEST(test_compute_center_root);
    RUN_TEST(test_compute_center_level1_children);
    RUN_TEST(test_compute_center_level2);
    RUN_TEST(test_compute_center_within_cube);
    RUN_TEST(test_compute_center_is_midpoint);
    RUN_TEST(test_compute_center_many_cubes);
    RUN_TEST(test_compute_center_offset_params);

    TEST_SUITE("Adjacent Cubes");
    RUN_TEST(test_adjacent_cubes_share_face);
    RUN_TEST(test_adjacent_cubes_all_axes);
    RUN_TEST(test_cube_size_decreases_with_level);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
