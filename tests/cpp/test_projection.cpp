// Tests for projected_coords and projected_size
// Build: g++ -std=c++11 -O2 -o test_projection test_projection.cpp -lm

#include "test_framework.h"

// ============================================================
// projected_coords: distance
// ============================================================

bool test_projected_coords_distance_at_origin() {
    setup_simple_camera();
    cube c = make_cube(0, 0, 0, 0);
    T r;
    projected_coords(c, 0, NULL, &r);
    // Cube center at (0,0,0), camera at origin, distance=0 clamped to min_dist=1
    ASSERT_NEAR(r, 1.0, 1e-10);
    return true;
}

bool test_projected_coords_distance_offset_cube() {
    setup_simple_camera();
    cube c = make_cube(1, 1, 1, 1);
    T r;
    projected_coords(c, 0, NULL, &r);
    // Center at (0.5,0.5,0.5), dist = sqrt(0.75) ≈ 0.866, clamped to min_dist=1
    ASSERT_NEAR(r, 1.0, 1e-10);
    return true;
}

bool test_projected_coords_distance_far_cube() {
    setup_simple_camera();
    params::min_dist = 0.01;  // Lower min_dist for this test
    // Use level 0, root cube -> center at (0,0,0)
    // We need a cube far from origin. At level 1, (1,1,1) -> center (0.5,0.5,0.5)
    // That's still close. The distance should be max(dist, 0.01)
    cube c = make_cube(1, 1, 1, 1);
    T r;
    projected_coords(c, 0, NULL, &r);
    // dist = sqrt(0.75) ≈ 0.866 > 0.01 so r = 0.866...
    ASSERT_GT(r, 0.5);
    setup_default_params();
    return true;
}

// ============================================================
// projected_coords: image coordinates
// ============================================================

bool test_projected_coords_image_coords() {
    setup_simple_camera();
    cube c = make_cube(1, 1, 1, 1);
    T icoords[3], r;
    projected_coords(c, 0, icoords, &r);
    // Cube center = (0.5, 0.5, 0.5), identity inv_pose -> Pc = (0.5, 0.5, 0.5)
    // K = [500,0,320; 0,500,240; 0,0,1]
    // icoords[2] = Pc[0]*0 + Pc[1]*0 + Pc[2]*1 = 0.5
    // icoords[0] = (Pc[0]*500 + Pc[1]*0 + Pc[2]*320) / icoords[2]
    //            = (250 + 160) / 0.5 = 820
    // icoords[1] = (Pc[0]*0 + Pc[1]*500 + Pc[2]*240) / 0.5
    //            = (250 + 120) / 0.5 = 740
    ASSERT_NEAR(icoords[0], 820.0, 1e-5);
    ASSERT_NEAR(icoords[1], 740.0, 1e-5);
    return true;
}

bool test_projected_coords_null_icoords() {
    setup_simple_camera();
    cube c = make_cube(0, 0, 0, 0);
    T r;
    projected_coords(c, 0, NULL, &r);
    ASSERT_NEAR(r, 1.0, 1e-10);
    return true;
}

bool test_projected_coords_null_r() {
    setup_simple_camera();
    cube c = make_cube(1, 1, 1, 1);
    T icoords[3];
    projected_coords(c, 0, icoords, NULL);
    ASSERT_TRUE(std::isfinite(icoords[0]));
    ASSERT_TRUE(std::isfinite(icoords[1]));
    return true;
}

bool test_projected_coords_different_cubes() {
    setup_simple_camera();
    // Use cubes that project to clearly different positions
    // Cube at (-0.5,-0.5,-0.5) and cube at (0.5,0.5,0.5) with camera at origin
    cube c1 = make_cube(0, 0, 0, 1);  // center (-0.5,-0.5,-0.5)
    cube c2 = make_cube(1, 1, 1, 1);  // center (0.5,0.5,0.5)
    T r1, r2;
    projected_coords(c1, 0, NULL, &r1);
    projected_coords(c2, 0, NULL, &r2);
    // Both cubes are at the same distance from origin but in opposite octants
    // Their distances should both be clamped to min_dist=1 or equal
    ASSERT_GT(r1, 0.0);
    ASSERT_GT(r2, 0.0);
    return true;
}

// ============================================================
// projected_size
// ============================================================

bool test_projected_size_positive() {
    setup_simple_camera();
    cube c = make_cube(0, 0, 0, 0);
    T size = projected_size(c, 0);
    ASSERT_GT(size, 0.0);
    return true;
}

bool test_projected_size_deeper_smaller() {
    setup_simple_camera();
    cube c1 = make_cube(0, 0, 0, 1);
    cube c2 = make_cube(0, 0, 0, 2);
    T size1 = projected_size(c1, 0);
    T size2 = projected_size(c2, 0);
    ASSERT_GT(size1, size2);
    return true;
}

bool test_projected_size_halves_per_level() {
    setup_simple_camera();
    // Same spatial location, deeper level -> roughly half the projected size
    // (not exactly because center positions differ slightly)
    cube c1 = make_cube(0, 0, 0, 1);
    cube c2 = make_cube(0, 0, 0, 2);
    T size1 = projected_size(c1, 0);
    T size2 = projected_size(c2, 0);
    T ratio = size1 / size2;
    // Should be close to 2 (within a factor since centers differ)
    ASSERT_GT(ratio, 1.5);
    ASSERT_LT(ratio, 3.0);
    return true;
}

bool test_projected_size_max_over_cameras() {
    setup_simple_camera();
    params::n_cams = 2;
    memcpy(s_cams + 23, s_cams, 23 * sizeof(T));
    s_cams[23 + 3] = 1.0;  // translate second camera

    cube c = make_cube(0, 0, 0, 1);
    T max_size = projected_size(c);
    T size0 = projected_size(c, 0);
    T size1 = projected_size(c, 1);
    T expected_max = size0 > size1 ? size0 : size1;
    ASSERT_NEAR(max_size, expected_max, 1e-10);
    return true;
}

bool test_projected_size_single_camera_consistency() {
    setup_simple_camera();
    cube c = make_cube(0, 0, 0, 1);
    T single = projected_size(c, 0);
    T maximum = projected_size(c);
    ASSERT_NEAR(single, maximum, 1e-10);
    return true;
}

bool test_projected_size_zero_cameras() {
    setup_default_params();
    params::n_cams = 0;
    cube c = make_cube(0, 0, 0, 1);
    T size = projected_size(c);
    // With zero cameras, max_size starts at 0 and is never updated
    ASSERT_NEAR(size, 0.0, 1e-10);
    return true;
}

bool test_projected_size_multiple_levels() {
    setup_simple_camera();
    T prev_size = 1e30;
    // Check projected size decreases with level for same octant
    for (int L = 1; L <= 5; L++) {
        cube c = make_cube(0, 0, 0, L);
        T s = projected_size(c, 0);
        ASSERT_GT(s, 0.0);
        ASSERT_LT(s, prev_size);
        prev_size = s;
    }
    return true;
}

// ============================================================
// Camera setup edge cases
// ============================================================

bool test_projected_coords_translated_camera() {
    setup_simple_camera();
    // Translate camera by (0, 0, 5) in the inverse pose
    s_cams[3] = 0;
    s_cams[7] = 0;
    s_cams[11] = 5.0;  // z-translation

    cube c = make_cube(0, 0, 0, 0);  // center at origin
    T r;
    projected_coords(c, 0, NULL, &r);
    // Pc = inv_pose * Pw, with Pw = (0,0,0) -> Pc = (0, 0, 5)
    // r = |Pc| = 5
    ASSERT_NEAR(r, 5.0, 1e-10);
    return true;
}

bool test_projected_size_with_different_focal_length() {
    setup_simple_camera();
    cube c = make_cube(0, 0, 0, 1);
    T size_500 = projected_size(c, 0);

    // Double the focal length (narrower FOV -> larger projected size)
    s_cams[12] = 1000.0;
    T size_1000 = projected_size(c, 0);
    ASSERT_GT(size_1000, size_500);

    setup_simple_camera();  // restore
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Projection Tests ===" << std::endl;

    TEST_SUITE("projected_coords: distance");
    RUN_TEST(test_projected_coords_distance_at_origin);
    RUN_TEST(test_projected_coords_distance_offset_cube);
    RUN_TEST(test_projected_coords_distance_far_cube);

    TEST_SUITE("projected_coords: image coordinates");
    RUN_TEST(test_projected_coords_image_coords);
    RUN_TEST(test_projected_coords_null_icoords);
    RUN_TEST(test_projected_coords_null_r);
    RUN_TEST(test_projected_coords_different_cubes);

    TEST_SUITE("projected_size");
    RUN_TEST(test_projected_size_positive);
    RUN_TEST(test_projected_size_deeper_smaller);
    RUN_TEST(test_projected_size_halves_per_level);
    RUN_TEST(test_projected_size_max_over_cameras);
    RUN_TEST(test_projected_size_single_camera_consistency);
    RUN_TEST(test_projected_size_zero_cameras);
    RUN_TEST(test_projected_size_multiple_levels);

    TEST_SUITE("Camera Edge Cases");
    RUN_TEST(test_projected_coords_translated_camera);
    RUN_TEST(test_projected_size_with_different_focal_length);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
