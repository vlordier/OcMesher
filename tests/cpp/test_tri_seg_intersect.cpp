// Tests for triangle-segment intersection
// Build: g++ -std=c++11 -O2 -o test_tri_seg_intersect test_tri_seg_intersect.cpp -lm

#include "test_framework.h"

// ============================================================
// Basic intersection cases
// ============================================================

bool test_through_center() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, -1};
    T s2[] = {0.1, 0.1, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_reverse_direction() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, 1};
    T s2[] = {0.1, 0.1, -1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_near_vertex() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.01, 0.01, -1};
    T s2[] = {0.01, 0.01, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_large_triangle() {
    T t1[] = {-100, -100, 0};
    T t2[] = {100, -100, 0};
    T t3[] = {0, 100, 0};
    T s1[] = {0, 0, -50};
    T s2[] = {0, 0, 50};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// No intersection cases
// ============================================================

bool test_parallel_no_intersection() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0, 0, 1};
    T s2[] = {1, 0, 1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_same_side() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.1, 0.1, 1};
    T s2[] = {0.1, 0.1, 2};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_outside_triangle() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {5, 5, -1};
    T s2[] = {5, 5, 1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_segment_behind() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.2, 0.2, -3};
    T s2[] = {0.2, 0.2, -1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_segment_in_front() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.2, 0.2, 1};
    T s2[] = {0.2, 0.2, 3};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// Tilted / offset planes
// ============================================================

bool test_negative_z_plane() {
    T t1[] = {0, 0, -5};
    T t2[] = {1, 0, -5};
    T t3[] = {0, 1, -5};
    T s1[] = {0.2, 0.2, -10};
    T s2[] = {0.2, 0.2, 0};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_tilted_plane() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 1};
    T t3[] = {0, 1, 0.5};
    T s1[] = {0.2, 0.2, -1};
    T s2[] = {0.2, 0.2, 2};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_xy_plane_triangle() {
    T t1[] = {-1, -1, 3};
    T t2[] = {1, -1, 3};
    T t3[] = {0, 1, 3};
    T s1[] = {0, 0, 0};
    T s2[] = {0, 0, 6};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// Negative coordinate regions
// ============================================================

bool test_all_negative_coords() {
    T t1[] = {-3, -3, -5};
    T t2[] = {-1, -3, -5};
    T t3[] = {-2, -1, -5};
    T s1[] = {-2, -2, -10};
    T s2[] = {-2, -2, 0};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_all_negative_miss() {
    T t1[] = {-3, -3, -5};
    T t2[] = {-1, -3, -5};
    T t3[] = {-2, -1, -5};
    T s1[] = {5, 5, -10};
    T s2[] = {5, 5, 0};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// Edge cases
// ============================================================

bool test_very_small_triangle() {
    T t1[] = {0, 0, 0};
    T t2[] = {1e-6, 0, 0};
    T t3[] = {0, 1e-6, 0};
    T s1[] = {1e-7, 1e-7, -1};
    T s2[] = {1e-7, 1e-7, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_near_miss_outside_hypotenuse() {
    // The triangle is defined by t1=(0,0,0), t2=(1,0,0), t3=(0,1,0)
    // The hypotenuse is the line x+y=1
    // A point at (0.6, 0.6) is outside (x+y=1.2 > 1)
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.6, 0.6, -1};
    T s2[] = {0.6, 0.6, 1};
    ASSERT_FALSE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_near_hit_inside_hypotenuse() {
    // A point at (0.3, 0.3) is inside (x+y=0.6 < 1)
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.3, 0.3, -1};
    T s2[] = {0.3, 0.3, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_diagonal_segment() {
    // Segment goes diagonally through the triangle plane
    T t1[] = {0, 0, 0};
    T t2[] = {2, 0, 0};
    T t3[] = {0, 2, 0};
    T s1[] = {0.3, 0.3, -1};
    T s2[] = {0.3, 0.3, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_long_segment() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.2, 0.2, -1000};
    T s2[] = {0.2, 0.2, 1000};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_short_segment_intersecting() {
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 1, 0};
    T s1[] = {0.2, 0.2, -0.001};
    T s2[] = {0.2, 0.2, 0.001};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_equilateral_triangle() {
    T h = sqrt(3.0) / 2.0;
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0.5, h, 0};
    T s1[] = {0.3, 0.2, -1};
    T s2[] = {0.3, 0.2, 1};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_right_angle_triangle_yz_plane() {
    // Triangle in YZ plane
    T t1[] = {0, 0, 0};
    T t2[] = {0, 1, 0};
    T t3[] = {0, 0, 1};
    T s1[] = {-1, 0.2, 0.2};
    T s2[] = {1, 0.2, 0.2};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

bool test_right_angle_triangle_xz_plane() {
    // Triangle in XZ plane
    T t1[] = {0, 0, 0};
    T t2[] = {1, 0, 0};
    T t3[] = {0, 0, 1};
    T s1[] = {0.2, -1, 0.2};
    T s2[] = {0.2, 1, 0.2};
    ASSERT_TRUE(tri_seg_intersect(t1, t2, t3, s1, s2));
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Triangle-Segment Intersection Tests ===" << std::endl;

    TEST_SUITE("Basic Intersection");
    RUN_TEST(test_through_center);
    RUN_TEST(test_reverse_direction);
    RUN_TEST(test_near_vertex);
    RUN_TEST(test_large_triangle);

    TEST_SUITE("No Intersection");
    RUN_TEST(test_parallel_no_intersection);
    RUN_TEST(test_same_side);
    RUN_TEST(test_outside_triangle);
    RUN_TEST(test_segment_behind);
    RUN_TEST(test_segment_in_front);

    TEST_SUITE("Tilted / Offset Planes");
    RUN_TEST(test_negative_z_plane);
    RUN_TEST(test_tilted_plane);
    RUN_TEST(test_xy_plane_triangle);

    TEST_SUITE("Negative Coordinate Regions");
    RUN_TEST(test_all_negative_coords);
    RUN_TEST(test_all_negative_miss);

    TEST_SUITE("Edge Cases");
    RUN_TEST(test_very_small_triangle);
    RUN_TEST(test_near_miss_outside_hypotenuse);
    RUN_TEST(test_near_hit_inside_hypotenuse);
    RUN_TEST(test_diagonal_segment);
    RUN_TEST(test_long_segment);
    RUN_TEST(test_short_segment_intersecting);
    RUN_TEST(test_equilateral_triangle);
    RUN_TEST(test_right_angle_triangle_yz_plane);
    RUN_TEST(test_right_angle_triangle_xz_plane);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
