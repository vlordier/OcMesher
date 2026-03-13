// Tests for C preprocessor macros defined in core.h
// Build: g++ -std=c++11 -O2 -o test_macros test_macros.cpp -lm

#include "test_framework.h"

// ============================================================
// cubex
// ============================================================

bool test_cubex_zero() {
    ASSERT_EQ(cubex(0), 0);
    return true;
}

bool test_cubex_one() {
    ASSERT_EQ(cubex(1), 1);
    return true;
}

bool test_cubex_small() {
    ASSERT_EQ(cubex(2), 8);
    ASSERT_EQ(cubex(3), 27);
    ASSERT_EQ(cubex(4), 64);
    ASSERT_EQ(cubex(5), 125);
    return true;
}

bool test_cubex_larger() {
    ASSERT_EQ(cubex(10), 1000);
    ASSERT_EQ(cubex(8), 512);
    return true;
}

bool test_cubex_negative() {
    // cubex(-1) = (-1)*(-1)*(-1) = -1
    ASSERT_EQ(cubex(-1), -1);
    ASSERT_EQ(cubex(-2), -8);
    return true;
}

// ============================================================
// cube_index
// ============================================================

bool test_cube_index_origin() {
    ASSERT_EQ(cube_index(0, 0, 0, 2), 0);
    ASSERT_EQ(cube_index(0, 0, 0, 3), 0);
    ASSERT_EQ(cube_index(0, 0, 0, 1), 0);
    return true;
}

bool test_cube_index_axes() {
    // x-axis: x*s*s
    ASSERT_EQ(cube_index(1, 0, 0, 2), 4);
    // y-axis: y*s
    ASSERT_EQ(cube_index(0, 1, 0, 2), 2);
    // z-axis: z
    ASSERT_EQ(cube_index(0, 0, 1, 2), 1);
    return true;
}

bool test_cube_index_corners() {
    ASSERT_EQ(cube_index(1, 1, 1, 2), 7);
    ASSERT_EQ(cube_index(2, 2, 2, 3), 26);
    return true;
}

bool test_cube_index_larger_strides() {
    ASSERT_EQ(cube_index(1, 0, 0, 4), 16);
    ASSERT_EQ(cube_index(0, 1, 0, 4), 4);
    ASSERT_EQ(cube_index(0, 0, 1, 4), 1);
    // Max corner of 4x4x4
    ASSERT_EQ(cube_index(3, 3, 3, 4), 63);
    return true;
}

bool test_cube_index_formula_consistency() {
    // Verify cube_index(x,y,z,s) == x*s*s + y*s + z for various values
    for (int s = 1; s <= 5; s++)
        for (int x = 0; x < s; x++)
            for (int y = 0; y < s; y++)
                for (int z = 0; z < s; z++)
                    ASSERT_EQ(cube_index(x, y, z, s), x * s * s + y * s + z);
    return true;
}

// ============================================================
// first_digit / second_digit
// ============================================================

bool test_first_digit_all_values() {
    ASSERT_EQ(first_digit(0), 0);
    ASSERT_EQ(first_digit(1), 1);
    ASSERT_EQ(first_digit(2), 0);
    ASSERT_EQ(first_digit(3), 1);
    ASSERT_EQ(first_digit(4), 0);
    ASSERT_EQ(first_digit(5), 1);
    ASSERT_EQ(first_digit(6), 0);
    ASSERT_EQ(first_digit(7), 1);
    return true;
}

bool test_second_digit_all_values() {
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

bool test_first_second_digit_bitfield_consistency() {
    // For octree child indices 0-7, first_digit gives bit 0,
    // second_digit gives bit 1
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(first_digit(i), (i >> 0) & 1);
        ASSERT_EQ(second_digit(i), (i >> 1) & 1);
    }
    return true;
}

bool test_first_digit_higher_bits_ignored() {
    // first_digit only looks at bit 0
    ASSERT_EQ(first_digit(8), 0);   // 1000 -> bit0 = 0
    ASSERT_EQ(first_digit(9), 1);   // 1001 -> bit0 = 1
    ASSERT_EQ(first_digit(16), 0);  // 10000 -> bit0 = 0
    ASSERT_EQ(first_digit(255), 1); // 11111111 -> bit0 = 1
    return true;
}

// ============================================================
// make_int3 / xpp / ypp / zpp
// ============================================================

bool test_make_int3_basic() {
    int3 v = make_int3(1, 2, 3);
    ASSERT_EQ(xpp(v), 1);
    ASSERT_EQ(ypp(v), 2);
    ASSERT_EQ(zpp(v), 3);
    return true;
}

bool test_make_int3_zeros() {
    int3 v = make_int3(0, 0, 0);
    ASSERT_EQ(xpp(v), 0);
    ASSERT_EQ(ypp(v), 0);
    ASSERT_EQ(zpp(v), 0);
    return true;
}

bool test_make_int3_negative() {
    int3 v = make_int3(-5, 10, -100);
    ASSERT_EQ(xpp(v), -5);
    ASSERT_EQ(ypp(v), 10);
    ASSERT_EQ(zpp(v), -100);
    return true;
}

bool test_make_int3_large_values() {
    int3 v = make_int3(1000000, 2000000, 3000000);
    ASSERT_EQ(xpp(v), 1000000);
    ASSERT_EQ(ypp(v), 2000000);
    ASSERT_EQ(zpp(v), 3000000);
    return true;
}

bool test_make_int3_ordering_in_set() {
    // int3 is a nested pair, so set ordering should be lexicographic
    set<int3> s;
    s.insert(make_int3(1, 2, 3));
    s.insert(make_int3(1, 2, 3));  // duplicate
    s.insert(make_int3(0, 0, 0));
    ASSERT_EQ((int)s.size(), 2);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Macro Tests ===" << std::endl;

    TEST_SUITE("cubex");
    RUN_TEST(test_cubex_zero);
    RUN_TEST(test_cubex_one);
    RUN_TEST(test_cubex_small);
    RUN_TEST(test_cubex_larger);
    RUN_TEST(test_cubex_negative);

    TEST_SUITE("cube_index");
    RUN_TEST(test_cube_index_origin);
    RUN_TEST(test_cube_index_axes);
    RUN_TEST(test_cube_index_corners);
    RUN_TEST(test_cube_index_larger_strides);
    RUN_TEST(test_cube_index_formula_consistency);

    TEST_SUITE("first_digit / second_digit");
    RUN_TEST(test_first_digit_all_values);
    RUN_TEST(test_second_digit_all_values);
    RUN_TEST(test_first_second_digit_bitfield_consistency);
    RUN_TEST(test_first_digit_higher_bits_ignored);

    TEST_SUITE("make_int3");
    RUN_TEST(test_make_int3_basic);
    RUN_TEST(test_make_int3_zeros);
    RUN_TEST(test_make_int3_negative);
    RUN_TEST(test_make_int3_large_values);
    RUN_TEST(test_make_int3_ordering_in_set);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
