// Tests for 3x3 determinant function
// Build: g++ -std=c++11 -O2 -o test_determinant test_determinant.cpp -lm

#include "test_framework.h"

// ============================================================
// Basic determinants
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

bool test_det_diagonal() {
    T m[3][3] = {{2, 0, 0}, {0, 3, 0}, {0, 0, 5}};
    ASSERT_NEAR(det(m), 30.0, 1e-10);
    return true;
}

bool test_det_negative_diagonal() {
    T m[3][3] = {{2, 0, 0}, {0, 3, 0}, {0, 0, -1}};
    ASSERT_NEAR(det(m), -6.0, 1e-10);
    return true;
}

bool test_det_known_value() {
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    // det = 1*(0-48) - 2*(0-42) + 3*(32-35) = -48+84-9 = 27
    ASSERT_NEAR(det(m), 27.0, 1e-10);
    return true;
}

// ============================================================
// Singular matrices (det = 0)
// ============================================================

bool test_det_singular_dependent_rows() {
    // row3 = row1 + row2
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {5, 7, 9}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_singular_duplicate_rows() {
    T m[3][3] = {{1, 2, 3}, {1, 2, 3}, {4, 5, 6}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_singular_proportional_rows() {
    T m[3][3] = {{1, 2, 3}, {2, 4, 6}, {7, 8, 9}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_singular_zero_row() {
    T m[3][3] = {{1, 2, 3}, {0, 0, 0}, {7, 8, 9}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_singular_zero_column() {
    T m[3][3] = {{0, 2, 3}, {0, 5, 6}, {0, 8, 9}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_antisymmetric() {
    T m[3][3] = {{0, 1, -1}, {-1, 0, 1}, {1, -1, 0}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

// ============================================================
// Mathematical properties
// ============================================================

bool test_det_scale_invariance() {
    T m1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    T d1 = det(m1);
    T k = 2.0;
    T m2[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) m2[i][j] = m1[i][j] * k;
    T d2 = det(m2);
    // det(kM) = k^3 * det(M)
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

bool test_det_row_swap_01_and_12() {
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    T d0 = det(m);

    // Swap rows 1 and 2
    T m12[3][3] = {{1, 2, 3}, {7, 8, 0}, {4, 5, 6}};
    T d12 = det(m12);
    ASSERT_NEAR(d0, -d12, 1e-10);

    // Swap rows 0 and 2
    T m02[3][3] = {{7, 8, 0}, {4, 5, 6}, {1, 2, 3}};
    T d02 = det(m02);
    ASSERT_NEAR(d0, -d02, 1e-10);
    return true;
}

bool test_det_row_scale() {
    T m[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
    T d1 = det(m);
    // Scale row 0 by 3
    T m2[3][3] = {{3, 6, 9}, {4, 5, 6}, {7, 8, 0}};
    T d2 = det(m2);
    ASSERT_NEAR(d2, 3.0 * d1, 1e-10);
    return true;
}

// ============================================================
// Edge cases with large/small values
// ============================================================

bool test_det_large_values() {
    T m[3][3] = {{1e6, 0, 0}, {0, 1e6, 0}, {0, 0, 1e6}};
    ASSERT_NEAR(det(m), 1e18, 1e8);
    return true;
}

bool test_det_small_values() {
    T m[3][3] = {{1e-6, 0, 0}, {0, 1e-6, 0}, {0, 0, 1e-6}};
    ASSERT_NEAR(det(m), 1e-18, 1e-28);
    return true;
}

bool test_det_mixed_sign() {
    T m[3][3] = {{-1, 2, -3}, {4, -5, 6}, {-7, 8, -9}};
    T d = det(m);
    // Just verify it's finite and consistent with transpose
    ASSERT_TRUE(std::isfinite(d));
    T mt[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) mt[i][j] = m[j][i];
    ASSERT_NEAR(d, det(mt), 1e-10);
    return true;
}

bool test_det_ones_matrix() {
    T m[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    ASSERT_NEAR(det(m), 0.0, 1e-10);
    return true;
}

bool test_det_permutation_matrix() {
    // Cyclic permutation: det = 1
    T m[3][3] = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    ASSERT_NEAR(det(m), 1.0, 1e-10);
    return true;
}

bool test_det_swap_permutation() {
    // Single swap: det = -1
    T m[3][3] = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
    ASSERT_NEAR(det(m), -1.0, 1e-10);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Determinant Tests ===" << std::endl;

    TEST_SUITE("Basic Determinants");
    RUN_TEST(test_det_identity);
    RUN_TEST(test_det_zero_matrix);
    RUN_TEST(test_det_diagonal);
    RUN_TEST(test_det_negative_diagonal);
    RUN_TEST(test_det_known_value);

    TEST_SUITE("Singular Matrices");
    RUN_TEST(test_det_singular_dependent_rows);
    RUN_TEST(test_det_singular_duplicate_rows);
    RUN_TEST(test_det_singular_proportional_rows);
    RUN_TEST(test_det_singular_zero_row);
    RUN_TEST(test_det_singular_zero_column);
    RUN_TEST(test_det_antisymmetric);

    TEST_SUITE("Mathematical Properties");
    RUN_TEST(test_det_scale_invariance);
    RUN_TEST(test_det_transpose_equality);
    RUN_TEST(test_det_row_swap_negation);
    RUN_TEST(test_det_row_swap_01_and_12);
    RUN_TEST(test_det_row_scale);

    TEST_SUITE("Edge Cases");
    RUN_TEST(test_det_large_values);
    RUN_TEST(test_det_small_values);
    RUN_TEST(test_det_mixed_sign);
    RUN_TEST(test_det_ones_matrix);
    RUN_TEST(test_det_permutation_matrix);
    RUN_TEST(test_det_swap_permutation);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
