// Tests for int_log function
// Build: g++ -std=c++11 -O2 -o test_int_log test_int_log.cpp -lm

#include "test_framework.h"

// ============================================================
// Powers of two
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
    ASSERT_EQ(int_log(512.0), 9);
    ASSERT_EQ(int_log(1024.0), 10);
    return true;
}

// ============================================================
// Non-powers (should ceil to next power)
// ============================================================

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

// ============================================================
// Just above/below powers of two
// ============================================================

bool test_int_log_just_above_power() {
    // Just above 2^n should give n+1
    ASSERT_EQ(int_log(2.001), 2);
    ASSERT_EQ(int_log(4.001), 3);
    ASSERT_EQ(int_log(8.001), 4);
    ASSERT_EQ(int_log(1024.001), 11);
    return true;
}

bool test_int_log_just_below_power() {
    // Just below 2^n should give n (ceil of log2)
    ASSERT_EQ(int_log(1.999), 1);
    ASSERT_EQ(int_log(3.999), 2);
    ASSERT_EQ(int_log(7.999), 3);
    ASSERT_EQ(int_log(15.999), 4);
    return true;
}

// ============================================================
// Edge cases: fractional, zero, very small
// ============================================================

bool test_int_log_fractional() {
    ASSERT_EQ(int_log(0.5), 0);
    ASSERT_EQ(int_log(0.25), 0);
    ASSERT_EQ(int_log(0.1), 0);
    ASSERT_EQ(int_log(0.001), 0);
    ASSERT_EQ(int_log(1.5), 1);
    return true;
}

bool test_int_log_zero() {
    // log2(0) = -inf, ceil(-inf) = -inf, max(0, -inf) = 0
    ASSERT_EQ(int_log(0.0), 0);
    return true;
}

bool test_int_log_very_small_positive() {
    ASSERT_EQ(int_log(1e-10), 0);
    ASSERT_EQ(int_log(1e-100), 0);
    return true;
}

// ============================================================
// Large values
// ============================================================

bool test_int_log_large_values() {
    ASSERT_EQ(int_log(1048576.0), 20);  // 2^20
    ASSERT_EQ(int_log(1048577.0), 21);  // 2^20 + 1
    ASSERT_EQ(int_log(1000000.0), 20);
    return true;
}

bool test_int_log_very_large() {
    // 2^30 = 1073741824
    ASSERT_EQ(int_log(1073741824.0), 30);
    ASSERT_EQ(int_log(1073741825.0), 31);
    return true;
}

// ============================================================
// Negative values (edge case)
// ============================================================

bool test_int_log_negative() {
    // log2 of negative is NaN, ceil(NaN) is implementation-defined
    // max(0, NaN) should give 0 since NaN comparisons return false
    // This tests the function doesn't crash
    int result = int_log(-1.0);
    ASSERT_GE(result, 0);
    return true;
}

// ============================================================
// Monotonicity
// ============================================================

bool test_int_log_monotonic() {
    // int_log should be non-decreasing
    T prev = 0;
    for (T x = 0.5; x <= 1024; x *= 1.5) {
        int cur = int_log(x);
        ASSERT_GE(cur, (int)prev);
        prev = cur;
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== int_log Tests ===" << std::endl;

    TEST_SUITE("Powers of Two");
    RUN_TEST(test_int_log_powers_of_two);

    TEST_SUITE("Non-Powers");
    RUN_TEST(test_int_log_non_powers);

    TEST_SUITE("Near Powers");
    RUN_TEST(test_int_log_just_above_power);
    RUN_TEST(test_int_log_just_below_power);

    TEST_SUITE("Edge Cases");
    RUN_TEST(test_int_log_fractional);
    RUN_TEST(test_int_log_zero);
    RUN_TEST(test_int_log_very_small_positive);
    RUN_TEST(test_int_log_negative);

    TEST_SUITE("Large Values");
    RUN_TEST(test_int_log_large_values);
    RUN_TEST(test_int_log_very_large);

    TEST_SUITE("Monotonicity");
    RUN_TEST(test_int_log_monotonic);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
