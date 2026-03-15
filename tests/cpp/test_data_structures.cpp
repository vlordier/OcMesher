// Tests for data structures: cube, node, computed_vertex, key conversions
// Build: g++ -std=c++11 -O2 -o test_data_structures test_data_structures.cpp -lm

#include "test_framework.h"

// ============================================================
// Leaf node marking
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

// ============================================================
// Grid node marking
// ============================================================

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

bool test_grid_node_level_roundtrip() {
    // Verify mark_grid_node and grid_node_level are inverses for many levels
    for (int level = 0; level <= 20; level++) {
        node n;
        mark_grid_node(n, level);
        ASSERT_TRUE(leaf_node(n));
        ASSERT_EQ(grid_node_level(n), level);
    }
    return true;
}

// ============================================================
// Cube status: regular / boundary / exterior
// ============================================================

bool test_cube_status_regular() {
    cube c;
    c.L = 0;
    ASSERT_TRUE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));

    c.L = 5;
    ASSERT_TRUE(is_regular(c));
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

bool test_cube_status_mutually_exclusive() {
    // Boundary and exterior are mutually exclusive with regular
    cube c;
    c.L = 0;
    ASSERT_TRUE(is_regular(c));

    mark_boundary(c);
    ASSERT_FALSE(is_regular(c));
    ASSERT_TRUE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));

    // Re-mark as exterior
    mark_exterior(c);
    ASSERT_FALSE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_TRUE(is_exterior(c));

    // Set back to regular
    c.L = 10;
    ASSERT_TRUE(is_regular(c));
    ASSERT_FALSE(is_boundary(c));
    ASSERT_FALSE(is_exterior(c));
    return true;
}

bool test_cube_status_level_zero_is_regular() {
    cube c;
    c.L = 0;
    ASSERT_TRUE(is_regular(c));
    return true;
}

// ============================================================
// Key conversions
// ============================================================

bool test_cube_key_conversion() {
    cube c = make_cube(1, 2, 3, 4);
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
        cube c = make_cube(test_coords[t][0], test_coords[t][1],
                           test_coords[t][2], test_coords[t][3]);
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
    cube c1 = make_cube(0, 0, 0, 1);
    cube c2 = make_cube(1, 0, 0, 1);
    ASSERT_TRUE(cube_to_key(c1) != cube_to_key(c2));

    // Same coords different level
    cube c3 = make_cube(0, 0, 0, 2);
    ASSERT_TRUE(cube_to_key(c1) != cube_to_key(c3));
    return true;
}

bool test_cube_key_in_set() {
    set<key_cube> s;
    cube c1 = make_cube(1, 2, 3, 4);
    cube c2 = make_cube(1, 2, 3, 4);  // same
    cube c3 = make_cube(1, 2, 3, 5);  // different level
    s.insert(cube_to_key(c1));
    s.insert(cube_to_key(c2));
    s.insert(cube_to_key(c3));
    ASSERT_EQ((int)s.size(), 2);
    return true;
}

bool test_cube_key_in_map() {
    map<key_cube, int> m;
    cube c = make_cube(5, 6, 7, 3);
    m[cube_to_key(c)] = 42;
    ASSERT_EQ(m[cube_to_key(c)], 42);
    ASSERT_EQ((int)m.size(), 1);
    return true;
}

bool test_cube_key_negative_coords() {
    // Negative coords should roundtrip correctly
    cube c = make_cube(-1, -2, -3, 5);
    key_cube key = cube_to_key(c);
    cube c2;
    key_to_cube(c2, key);
    ASSERT_EQ(c2.coords[0], -1);
    ASSERT_EQ(c2.coords[1], -2);
    ASSERT_EQ(c2.coords[2], -3);
    ASSERT_EQ(c2.L, 5);
    return true;
}

// ============================================================
// Struct init
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

bool test_computed_vertex_midpoint() {
    computed_vertex cv;
    cv.c[0] = 10.0;
    cv.c[1] = 20.0;
    cv.c[2] = 30.0;
    cv.l = 0.25;
    cv.r = 0.75;
    T mid = (cv.l + cv.r) / 2.0;
    ASSERT_NEAR(mid, 0.5, 1e-10);
    return true;
}

bool test_cube_struct_init() {
    cube c = make_cube(5, 10, 15, 3);
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

bool test_node_all_nxts_minus_one() {
    node n = make_partial_root();
    for (int i = 0; i < 8; i++)
        ASSERT_EQ(n.nxts[i], -1);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Data Structure Tests ===" << std::endl;

    TEST_SUITE("Leaf Node Marking");
    RUN_TEST(test_leaf_node_marking);

    TEST_SUITE("Grid Node Marking");
    RUN_TEST(test_grid_node_marking);
    RUN_TEST(test_grid_node_level_roundtrip);

    TEST_SUITE("Cube Status");
    RUN_TEST(test_cube_status_regular);
    RUN_TEST(test_cube_status_boundary);
    RUN_TEST(test_cube_status_exterior);
    RUN_TEST(test_cube_status_mutually_exclusive);
    RUN_TEST(test_cube_status_level_zero_is_regular);

    TEST_SUITE("Key Conversions");
    RUN_TEST(test_cube_key_conversion);
    RUN_TEST(test_cube_key_roundtrip_various);
    RUN_TEST(test_cube_key_uniqueness);
    RUN_TEST(test_cube_key_in_set);
    RUN_TEST(test_cube_key_in_map);
    RUN_TEST(test_cube_key_negative_coords);

    TEST_SUITE("Structs");
    RUN_TEST(test_computed_vertex_init);
    RUN_TEST(test_computed_vertex_midpoint);
    RUN_TEST(test_cube_struct_init);
    RUN_TEST(test_node_struct_init);
    RUN_TEST(test_node_all_nxts_minus_one);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
