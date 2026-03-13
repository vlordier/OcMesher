// Tests for octree operations: expand, partial_expand, search, divide, enumerate
// Build: g++ -std=c++11 -O2 -o test_octree test_octree.cpp -lm

#include "test_framework.h"

// ============================================================
// expand_octree
// ============================================================

bool test_expand_root() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
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

bool test_expand_child_coords() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(nodes[1 + i].c.coords[0], (i >> 0) & 1);
        ASSERT_EQ(nodes[1 + i].c.coords[1], (i >> 1) & 1);
        ASSERT_EQ(nodes[1 + i].c.coords[2], (i >> 2) & 1);
    }
    return true;
}

bool test_expand_grandchildren() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);
    expand_octree(nodes, 1);

    ASSERT_EQ((int)nodes.size(), 17);
    ASSERT_FALSE(leaf_node(nodes[1]));

    for (int i = 9; i < 17; i++) {
        ASSERT_TRUE(leaf_node(nodes[i]));
        ASSERT_EQ(nodes[i].c.L, 2);
    }
    ASSERT_EQ(nodes[9].c.coords[0], 0);
    ASSERT_EQ(nodes[9].c.coords[1], 0);
    ASSERT_EQ(nodes[9].c.coords[2], 0);
    return true;
}

bool test_expand_preserves_siblings() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
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

bool test_expand_all_children() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    for (int i = 1; i <= 8; i++) expand_octree(nodes, i);

    ASSERT_EQ((int)nodes.size(), 9 + 64);
    for (int i = 9; i < 73; i++) {
        ASSERT_TRUE(leaf_node(nodes[i]));
        ASSERT_EQ(nodes[i].c.L, 2);
    }
    return true;
}

bool test_expand_spatial_coverage() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
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

bool test_expand_deep_tree() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());

    int current = 0;
    for (int depth = 0; depth < 5; depth++) {
        expand_octree(nodes, current);
        current = nodes[current].nxts[0];
    }
    ASSERT_TRUE(leaf_node(nodes[current]));
    ASSERT_EQ(nodes[current].c.L, 5);
    ASSERT_EQ(nodes[current].c.coords[0], 0);
    ASSERT_EQ(nodes[current].c.coords[1], 0);
    ASSERT_EQ(nodes[current].c.coords[2], 0);
    return true;
}

bool test_expand_non_zero_child_path() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());

    int current = 0;
    for (int depth = 0; depth < 3; depth++) {
        expand_octree(nodes, current);
        current = nodes[current].nxts[7];
    }
    ASSERT_TRUE(leaf_node(nodes[current]));
    ASSERT_EQ(nodes[current].c.L, 3);
    ASSERT_EQ(nodes[current].c.coords[0], 7);
    ASSERT_EQ(nodes[current].c.coords[1], 7);
    ASSERT_EQ(nodes[current].c.coords[2], 7);
    return true;
}

// ============================================================
// partial_expand_octree
// ============================================================

bool test_partial_expand_single() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    partial_expand_octree(nodes, 0, 1);
    ASSERT_EQ((int)nodes.size(), 2);
    ASSERT_EQ(nodes[0].nxts[0], 1);
    for (int i = 1; i < 8; i++) ASSERT_EQ(nodes[0].nxts[i], -1);
    return true;
}

bool test_partial_expand_multiple() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    partial_expand_octree(nodes, 0, 7);
    ASSERT_EQ((int)nodes.size(), 4);
    ASSERT_NE(nodes[0].nxts[0], -1);
    ASSERT_NE(nodes[0].nxts[1], -1);
    ASSERT_NE(nodes[0].nxts[2], -1);
    for (int i = 3; i < 8; i++) ASSERT_EQ(nodes[0].nxts[i], -1);
    return true;
}

bool test_partial_expand_idempotent() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

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
    nodes.push_back(make_partial_root());

    partial_expand_octree(nodes, 0, 0xFF);
    ASSERT_EQ((int)nodes.size(), 9);
    for (int i = 0; i < 8; i++) {
        ASSERT_GE(nodes[0].nxts[i], 1);
    }
    return true;
}

bool test_partial_expand_child_coords() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

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

bool test_partial_expand_incremental() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    // Expand one child at a time
    for (int i = 0; i < 8; i++) {
        partial_expand_octree(nodes, 0, 1 << i);
        ASSERT_NE(nodes[0].nxts[i], -1);
    }
    ASSERT_EQ((int)nodes.size(), 9);
    return true;
}

bool test_partial_expand_children_have_minus_one_nxts() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    partial_expand_octree(nodes, 0, 0xFF);

    for (int i = 0; i < 8; i++) {
        int child = nodes[0].nxts[i];
        for (int j = 0; j < 8; j++) {
            ASSERT_EQ(nodes[child].nxts[j], -1);
        }
    }
    return true;
}

// ============================================================
// enumerate_vertices
// ============================================================

bool test_enumerate_vertices_level0_count() {
    setup_default_params();
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    vertex v[8];
    enumerate_vertices(v, n);

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

    vertex v[27];
    enumerate_vertices(v, n);

    ASSERT_EQ(v[0].L, 0);
    ASSERT_EQ(v[0].coords[0], 0);
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

    for (int i = 0; i < 27; i++) {
        ASSERT_GE(v[i].L, 0);
    }
    return true;
}

bool test_enumerate_vertices_unique_keys() {
    setup_default_params();
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 1;

    vertex v[8];
    enumerate_vertices(v, n);

    set<key_cube> keys;
    for (int i = 0; i < 8; i++) {
        keys.insert(cube_to_key(v[i]));
    }
    ASSERT_EQ((int)keys.size(), 8);
    return true;
}

bool test_enumerate_vertices_level1_unique_keys() {
    setup_default_params();
    node n;
    mark_grid_node(n, 1);
    n.c.coords[0] = 0;
    n.c.coords[1] = 0;
    n.c.coords[2] = 0;
    n.c.L = 0;

    vertex v[27];
    enumerate_vertices(v, n);

    set<key_cube> keys;
    for (int i = 0; i < 27; i++) {
        keys.insert(cube_to_key(v[i]));
    }
    ASSERT_EQ((int)keys.size(), 27);
    return true;
}

bool test_enumerate_vertices_level0_offset_node() {
    setup_default_params();
    node n;
    mark_grid_node(n, 0);
    n.c.coords[0] = 1;
    n.c.coords[1] = 1;
    n.c.coords[2] = 1;
    n.c.L = 1;

    vertex v[8];
    enumerate_vertices(v, n);

    // First vertex should be at (1*1,1*1,1*1) = (1,1,1), promoted to L=0
    ASSERT_EQ(v[0].coords[0], 1);
    ASSERT_EQ(v[0].coords[1], 1);
    ASSERT_EQ(v[0].coords[2], 1);
    // (1,1,1) at L=1 -> all odd, can't be promoted
    ASSERT_EQ(v[0].L, 1);
    return true;
}

// ============================================================
// search
// ============================================================

bool test_search_out_of_bounds_zero() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    int coords[3] = {0, 0, 0};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

bool test_search_out_of_bounds_max() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    int coords[3] = {2, 1, 1};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

bool test_search_valid_interior() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
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
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    int coords[3] = {-1, 1, 1};
    cube result = search(&nodes[0], coords, 1);
    ASSERT_TRUE(is_boundary(result));
    return true;
}

bool test_search_all_boundary_coords() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    // All coords at boundary (0 or max) should be boundary
    int max_coord = 1 << 1;  // L=1 -> max=2
    int boundary_coords[][3] = {
        {0, 1, 1},
        {max_coord, 1, 1},
        {1, 0, 1},
        {1, max_coord, 1},
        {1, 1, 0},
        {1, 1, max_coord},
    };
    for (int i = 0; i < 6; i++) {
        cube result = search(&nodes[0], boundary_coords[i], 1);
        ASSERT_TRUE(is_boundary(result));
    }
    return true;
}

bool test_search_include_exterior_flag() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    // Only expand child 0
    partial_expand_octree(nodes, 0, 1);

    // Search for a point that's in an unexpanded child
    // At level 2, coords (3,1,1) should be in child 1 (x=1)
    int coords[3] = {3, 1, 1};
    cube result = search(&nodes[0], coords, 2);
    ASSERT_TRUE(is_exterior(result));

    // With include_exterior, we should get the cube back
    cube result2 = search(&nodes[0], coords, 2, true);
    ASSERT_TRUE(is_regular(result2));
    ASSERT_EQ(result2.L, 1);
    return true;
}

// ============================================================
// divide_to_cube
// ============================================================

bool test_divide_to_cube_level1() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    cube target = make_cube(0, 0, 0, 1);
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
    nodes.push_back(make_partial_root());

    cube target = make_cube(1, 1, 1, 2);
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
    nodes.push_back(make_partial_root());

    cube target = make_cube(3, 3, 3, 3);
    int node_id = divide_to_cube(nodes, target);
    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.L, 3);
    ASSERT_GT((int)nodes.size(), 2);
    return true;
}

bool test_divide_multiple_cubes() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    cube t1 = make_cube(0, 0, 0, 2);
    cube t2 = make_cube(3, 3, 3, 2);
    int id1 = divide_to_cube(nodes, t1);
    int id2 = divide_to_cube(nodes, t2);

    ASSERT_TRUE(leaf_node(nodes[id1]));
    ASSERT_TRUE(leaf_node(nodes[id2]));
    ASSERT_NE(id1, id2);
    ASSERT_EQ(nodes[id1].c.coords[0], 0);
    ASSERT_EQ(nodes[id2].c.coords[0], 3);
    return true;
}

bool test_divide_to_all_level1_cubes() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++)
            for (int z = 0; z < 2; z++) {
                cube target = make_cube(x, y, z, 1);
                int node_id = divide_to_cube(nodes, target);
                ASSERT_TRUE(leaf_node(nodes[node_id]));
                ASSERT_EQ(nodes[node_id].c.coords[0], x);
                ASSERT_EQ(nodes[node_id].c.coords[1], y);
                ASSERT_EQ(nodes[node_id].c.coords[2], z);
            }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Octree Tests ===" << std::endl;

    TEST_SUITE("expand_octree");
    RUN_TEST(test_expand_root);
    RUN_TEST(test_expand_child_coords);
    RUN_TEST(test_expand_grandchildren);
    RUN_TEST(test_expand_preserves_siblings);
    RUN_TEST(test_expand_all_children);
    RUN_TEST(test_expand_spatial_coverage);
    RUN_TEST(test_expand_deep_tree);
    RUN_TEST(test_expand_non_zero_child_path);

    TEST_SUITE("partial_expand_octree");
    RUN_TEST(test_partial_expand_single);
    RUN_TEST(test_partial_expand_multiple);
    RUN_TEST(test_partial_expand_idempotent);
    RUN_TEST(test_partial_expand_all_children);
    RUN_TEST(test_partial_expand_child_coords);
    RUN_TEST(test_partial_expand_incremental);
    RUN_TEST(test_partial_expand_children_have_minus_one_nxts);

    TEST_SUITE("enumerate_vertices");
    RUN_TEST(test_enumerate_vertices_level0_count);
    RUN_TEST(test_enumerate_vertices_level0_coords);
    RUN_TEST(test_enumerate_vertices_level1);
    RUN_TEST(test_enumerate_vertices_level1_all_valid);
    RUN_TEST(test_enumerate_vertices_unique_keys);
    RUN_TEST(test_enumerate_vertices_level1_unique_keys);
    RUN_TEST(test_enumerate_vertices_level0_offset_node);

    TEST_SUITE("search");
    RUN_TEST(test_search_out_of_bounds_zero);
    RUN_TEST(test_search_out_of_bounds_max);
    RUN_TEST(test_search_valid_interior);
    RUN_TEST(test_search_negative_coords);
    RUN_TEST(test_search_all_boundary_coords);
    RUN_TEST(test_search_include_exterior_flag);

    TEST_SUITE("divide_to_cube");
    RUN_TEST(test_divide_to_cube_level1);
    RUN_TEST(test_divide_to_cube_level2);
    RUN_TEST(test_divide_to_cube_creates_path);
    RUN_TEST(test_divide_multiple_cubes);
    RUN_TEST(test_divide_to_all_level1_cubes);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
