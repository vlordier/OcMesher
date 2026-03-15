// Integration tests combining multiple C++ core components
// Build: g++ -std=c++11 -O2 -o test_integration test_integration.cpp -lm

#include "test_framework.h"

// ============================================================
// Octree build and query
// ============================================================

bool test_octree_build_and_search() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    for (int i = 1; i <= 8; i++) mark_grid_node(nodes[i], 0);

    int found_count = 0;
    for (int x = 1; x < 4; x++)
        for (int y = 1; y < 4; y++)
            for (int z = 1; z < 4; z++) {
                int coords[3] = {x, y, z};
                cube result = search(&nodes[0], coords, 3);
                if (is_regular(result)) found_count++;
            }
    ASSERT_GT(found_count, 0);
    return true;
}

bool test_octree_divide_and_search() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    cube target = make_cube(1, 1, 1, 2);
    int node_id = divide_to_cube(nodes, target);

    ASSERT_TRUE(leaf_node(nodes[node_id]));
    ASSERT_EQ(nodes[node_id].c.L, 2);
    return true;
}

// ============================================================
// Coordinate system consistency
// ============================================================

bool test_center_is_midpoint_of_corners() {
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

// ============================================================
// Full expansion + vertex enumeration + edges
// ============================================================

bool test_full_pipeline_uniform_sdf() {
    setup_default_params();
    params::n_elements = 1;

    // Build octree
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    // Mark all children as grid nodes
    for (int i = 1; i <= 8; i++) mark_grid_node(nodes[i], 0);

    // For each child, enumerate vertices and check find_edges with uniform SDF
    for (int child = 1; child <= 8; child++) {
        vertex v[8];
        enumerate_vertices(v, nodes[child]);

        map<key_cube, int> vmap;
        for (int j = 0; j < 8; j++) vmap[cube_to_key(v[j])] = j;

        sdfT sdf[8];
        for (int j = 0; j < 8; j++) sdf[j] = 1.0f;

        vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
        find_edges(nodes[child], vmap, sdf, bipolar_edges);

        for (int j = 0; j <= params::n_elements; j++)
            ASSERT_EQ((int)bipolar_edges[j].size(), 0);
    }
    return true;
}

bool test_full_pipeline_with_crossing() {
    setup_default_params();
    params::n_elements = 1;

    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);
    mark_grid_node(nodes[1], 0);

    vertex v[8];
    enumerate_vertices(v, nodes[1]);

    map<key_cube, int> vmap;
    for (int i = 0; i < 8; i++) vmap[cube_to_key(v[i])] = i;

    // One negative vertex -> 3 bipolar edges
    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = 1.0f;
    sdf[0] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(nodes[1], vmap, sdf, bipolar_edges);

    ASSERT_EQ((int)bipolar_edges[0].size(), 3);
    return true;
}

// ============================================================
// Deep octree with projection
// ============================================================

bool test_deep_octree_projected_sizes_decrease() {
    setup_simple_camera();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());

    // Build a path going deeper
    int current = 0;
    T prev_size = 1e30;
    for (int depth = 0; depth < 4; depth++) {
        expand_octree(nodes, current);
        current = nodes[current].nxts[0];

        T ps = projected_size(nodes[current].c, 0);
        ASSERT_GT(ps, 0.0);
        ASSERT_LT(ps, prev_size);
        prev_size = ps;
    }
    return true;
}

bool test_projection_consistency_across_children() {
    setup_simple_camera();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    // All 8 children should have positive projected sizes
    for (int i = 1; i <= 8; i++) {
        T ps = projected_size(nodes[i].c, 0);
        ASSERT_GT(ps, 0.0);
    }
    return true;
}

// ============================================================
// Divide + search consistency
// ============================================================

bool test_divide_then_search_multiple_cubes() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_partial_root());

    // Divide to create several cubes at level 2
    cube targets[4] = {
        make_cube(0, 0, 0, 2),
        make_cube(1, 0, 0, 2),
        make_cube(0, 1, 0, 2),
        make_cube(0, 0, 1, 2),
    };

    int ids[4];
    for (int i = 0; i < 4; i++) {
        ids[i] = divide_to_cube(nodes, targets[i]);
        ASSERT_TRUE(leaf_node(nodes[ids[i]]));
    }

    // All IDs should be different
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            ASSERT_NE(ids[i], ids[j]);
    return true;
}

// ============================================================
// Key conversion + vertex enumeration
// ============================================================

bool test_vertex_keys_are_consistent() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);
    mark_grid_node(nodes[1], 0);
    mark_grid_node(nodes[2], 0);

    vertex v1[8], v2[8];
    enumerate_vertices(v1, nodes[1]);
    enumerate_vertices(v2, nodes[2]);

    // Verify that shared vertices between adjacent children have the same key
    // Nodes 1 and 2: child 0 = (0,0,0) and child 1 = (1,0,0)
    // They share the face at x=1 (in level-1 coords)
    set<key_cube> keys1, keys2;
    for (int i = 0; i < 8; i++) {
        keys1.insert(cube_to_key(v1[i]));
        keys2.insert(cube_to_key(v2[i]));
    }

    // Count shared keys
    int shared = 0;
    for (set<key_cube>::iterator it = keys1.begin(); it != keys1.end(); ++it) {
        if (keys2.count(*it)) shared++;
    }
    // Adjacent cubes at same level share 4 vertices on their common face
    ASSERT_EQ(shared, 4);
    return true;
}

// ============================================================
// Stress: large octree
// ============================================================

bool test_large_octree_node_count() {
    setup_default_params();
    vector<node> nodes;
    nodes.push_back(make_root_leaf());
    expand_octree(nodes, 0);

    // Expand all level-1 children
    for (int i = 1; i <= 8; i++) expand_octree(nodes, i);

    // Expand all level-2 children of child 0
    for (int i = 0; i < 8; i++) {
        int child_id = nodes[1].nxts[i];
        expand_octree(nodes, child_id);
    }

    // Should have: 1 + 8 + 64 + 64 = 137 nodes
    // (root + 8 children + 64 grandchildren + 64 great-grandchildren of child 0)
    ASSERT_EQ((int)nodes.size(), 1 + 8 + 64 + 64);

    // All great-grandchildren should be leaves at level 3
    for (int i = 0; i < 8; i++) {
        int child_id = nodes[1].nxts[i];
        for (int j = 0; j < 8; j++) {
            int gc_id = nodes[child_id].nxts[j];
            ASSERT_TRUE(leaf_node(nodes[gc_id]));
            ASSERT_EQ(nodes[gc_id].c.L, 3);
        }
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Integration Tests ===" << std::endl;

    TEST_SUITE("Octree Build and Query");
    RUN_TEST(test_octree_build_and_search);
    RUN_TEST(test_octree_divide_and_search);

    TEST_SUITE("Coordinate Consistency");
    RUN_TEST(test_center_is_midpoint_of_corners);
    RUN_TEST(test_adjacent_cubes_share_face);

    TEST_SUITE("Full Pipeline");
    RUN_TEST(test_full_pipeline_uniform_sdf);
    RUN_TEST(test_full_pipeline_with_crossing);

    TEST_SUITE("Deep Octree + Projection");
    RUN_TEST(test_deep_octree_projected_sizes_decrease);
    RUN_TEST(test_projection_consistency_across_children);

    TEST_SUITE("Divide + Search");
    RUN_TEST(test_divide_then_search_multiple_cubes);

    TEST_SUITE("Key Consistency");
    RUN_TEST(test_vertex_keys_are_consistent);

    TEST_SUITE("Stress");
    RUN_TEST(test_large_octree_node_count);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
