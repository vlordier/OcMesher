// Tests for find_edges and compute_boundary
// Build: g++ -std=c++11 -O2 -o test_find_edges test_find_edges.cpp -lm

#include "test_framework.h"

// ============================================================
// compute_boundary
// ============================================================

bool test_compute_boundary_same_cube() {
    cube c = make_cube(1, 1, 1, 2);
    cube bound = make_cube(1, 1, 1, 2);
    int instance = compute_boundary(c, bound);
    ASSERT_EQ(instance, 255);
    return true;
}

bool test_compute_boundary_valid_range() {
    cube c = make_cube(0, 0, 0, 2);
    cube bound = make_cube(0, 0, 0, 1);
    int instance = compute_boundary(c, bound);
    ASSERT_GE(instance, 0);
    ASSERT_TRUE(instance <= 255);
    return true;
}

bool test_compute_boundary_corner_cube() {
    cube c = make_cube(0, 0, 0, 2);
    cube bound = make_cube(0, 0, 0, 1);
    int instance = compute_boundary(c, bound);
    ASSERT_GT(instance, 0);
    // Interior octant (7) should not be on boundary
    ASSERT_TRUE((instance & (1 << 7)) == 0);
    return true;
}

bool test_compute_boundary_opposite_corner() {
    cube c = make_cube(1, 1, 1, 2);
    cube bound = make_cube(0, 0, 0, 1);
    int instance = compute_boundary(c, bound);
    ASSERT_GE(instance, 0);
    ASSERT_TRUE(instance <= 255);
    // Octant 0 should not be on boundary (it's interior)
    ASSERT_TRUE((instance & 1) == 0);
    return true;
}

bool test_compute_boundary_symmetry() {
    // Corner (0,0,0) has octant 7 interior
    // Corner (1,1,1) has octant 0 interior (by symmetry)
    cube bound = make_cube(0, 0, 0, 1);
    cube c0 = make_cube(0, 0, 0, 2);
    cube c1 = make_cube(1, 1, 1, 2);
    int inst0 = compute_boundary(c0, bound);
    int inst1 = compute_boundary(c1, bound);
    // By symmetry of the octree, the pattern should be "reversed"
    ASSERT_TRUE((inst0 & (1 << 7)) == 0);
    ASSERT_TRUE((inst1 & (1 << 0)) == 0);
    return true;
}

// ============================================================
// find_edges: uniform SDF (no crossings)
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

bool test_find_edges_large_positive() {
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
    for (int i = 0; i < 8; i++) sdf[i] = 1e6f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    for (int i = 0; i <= params::n_elements; i++)
        ASSERT_EQ((int)bipolar_edges[i].size(), 0);
    return true;
}

// ============================================================
// find_edges: with crossings
// ============================================================

bool test_find_edges_z_split() {
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

    // Split by z: vid ordering is i + (ss+1)*j + (ss+1)^2*k
    // For ss=1: vid = i + 2*j + 4*k
    // k=0: positive, k=1: negative
    sdfT sdf[8];
    for (int i = 0; i < 4; i++) sdf[i] = 1.0f;
    for (int i = 4; i < 8; i++) sdf[i] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    int total = 0;
    for (int i = 0; i <= params::n_elements; i++)
        total += bipolar_edges[i].size();
    ASSERT_GT(total, 0);
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
    sdf[0] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // Vertex 0 connects to 3 neighbors via edges -> 3 bipolar edges
    ASSERT_EQ((int)bipolar_edges[0].size(), 3);
    return true;
}

bool test_find_edges_x_split() {
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

    // Split by x: vid = i + 2*j + 4*k, i=0 positive, i=1 negative
    sdfT sdf[8];
    for (int i = 0; i < 8; i++) {
        // even indices (i component=0) -> positive
        sdf[i] = (i & 1) ? -1.0f : 1.0f;
    }

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // 4 edges cross the x-axis boundary
    ASSERT_EQ((int)bipolar_edges[0].size(), 4);
    return true;
}

bool test_find_edges_all_negative_one_positive() {
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
    sdf[7] = 1.0f;  // Only vertex 7 (1,1,1) is positive

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // Vertex 7 connects to 3 neighbors
    ASSERT_EQ((int)bipolar_edges[0].size(), 3);
    return true;
}

// ============================================================
// find_edges: multi-element
// ============================================================

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

    sdfT sdf[16];
    for (int i = 0; i < 8; i++) {
        sdf[i * 2 + 0] = 1.0f;                       // element 0: all positive
        sdf[i * 2 + 1] = (i < 4) ? 1.0f : -1.0f;     // element 1: split by z
    }

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    ASSERT_EQ((int)bipolar_edges[0].size(), 0);   // element 0: no crossing
    ASSERT_EQ((int)bipolar_edges[1].size(), 4);   // element 1: 4 crossings
    return true;
}

bool test_find_edges_multi_element_both_crossing() {
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

    sdfT sdf[16];
    for (int i = 0; i < 8; i++) {
        sdf[i * 2 + 0] = (i < 4) ? 1.0f : -1.0f;   // element 0: z-split
        sdf[i * 2 + 1] = (i & 1) ? -1.0f : 1.0f;    // element 1: x-split
    }

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    ASSERT_EQ((int)bipolar_edges[0].size(), 4);
    ASSERT_EQ((int)bipolar_edges[1].size(), 4);
    return true;
}

// ============================================================
// find_edges: edge cases
// ============================================================

bool test_find_edges_zero_sdf_positive_side() {
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

    // SDF = 0 is treated as positive (>= 0), so no crossing with all >= 0
    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = 0.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    for (int i = 0; i <= params::n_elements; i++)
        ASSERT_EQ((int)bipolar_edges[i].size(), 0);
    return true;
}

bool test_find_edges_zero_and_negative() {
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

    // Mix of 0 (positive) and negative
    sdfT sdf[8];
    for (int i = 0; i < 4; i++) sdf[i] = 0.0f;
    for (int i = 4; i < 8; i++) sdf[i] = -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // 0 vs -1 -> crossing exists
    ASSERT_GT((int)bipolar_edges[0].size(), 0);
    return true;
}

bool test_find_edges_alternating_pattern() {
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

    // Checkerboard: alternating positive/negative based on parity
    // vid = i + 2*j + 4*k, parity = (i+j+k)%2
    // For a unit cube with ss=1, edges go along each axis between adjacent vertices
    // An edge between vertices differing in exactly one coordinate will always be bipolar
    // There are 12 edges in a cube: 4 along each axis
    // With this checkerboard pattern, every edge crosses because parity flips
    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // The actual count depends on the vertex ordering from enumerate_vertices
    // which maps vid indices to promoted vertices. Just verify we have crossings.
    ASSERT_GT((int)bipolar_edges[0].size(), 0);
    return true;
}

bool test_find_edges_very_small_sdf() {
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

    // Very small positive values except one negative
    sdfT sdf[8];
    for (int i = 0; i < 8; i++) sdf[i] = 1e-7f;
    sdf[0] = -1e-7f;

    vector<vector<key_edge> > bipolar_edges(params::n_elements + 1);
    find_edges(n, vmap, sdf, bipolar_edges);

    // Still should detect the crossing
    ASSERT_EQ((int)bipolar_edges[0].size(), 3);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Find Edges Tests ===" << std::endl;

    TEST_SUITE("compute_boundary");
    RUN_TEST(test_compute_boundary_same_cube);
    RUN_TEST(test_compute_boundary_valid_range);
    RUN_TEST(test_compute_boundary_corner_cube);
    RUN_TEST(test_compute_boundary_opposite_corner);
    RUN_TEST(test_compute_boundary_symmetry);

    TEST_SUITE("find_edges: uniform SDF");
    RUN_TEST(test_find_edges_uniform_positive);
    RUN_TEST(test_find_edges_uniform_negative);
    RUN_TEST(test_find_edges_large_positive);

    TEST_SUITE("find_edges: with crossings");
    RUN_TEST(test_find_edges_z_split);
    RUN_TEST(test_find_edges_single_negative_vertex);
    RUN_TEST(test_find_edges_x_split);
    RUN_TEST(test_find_edges_all_negative_one_positive);

    TEST_SUITE("find_edges: multi-element");
    RUN_TEST(test_find_edges_multi_element);
    RUN_TEST(test_find_edges_multi_element_both_crossing);

    TEST_SUITE("find_edges: edge cases");
    RUN_TEST(test_find_edges_zero_sdf_positive_side);
    RUN_TEST(test_find_edges_zero_and_negative);
    RUN_TEST(test_find_edges_alternating_pattern);
    RUN_TEST(test_find_edges_very_small_sdf);

    TEST_RESULTS();
    return g_failed > 0 ? 1 : 0;
}
