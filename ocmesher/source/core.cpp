// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root
// directory of this source tree.

// Authors: Zeyu Ma

#include "core.h"

namespace coarse {
std::vector<Node> nodes; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::priority_queue<std::pair<T, int>>
    nodes_heap;                // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> nodes_vector; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace coarse

namespace fine {
int start_node, end_node; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::pair<int, Int3>>
    cubes_queue;                     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<bool> cubes;             // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> cubes_index;        // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> vertices;           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> vertices_index;     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Vertex> output_vertices; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int>
    output_vertices_index; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace fine

namespace solid {
std::vector<Cube> cubes;                                          // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_set<KeyCube, KeyCubeHash> cubes_set;               // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_set<KeyCube, KeyCubeHash> visible_set,
    occluded_set; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace solid

namespace final_ns {
std::queue<int> new_nodes;            // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Vertex> v;                // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Cube> visible_nodes_cube; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> occluded_nodes_id;   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
int gl, start_node, end_node, size0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<KeyCube, int, KeyCubeHash> vertices;     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> bipolar_edges_s;     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::vector<KeyEdge>>
    bipolar_edges; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::vector<int>>
    bipolar_edges_vindices; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::unordered_map<std::pair<int, KeyCube>, int, IntKeyCubeHash>
    bipolar_edges_vertices;    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<int> vertices_cnt; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::vector<KeyCube>>
    bipolar_edges_vertices_vector; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::vector<ComputedVertex>>
    bipolar_edges_computed_vertices; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Node> nodes;             // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::vector<bool>>
    in_view_tag;            // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<Cube> searched; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace final_ns

namespace computing {
std::vector<int> faces; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::pair<int, ComputedVertex>>
    edge_vertices; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<bool>
    edge_vertices_in_view_tag; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<std::pair<int, ComputedVertex>>
    face_vertices; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
std::vector<bool>
    face_vertices_in_view_tag; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// Hash for std::pair<int,int> used by computing::face_vertices_map.
struct IntPairHash {
    auto operator()(const std::pair<int, int>& p) const noexcept -> std::size_t {
        std::size_t h = std::hash<int>{}(p.first);
        h ^= std::hash<int>{}(p.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
std::unordered_map<std::pair<int, int>, int, IntPairHash>
    face_vertices_map; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace computing

extern "C" {
int run_coarse(                                                                         // NOLINT
    T* center, T size, int n_cams, T* cams, T pixels_per_cube, T occ_scale, T min_dist, // NOLINT
    int coarse_count, int memory_limit_mb, int n_elements) {
    using namespace coarse;
    params::center = center;
    params::size = size;
    params::n_cams = n_cams;
    params::cams = cams;
    params::pixels_per_cube = pixels_per_cube;
    params::occ_scale = occ_scale;
    params::min_dist = min_dist;
    params::coarse_count = coarse_count;
    params::memory_limit_mb = memory_limit_mb;
    params::n_elements = n_elements;
    params::precompute_cam_pix_ang();
    Node root;
    markLeafNode(root);
    memset(root.m_c.m_coords, 0, 3 * sizeof(int));
    root.m_c.m_l = 0;
    nodes.clear();
    nodes.push_back(root);
    nodes_heap = std::priority_queue<std::pair<T, int>>();
    nodes_heap.emplace(projectedSize(root.m_c), 0);

    while (!nodes_heap.empty() && nodes.size() < static_cast<size_t>(coarse_count)) {
        auto top = nodes_heap.top();
        if (top.first < occ_scale)
            break;
        nodes_heap.pop();
        int i0 = static_cast<int>(nodes.size());
        expandOctree(nodes, top.second);
        for (int i = 0; i < 8; i++)
            nodes_heap.emplace(projectedSize(nodes[i0 + i].m_c), i0 + i);
    }
    fine::end_node = 0;
    solid::cubes.clear();
    solid::cubes_set.clear();
    nodes_vector.clear();
    while (!nodes_heap.empty()) {
        auto top = nodes_heap.top();
        markGridNode(nodes[top.second], std::max(0, intLog(top.first / params::occ_scale)));
        nodes_vector.push_back(top.second);
        nodes_heap.pop();
    }
    return static_cast<int>(nodes_vector.size());
}

int fine_group() { // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    using namespace coarse;
    using namespace fine;
    cubes_queue.clear();
    cubes_index.clear();
    vertices_index.clear();
    if (end_node == static_cast<int>(nodes_vector.size())) {
        cubes.clear();
        vertices.clear();
        nodes_vector.clear();
        return 0;
    }
    start_node = end_node;
    int cubes_size = 0, vertices_size = 0;
    for (;;) {
        int top = nodes_vector[end_node++];
        int s = gridNodeLevel(nodes[top]);
        cubes_index.push_back(cubes_size);
        vertices_index.push_back(vertices_size);
        cubes_size += cubex(1 << s);
        vertices_size += cubex((1 << s) + 1);
        if (((vertices_size >> 20) * (sizeof(bool) + 6 * sizeof(int) + sizeof(Vertex) +
                                      (3 + params::n_elements) * sizeof(T))) >
            static_cast<size_t>(params::memory_limit_mb))
            break;
        if (end_node == static_cast<int>(nodes_vector.size()))
            break;
    }
    cubes = std::vector<bool>(cubes_size, false);
    vertices = std::vector<int>(vertices_size, -1);

    for (int i = 0; i < end_node - start_node; i++) {
        int ss = 1 << gridNodeLevel(nodes[nodes_vector[start_node + i]]);
        for (int j = 0; j < ss; j++)     // NOLINT(readability-identifier-length)
            for (int k = 0; k < ss; k++) // NOLINT(readability-identifier-length)
                for (int f = 0; f < 6; f++) {
                    int coords[3];
                    coords[f / 2] = (f & 1) * (ss - 1);
                    coords[(f / 2 + 1) % 3] = j;
                    coords[(f / 2 + 2) % 3] = k;
                    int ci = cubeIndex(coords[0], coords[1], coords[2], ss),
                        ici = cubes_index[i] + ci;
                    if (cubes[ici])
                        continue;
                    cubes[ici] = true;
                    cubes_queue.emplace_back(i, makeInt3(coords[0], coords[1], coords[2]));
                }
    }
    return end_node - start_node;
}

int fine_iteration( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    sdfT* sdf) {
    using namespace coarse;
    using namespace fine;
    if (sdf != nullptr) {
        for (int i = 0; i < static_cast<int>(output_vertices.size());
             i++) { // NOLINT(modernize-loop-convert)
            assert(!std::isnan(sdf[i]));
            vertices[output_vertices_index[i]] = sdf[i] >= 0 ? 1 : 2;
        }
        int cqs = static_cast<int>(cubes_queue.size());
        for (int j = 0; j < cqs; j++) { // NOLINT(readability-identifier-length)
            int i = cubes_queue[j].first;
            int s = gridNodeLevel(nodes[nodes_vector[start_node + i]]), ss = 1 << s;
            Cube cubei = nodes[nodes_vector[start_node + i]].m_c;
            Int3 cqj = cubes_queue[j].second;
            int coords[3] = {cqj.first, cqj.second.first, cqj.second.second};
            bool flag = false;
            for (int e = 0; e < 12; e++) { // NOLINT(readability-identifier-length)
                int vcoords[3];
                vcoords[e / 4] = coords[e / 4] + 1;
                vcoords[(e / 4 + 1) % 3] = coords[(e / 4 + 1) % 3] + (e & 1);
                vcoords[(e / 4 + 2) % 3] = coords[(e / 4 + 2) % 3] + ((e >> 1) & 1);
                int sign1 = vertices[vertices_index[i] +
                                     cubeIndex(vcoords[0], vcoords[1], vcoords[2], ss + 1)];
                bool border1 = false, border2 = false;
                for (int p = 0; p < 3; p++) // NOLINT(modernize-loop-convert)
                    border1 = border1 || (vcoords[p] == 0 || vcoords[p] == ss);
                vcoords[e / 4] = coords[e / 4];
                int sign2 = vertices[vertices_index[i] +
                                     cubeIndex(vcoords[0], vcoords[1], vcoords[2], ss + 1)];
                for (int p = 0; p < 3; p++) // NOLINT(modernize-loop-convert)
                    border2 = border2 || (vcoords[p] == 0 || vcoords[p] == ss);
                if (sign1 != sign2) {
                    flag = true;
                    if (border1 && border2) {
                        for (int k = 0; k < 4; k++) { // NOLINT(readability-identifier-length)
                            int inter_coords[3];
                            for (int p = 0; p < 3; p++)
                                assign(inter_coords[p], cubei.m_coords[p], 2 << s, 2 * vcoords[p]);
                            assign(inter_coords[e / 4], inter_coords[e / 4], 1, 1);
                            assign(inter_coords[(e / 4 + 1) % 3], inter_coords[(e / 4 + 1) % 3], 1,
                                   -1 + 2 * (k & 1));
                            assign(inter_coords[(e / 4 + 2) % 3], inter_coords[(e / 4 + 2) % 3], 1,
                                   -1 + 2 * ((k >> 1) & 1));
                            Cube new_cube = search(&nodes[0], inter_coords, cubei.m_l + s + 1);
                            if (!isBoundary(new_cube))
                                solid::cubes_set.insert(cubeToKey(new_cube));
                        }
                    } else {
                        for (int c = 0; c < 4; c++) { // NOLINT(readability-identifier-length)
                            int ccoords[3];
                            ccoords[e / 4] = vcoords[e / 4];
                            ccoords[(e / 4 + 1) % 3] = vcoords[(e / 4 + 1) % 3] + (c & 1) - 1;
                            ccoords[(e / 4 + 2) % 3] =
                                vcoords[(e / 4 + 2) % 3] + ((c >> 1) & 1) - 1;
                            int ci = cubeIndex(ccoords[0], ccoords[1], ccoords[2], ss),
                                ici = cubes_index[i] + ci;
                            if (cubes[ici])
                                continue;
                            cubes[ici] = true;
                            cubes_queue.emplace_back(i,
                                                     makeInt3(ccoords[0], ccoords[1], ccoords[2]));
                            Cube c0;
                            for (int p = 0; p < 3; p++)
                                assign(c0.m_coords[p], cubei.m_coords[p], 1 << s, ccoords[p]);
                            c0.m_l = cubei.m_l + s;
                            solid::cubes_set.insert(cubeToKey(c0));
                        }
                    }
                }
            }
            if (flag) {
                Cube c0;
                for (int p = 0; p < 3; p++)
                    assign(c0.m_coords[p], cubei.m_coords[p], 1 << s, coords[p]);
                c0.m_l = cubei.m_l + s;
                solid::cubes_set.insert(cubeToKey(c0));
            }
        }
        cubes_queue.erase(cubes_queue.begin(), cubes_queue.begin() + cqs);
    }
    output_vertices.clear();
    output_vertices_index.clear();
    int cqs = static_cast<int>(cubes_queue.size());
    for (int j = 0; j < cqs; j++) { // NOLINT(readability-identifier-length)
        int i = cubes_queue[j].first;
        int s = gridNodeLevel(nodes[nodes_vector[start_node + i]]), ss = 1 << s;
        Int3 cqj = cubes_queue[j].second;
        for (int dx = 0; dx < 2; dx++)
            for (int dy = 0; dy < 2; dy++)
                for (int dz = 0; dz < 2; dz++) {
                    int coords[3] = {cqj.first + dx, cqj.second.first + dy, cqj.second.second + dz};
                    int ci = cubeIndex(coords[0], coords[1], coords[2], ss + 1),
                        ici = vertices_index[i] + ci;
                    if (vertices[ici] == -1) {
                        Vertex vx;
                        Cube c = nodes[nodes_vector[start_node + i]].m_c;
                        for (int p = 0; p < 3; p++)
                            assign(vx.m_coords[p], c.m_coords[p], 1 << s, coords[p]);
                        vx.m_l = c.m_l + s;
                        vertices[ici] = 0;
                        output_vertices.push_back(vx);
                        output_vertices_index.push_back(ici);
                    }
                }
    }
    return static_cast<int>(output_vertices.size());
}

void fine_iteration_output( // NOLINT(readability-identifier-naming,
                            // modernize-use-trailing-return-type)
    T* xyz) {
    using namespace params;
    using namespace fine;
    for (int i = 0; i < static_cast<int>(output_vertices.size());
         i++) { // NOLINT(modernize-loop-convert)
        computeCoords(xyz + static_cast<ptrdiff_t>(i) * 3, output_vertices[i].m_coords,
                      output_vertices[i].m_l);
    }
}

int vis_filter( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    bool simplify_occluded, int relax_iters) {
    using namespace params;
    using namespace solid;
    for (const auto& key : cubes_set) {
        Cube c;
        keyToCube(c, key);
        cubes.push_back(c);
    }
    cubes_set.clear();

    std::vector<char> visible(cubes.size(), 0); // char avoids vector<bool> bit-packing data races
    T factor = 10;
    std::vector<T> canvas;
    for (int k = 0; k < n_cams; k++) { // NOLINT(readability-identifier-length)
        T* current_cam = cams + static_cast<ptrdiff_t>(k) * (12 + 9 + 2);
        int height = static_cast<int>(current_cam[21] / factor),
            width = static_cast<int>(current_cam[22] / factor);
        if (simplify_occluded) {
            // Build per-thread depth buffers to avoid omp critical contention,
            // then reduce with a single sequential min-pass.
            int nth = omp_get_max_threads();
            int hw = height * width;
            std::vector<T> tcanvas(static_cast<std::size_t>(nth) * static_cast<std::size_t>(hw),
                                   std::numeric_limits<T>::infinity());
            int cubes_n = static_cast<int>(cubes.size());
#pragma omp parallel for
            for (int i = 0; i < cubes_n; i++) { // NOLINT(modernize-loop-convert)
                T image_coords[3];
                projectedCoords(cubes[i], k, image_coords, nullptr);
                if (image_coords[2] >= 0) {
                    int x = static_cast<int>(std::floor(image_coords[0] / factor));
                    int y = static_cast<int>(std::floor(image_coords[1] / factor));
                    if (x >= 0 && y >= 0 && x < width && y < height) {
                        T& cell = tcanvas[static_cast<std::size_t>(omp_get_thread_num()) *
                                              static_cast<std::size_t>(hw) +
                                          static_cast<std::size_t>(x) *
                                              static_cast<std::size_t>(height) +
                                          static_cast<std::size_t>(y)];
                        if (image_coords[2] < cell)
                            cell = image_coords[2];
                    }
                }
            }
            canvas = std::vector<T>(static_cast<std::size_t>(hw), std::numeric_limits<T>::infinity());
            for (int t = 0; t < nth; t++)
                for (int j = 0; j < hw; j++)
                    if (tcanvas[static_cast<std::size_t>(t) * static_cast<std::size_t>(hw) + j] <
                        canvas[j])
                        canvas[j] =
                            tcanvas[static_cast<std::size_t>(t) * static_cast<std::size_t>(hw) + j];
        }
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(cubes.size()); i++) { // NOLINT(modernize-loop-convert)
            T image_coords[3];
            projectedCoords(cubes[i], k, image_coords, nullptr);
            if (image_coords[2] >= 0) {
                int x = static_cast<int>(std::floor(image_coords[0] / factor));
                int y = static_cast<int>(std::floor(image_coords[1] / factor));
                if (x >= -relax_iters && y >= -relax_iters && x < width + relax_iters &&
                    y < height + relax_iters) {
                    if (simplify_occluded) {
                        for (int dx = -relax_iters; dx <= relax_iters; dx++)
                            for (int dy = -relax_iters; dy <= relax_iters; dy++) {
                                int nx = x + dx, ny = y + dy;
                                if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                                    if (image_coords[2] <=
                                        canvas[static_cast<std::size_t>(nx) *
                                                   static_cast<std::size_t>(height) +
                                               static_cast<std::size_t>(ny)]) {
                                        visible[i] = true;
                                    }
                                }
                            }
                    } else {
                        visible[i] = true;
                    }
                }
            }
        }
    }
    visible_set.clear();
    occluded_set.clear();
    std::unordered_set<KeyCube, KeyCubeHash> new_visible_set, old_visible_set;
    for (int i = 0; i < static_cast<int>(visible.size()); i++) { // NOLINT(modernize-loop-convert)
        if (visible[i])
            visible_set.insert(cubeToKey(cubes[i]));
        else
            occluded_set.insert(cubeToKey(cubes[i]));
    }
    visible.clear();
    cubes.clear();
    for (int i = 0; i < relax_iters; i++) {
        for (const auto& key : visible_set) {
            Cube c;
            keyToCube(c, key);
            int coords[3];
            for (int f = 0; f < 6; f++) {
                assign(coords[f / 3], c.m_coords[f / 3], 2, -1 + 4 * (f & 1));
                assign(coords[(f / 3 + 1) % 3], c.m_coords[(f / 3 + 1) % 3], 2, 1);
                assign(coords[(f / 3 + 2) % 3], c.m_coords[(f / 3 + 2) % 3], 2, 1);
                Cube new_cube = search(&coarse::nodes[0], coords, c.m_l + 1);
                if (!isBoundary(new_cube)) {
                    if (occluded_set.count(cubeToKey(new_cube))) {
                        occluded_set.erase(cubeToKey(new_cube));
                        new_visible_set.insert(cubeToKey(new_cube));
                    }
                }
            }
        }
        old_visible_set.insert(visible_set.begin(), visible_set.end());
        visible_set = new_visible_set;
        new_visible_set.clear();
    }
    visible_set.insert(old_visible_set.begin(), old_visible_set.end());
    old_visible_set.clear();
    new_visible_set.clear();

    Node root;
    memset(root.m_nxts, -1, 8 * sizeof(int));
    memset(root.m_c.m_coords, 0, 3 * sizeof(int));
    root.m_c.m_l = 0;
    final_ns::nodes.clear();
    final_ns::nodes.push_back(root);
    final_ns::visible_nodes_cube.clear();
    final_ns::occluded_nodes_id.clear();
    for (const auto& key : visible_set) {
        Cube c;
        keyToCube(c, key);
        final_ns::visible_nodes_cube.push_back(c);
    }
    for (const auto& key : occluded_set) {
        Cube c;
        keyToCube(c, key);
        final_ns::occluded_nodes_id.push_back(divideToCube(final_ns::nodes, c));
    }
    final_ns::bipolar_edges.clear();
    final_ns::bipolar_edges_s.clear();
    final_ns::bipolar_edges_vertices.clear();
    final_ns::in_view_tag.clear();
    final_ns::bipolar_edges_vindices.clear();
    for (int i = 0; i < params::n_elements; i++) {
        final_ns::bipolar_edges.emplace_back();
        final_ns::bipolar_edges_s.push_back(0);
        final_ns::bipolar_edges_vindices.emplace_back();
        final_ns::in_view_tag.emplace_back();
    }
    final_ns::vertices_cnt = std::vector<int>(5, 0);
    final_ns::bipolar_edges.emplace_back();
    final_ns::start_node = 0;
    final_ns::gl = 0;
    for (;;) {
        int total_nodes = 0;
        int vis_cube_n = static_cast<int>(final_ns::visible_nodes_cube.size());
        for (int i = 0; i < vis_cube_n; i++) { // NOLINT(modernize-loop-convert)
            int is =
                std::max(0, intLog(projectedSize(final_ns::visible_nodes_cube[i])) - final_ns::gl);
            total_nodes += std::max(0, cubex(1 << is) - cubex((1 << is) - 2));
        }
        if (((final_ns::nodes.size() + total_nodes) >> 20) * sizeof(Node) <
            static_cast<size_t>(params::memory_limit_mb * 0.6))
            break;
        final_ns::gl++;
    }
    return static_cast<int>(final_ns::visible_nodes_cube.size());
}

int final_iteration() { // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    using namespace final_ns;
    assert(((nodes.size() >> 20) * sizeof(Node)) <
           static_cast<size_t>(params::memory_limit_mb * 0.8));
    if (start_node == static_cast<int>(visible_nodes_cube.size())) {
        coarse::nodes.clear();
        solid::visible_set.clear();
        visible_nodes_cube.clear();
        return 0;
    }
    int total_nodes = 0, total_verts = 0;
    for (end_node = start_node; end_node < static_cast<int>(visible_nodes_cube.size());
         end_node++) {
        int is = intLog(projectedSize(visible_nodes_cube[end_node]));
        total_nodes += (1 << (3 * (is - gl))) * 8 / 7;
        total_verts += cubex((1 << is) + 1);
        if (((nodes.size() + total_nodes) >> 20) * sizeof(Node) +
                (total_verts >> 20) *
                    (sizeof(Vertex) + sizeof(T*) + sizeof(KeyCube) + sizeof(int) +
                     2 * sizeof(KeyCube*) + (3 + params::n_elements) * sizeof(T)) >
            static_cast<size_t>(params::memory_limit_mb))
            break;
    }
    assert(end_node != start_node);

    vertices.clear();
    size0 = static_cast<int>(nodes.size());
    assert(new_nodes.empty());
    for (int i = start_node; i < end_node; i++) {
        new_nodes.push(divideToCube(nodes, visible_nodes_cube[i]));
        while (!new_nodes.empty()) {
            int ind = new_nodes.front();
            new_nodes.pop();
            T s = projectedSize(nodes[ind].m_c);
            if (s > static_cast<T>(1 << gl)) {
                int base_size = static_cast<int>(nodes.size());
                expandOctree(nodes, ind);
                for (int j = 0; j < 8; j++)
                    new_nodes.push(base_size + j); // NOLINT(readability-identifier-length)
            } else {
                int is = intLog(s);
                markGridNode(nodes[ind], is);
                v.resize(cubex((1 << is) + 1));
                enumerateVertices(&v[0], nodes[ind]);
                for (int j = 0; j < cubex((1 << is) + 1);
                     j++) // NOLINT(readability-identifier-length)
                    if (!vertices.count(cubeToKey(v[j])))
                        vertices[cubeToKey(v[j])] = static_cast<int>(vertices.size());
            }
        }
    }
    assert(!vertices.empty());
    return static_cast<int>(vertices.size());
}

int final_iteration_occluded() { // NOLINT(readability-identifier-naming,
                                 // modernize-use-trailing-return-type)
    using namespace final_ns;
    int occl_n = static_cast<int>(occluded_nodes_id.size());
    for (int i = 0; i < occl_n; i++) { // NOLINT(modernize-loop-convert)
        v.resize(8);
        enumerateVertices(&v[0], nodes[occluded_nodes_id[i]]);
        for (int j = 0; j < 8; j++) // NOLINT(readability-identifier-length)
            if (!vertices.count(cubeToKey(v[j])))
                vertices[cubeToKey(v[j])] = static_cast<int>(vertices.size());
    }
    return static_cast<int>(vertices.size());
}

void final_iteration2( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    T* xyz) {
    using namespace params;
    using namespace final_ns;
    for (const auto& [key, val] : vertices) {
        Vertex vx;
        keyToCube(vx, key);
        computeCoords(xyz + static_cast<ptrdiff_t>(val) * 3, vx.m_coords, vx.m_l);
    }
}

int final_iteration3( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    sdfT* sdf) {
    using namespace final_ns;
    using namespace solid;
    int start = static_cast<int>(visible_nodes_cube.size()) - start_node;
    for (int i = 0; i < static_cast<int>(vertices.size()) * params::n_elements; i++)
        assert(!std::isnan(sdf[i]));
    for (int i = size0; i < static_cast<int>(nodes.size()); i++)
        if (isLeafNode(nodes[i]))
            findEdges(nodes[i], vertices, sdf, bipolar_edges);
    vertices.clear();
    for (int i = 0; i < params::n_elements + 1; i++) {
        auto& bei = bipolar_edges[i];
        int s = (i == params::n_elements) ? 0 : bipolar_edges_s[i];
        std::sort(bei.begin() + s, bei.end());
        bei.erase(std::unique(bei.begin() + s, bei.end()), bei.end());
        int e = static_cast<int>(bei.size()); // NOLINT(readability-identifier-length)
        searched.resize(static_cast<std::size_t>(e - s) * 4);
#pragma omp parallel for
        for (int j = s; j < e; j++) { // NOLINT(readability-identifier-length)
            Cube e0;
            keyToCube(e0, bei[j].second);
            int dir = bei[j].first;
            int coords[3], level = e0.m_l + 1;
            int ks[4] = {0, 1, 3, 2};
            bool flag = true, flag2 = true;
            for (int ik = 0; ik < 4; ik++) {
                int k = dir > 0 ? ks[ik] : ks[3 - ik]; // NOLINT(readability-identifier-length)
                int c0 = std::abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
                assign(coords[c0], e0.m_coords[c0], 2, 1);
                assign(coords[c1], e0.m_coords[c1], 2, -1 + 2 * (k & 1));
                assign(coords[c2], e0.m_coords[c2], 2, -1 + 2 * ((k >> 1) & 1));
                Cube& cx =
                    searched[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(ik)];
                cx = search(&nodes[0], coords, level);
                if (isBoundary(cx)) {
                    flag = false;
                    break;
                } else if (isExterior(cx)) {
                    flag2 = false;
                }
            }
            if (!flag) {
                for (int k = 0; k < 4; k++) // NOLINT(readability-identifier-length)
                    markBoundary(searched[static_cast<std::size_t>(j - s) * 4 +
                                          static_cast<std::size_t>(k)]);
            } else if (!flag2) {
                for (int k = 0; k < 4; k++) // NOLINT(readability-identifier-length)
                    markExterior(searched[static_cast<std::size_t>(j - s) * 4 +
                                          static_cast<std::size_t>(k)]);
            }
        }
        if (i == params::n_elements)
            continue;
        int j1 = s, j2 = e - 1;
        while (j1 < j2) {
            while (
                j1 < e &&
                isRegular(searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s)]))
                j1++;
            while (j2 >= s &&
                   !isRegular(
                       searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s)]))
                j2--;
            if (j1 < j2) {
                std::swap(bei[j1], bei[j2]);
                for (int k = 0; k < 4; k++) // NOLINT(readability-identifier-length)
                    std::swap(
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s) +
                                 static_cast<std::size_t>(k)],
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s) +
                                 static_cast<std::size_t>(k)]);
                j1++;
                j2--;
            }
        }
        auto& ivt = in_view_tag[i];
        for (int j = s; j < j1; j++) {    // NOLINT(readability-identifier-length)
            for (int k = 0; k < 4; k++) { // NOLINT(readability-identifier-length)
                Cube cx =
                    searched[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(k)];
                int vid;
                KeyCube key = cubeToKey(cx);
                if (bipolar_edges_vertices.count(std::make_pair(i, key)))
                    vid = bipolar_edges_vertices[std::make_pair(i, key)];
                else {
                    vid = bipolar_edges_vertices[std::make_pair(i, key)] = vertices_cnt[i]++;
                    ivt.push_back(occluded_set.count(key) == 0);
                }
                bipolar_edges_vindices[i].push_back(vid);
            }
        }
        bipolar_edges_s[i] = j1;
        j2 = e - 1;
        while (j1 < j2) {
            while (j1 < e &&
                   isExterior(
                       searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s)]))
                j1++;
            while (j2 >= bipolar_edges_s[i] &&
                   isBoundary(
                       searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s)]))
                j2--;
            if (j1 < j2) {
                std::swap(bei[j1], bei[j2]);
                for (int k = 0; k < 4; k++) // NOLINT(readability-identifier-length)
                    std::swap(
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s) +
                                 static_cast<std::size_t>(k)],
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s) +
                                 static_cast<std::size_t>(k)]);
                j1++;
                j2--;
            }
        }
        bei.erase(bei.begin() + j1, bei.end());
    }

    for (int i = 0; i < size0; i++) // NOLINT(modernize-loop-convert)
        for (int j = 0; j < 8; j++) // NOLINT(readability-identifier-length, modernize-loop-convert)
            if (nodes[i].m_nxts[j] >= size0)
                nodes[i].m_nxts[j] = -1;
    nodes.erase(nodes.begin() + size0, nodes.end());

    for (int i = start_node; i < end_node; i++) {
        assert(new_nodes.empty());
        int nodes_id = divideToCube(nodes, visible_nodes_cube[i]);
        memset(nodes[nodes_id].m_nxts, -1, 3 * sizeof(int));
        new_nodes.push(nodes_id);
        while (!new_nodes.empty()) {
            int ind = new_nodes.front();
            new_nodes.pop();
            T s = projectedSize(nodes[ind].m_c);
            if (s > static_cast<T>(1 << gl)) {
                int instance = computeBoundary(nodes[ind].m_c, nodes[nodes_id].m_c);
                int size0_local = static_cast<int>(nodes.size());
                partialExpandOctree(nodes, ind, instance);
                for (int j = size0_local; j < static_cast<int>(nodes.size()); j++)
                    new_nodes.push(j); // NOLINT(readability-identifier-length)
            } else {
                markGridNode(nodes[ind], intLog(s));
            }
        }
    }

    auto& ben = bipolar_edges[params::n_elements];
    for (int i = 0; i < static_cast<int>(ben.size()); i++) { // NOLINT(modernize-loop-convert)
        int dir = ben[i].first;
        Cube e0;
        keyToCube(e0, ben[i].second);
        int coords[3], level = e0.m_l + 1;
        for (int k = 0; k < 4; k++) { // NOLINT(readability-identifier-length)
            int c0 = std::abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
            assign(coords[c0], e0.m_coords[c0], 2, 1);
            assign(coords[c1], e0.m_coords[c1], 2, -1 + 2 * (k & 1));
            assign(coords[c2], e0.m_coords[c2], 2, -1 + 2 * ((k >> 1) & 1));
            Cube c = search(&coarse::nodes[0], coords, level);
            if (!isBoundary(c)) {
                KeyCube key = cubeToKey(c);
                if (solid::occluded_set.count(key) || solid::visible_set.count(key))
                    continue;
                solid::visible_set.insert(key);
                visible_nodes_cube.push_back(c);
            }
        }
    }
    ben.clear();

    start_node = end_node;
    return start - (static_cast<int>(visible_nodes_cube.size()) - start_node);
}

void final_iteration3_occluded( // NOLINT(readability-identifier-naming,
                                // modernize-use-trailing-return-type)
    sdfT* sdf) {
    using namespace final_ns;
    using namespace solid;
    int vert_n = static_cast<int>(vertices.size()) * params::n_elements;
    for (int i = 0; i < vert_n; i++) // NOLINT(modernize-loop-convert)
        assert(!std::isnan(sdf[i]));
    int occl_n = static_cast<int>(occluded_nodes_id.size());
    for (int i = 0; i < occl_n; i++) // NOLINT(modernize-loop-convert)
        findEdges(nodes[occluded_nodes_id[i]], vertices, sdf, bipolar_edges);
    vertices.clear();
    occluded_nodes_id.clear();
}

void final_remaining( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    int* nv) {
    using namespace final_ns;
    bipolar_edges_computed_vertices.clear();
    bipolar_edges_vertices_vector.clear();
    for (int i = 0; i < params::n_elements; i++) {
        auto& bei = bipolar_edges[i];
        int s = bipolar_edges_s[i];
        std::sort(bei.begin() + s, bei.end());
        bei.erase(std::unique(bei.begin() + s, bei.end()), bei.end());
        int e = static_cast<int>(bei.size()); // NOLINT(readability-identifier-length)
        searched.resize(static_cast<std::size_t>(e - s) * 4);
        std::vector<bool> tags(static_cast<std::size_t>(e - s) * 4);
#pragma omp parallel for
        for (int j = s; j < e; j++) { // NOLINT(readability-identifier-length)
            Cube e0;
            keyToCube(e0, bei[j].second);
            int dir = bei[j].first;
            int coords[3], level = e0.m_l + 1;
            int ks[4] = {0, 1, 3, 2};
            bool flag = false;
            for (int ik = 0; ik < 4; ik++) {
                int k = dir > 0 ? ks[ik] : ks[3 - ik]; // NOLINT(readability-identifier-length)
                int c0 = std::abs(dir) - 1, c1 = (c0 + 1) % 3, c2 = (c1 + 1) % 3;
                assign(coords[c0], e0.m_coords[c0], 2, 1);
                assign(coords[c1], e0.m_coords[c1], 2, -1 + 2 * (k & 1));
                assign(coords[c2], e0.m_coords[c2], 2, -1 + 2 * ((k >> 1) & 1));
                Cube& cx =
                    searched[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(ik)];
                cx = search(&nodes[0], coords, level, true);
                if (isBoundary(cx)) {
                    flag = true;
                    break;
                } else {
                    tags[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(ik)] =
                        solid::occluded_set.count(cubeToKey(cx)) == 0 &&
                        !isExterior(search(&nodes[0], coords, level));
                }
            }
            if (flag) {
                for (int k = 0; k < 4; k++) // NOLINT(readability-identifier-length)
                    markBoundary(searched[static_cast<std::size_t>(j - s) * 4 +
                                          static_cast<std::size_t>(k)]);
            }
        }
        int j1 = s, j2 = e - 1;
        while (j1 < j2) {
            while (
                j1 < e &&
                isRegular(searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s)]))
                j1++;
            while (j2 >= s &&
                   !isRegular(
                       searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s)]))
                j2--;
            if (j1 < j2) {
                std::swap(bei[j1], bei[j2]);
                for (int k = 0; k < 4; k++) { // NOLINT(readability-identifier-length)
                    std::swap(
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s) +
                                 static_cast<std::size_t>(k)],
                        searched[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s) +
                                 static_cast<std::size_t>(k)]);
                    bool tmp = tags[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s) +
                                    static_cast<std::size_t>(k)];
                    tags[static_cast<std::size_t>(4) * static_cast<std::size_t>(j1 - s) +
                         static_cast<std::size_t>(k)] =
                        tags[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s) +
                             static_cast<std::size_t>(k)];
                    tags[static_cast<std::size_t>(4) * static_cast<std::size_t>(j2 - s) +
                         static_cast<std::size_t>(k)] = tmp;
                }
                j1++;
                j2--;
            }
        }
        auto& ivt = in_view_tag[i];
        for (int j = s; j < j1; j++) {    // NOLINT(readability-identifier-length)
            for (int k = 0; k < 4; k++) { // NOLINT(readability-identifier-length)
                Cube cx =
                    searched[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(k)];
                int vid;
                KeyCube key = cubeToKey(cx);
                if (bipolar_edges_vertices.count(std::make_pair(i, key)))
                    vid = bipolar_edges_vertices[std::make_pair(i, key)];
                else {
                    vid = bipolar_edges_vertices[std::make_pair(i, key)] = vertices_cnt[i]++;
                    ivt.push_back(
                        tags[static_cast<std::size_t>(j - s) * 4 + static_cast<std::size_t>(k)]);
                }
                bipolar_edges_vindices[i].push_back(vid);
            }
        }
        bei.erase(bei.begin() + j1, bei.end());
        bipolar_edges_computed_vertices.emplace_back(vertices_cnt[i]);
        bipolar_edges_vertices_vector.emplace_back(vertices_cnt[i]);
        nv[i] = vertices_cnt[i];
    }
    for (const auto& [bkey, bval] : bipolar_edges_vertices) {
        int i = bkey.first;
        auto& becvi = bipolar_edges_computed_vertices[i];
        auto& bevvi = bipolar_edges_vertices_vector[i];
        Cube c;
        keyToCube(c, bkey.second);
        computeCenter(becvi[bval].m_c, c);
        becvi[bval].m_l = 0;
        becvi[bval].m_r = 0.5 * params::size / (1 << c.m_l);
        bevvi[bval] = bkey.second;
    }
    bipolar_edges_vertices.clear();
    nodes.clear();
    solid::occluded_set.clear();
}

void get_verts_center( // NOLINT(readability-identifier-length, readability-identifier-naming,
                       // modernize-use-trailing-return-type)
    int e, T* positions) {
    using namespace final_ns;
    auto& becv = bipolar_edges_computed_vertices[e];
    for (int i = 0; i < static_cast<int>(becv.size()); i++) {
        memcpy(positions + static_cast<ptrdiff_t>(3) * i, becv[i].m_c, 3 * sizeof(T));
    }
}

void get_extra_verts_center( // NOLINT(readability-identifier-naming,
                             // modernize-use-trailing-return-type)
    T* epositions, T* fpositions) {
    using namespace computing;
    for (int i = 0; i < static_cast<int>(edge_vertices.size()); i++) {
        memcpy(epositions + static_cast<ptrdiff_t>(3) * i, edge_vertices[i].second.m_c,
               3 * sizeof(T));
    }
    for (int i = 0; i < static_cast<int>(face_vertices.size()); i++) {
        memcpy(fpositions + static_cast<ptrdiff_t>(3) * i, face_vertices[i].second.m_c,
               3 * sizeof(T));
    }
}

void update_verts( // NOLINT(readability-identifier-length, readability-identifier-naming,
                   // modernize-use-trailing-return-type)
    int e, sdfT* sdf, sdfT* center_sdf, T* positions) {
    using namespace final_ns;
    auto& becv = bipolar_edges_computed_vertices[e];
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(becv.size()); i++) {
        if (sdf != nullptr) {
            T mid = (becv[i].m_l + becv[i].m_r) / 2;
            bool bipolar = false;
            for (int j = 0; j < 8; j++) // NOLINT(readability-identifier-length)
                if ((sdf[i * 8 + j] >= 0) != (center_sdf[i] >= 0)) {
                    bipolar = true;
                    break;
                }
            if (bipolar)
                becv[i].m_r = mid;
            else
                becv[i].m_l = mid;
        }
        T mid = (becv[i].m_l + becv[i].m_r) / 2;
        for (int j = 0; j < 8; j++) { // NOLINT(readability-identifier-length)
            int cid = i * 8 + j;
            for (int k = 0; k < 3; k++) // NOLINT(readability-identifier-length)
                positions[cid * 3 + k] = becv[i].m_c[k] + (((j >> k) & 1) * 2 - 1) * mid;
        }
    }
}

void update_extra_verts(sdfT* esdf, sdfT* fsdf, sdfT* ecenter_sdf, sdfT* fcenter_sdf, // NOLINT
                        T* epositions, T* fpositions) {
    using namespace computing;
    int edge_n = 2;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(edge_vertices.size()); i++) {
        if (esdf != nullptr) {
            T mid = (edge_vertices[i].second.m_l + edge_vertices[i].second.m_r) / 2;
            bool bipolar = false;
            for (int j = 0; j < edge_n; j++) // NOLINT(readability-identifier-length)
                if ((esdf[i * edge_n + j] >= 0) != (ecenter_sdf[i] >= 0)) {
                    bipolar = true;
                    break;
                }
            if (bipolar)
                edge_vertices[i].second.m_r = mid;
            else
                edge_vertices[i].second.m_l = mid;
        }
        T mid = (edge_vertices[i].second.m_l + edge_vertices[i].second.m_r) / 2;
        for (int j = 0; j < edge_n; j++) { // NOLINT(readability-identifier-length)
            int cid = i * edge_n + j;
            for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                epositions[cid * 3 + k] = edge_vertices[i].second.m_c[k];
                if (k == edge_vertices[i].first)
                    epositions[cid * 3 + k] += (j * 2 - 1) * mid;
            }
        }
    }
    int face_n = 4;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(face_vertices.size()); i++) {
        if (fsdf != nullptr) {
            T mid = (face_vertices[i].second.m_l + face_vertices[i].second.m_r) / 2;
            bool bipolar = false;
            for (int j = 0; j < face_n; j++) // NOLINT(readability-identifier-length)
                if ((fsdf[i * face_n + j] >= 0) != (fcenter_sdf[i] >= 0)) {
                    bipolar = true;
                    break;
                }
            if (bipolar)
                face_vertices[i].second.m_r = mid;
            else
                face_vertices[i].second.m_l = mid;
        }
        T mid = (face_vertices[i].second.m_l + face_vertices[i].second.m_r) / 2;
        for (int j = 0; j < face_n; j++) { // NOLINT(readability-identifier-length)
            int cid = i * face_n + j;
            for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                fpositions[cid * 3 + k] = face_vertices[i].second.m_c[k];
                if (k == (face_vertices[i].first + 1) % 3)
                    fpositions[cid * 3 + k] += (firstDigit(j) * 2 - 1) * mid;
                else if (k == (face_vertices[i].first + 2) % 3)
                    fpositions[cid * 3 + k] += (secondDigit(j) * 2 - 1) * mid;
            }
        }
    }
}

void get_lr_verts( // NOLINT(readability-identifier-length, readability-identifier-naming,
                   // modernize-use-trailing-return-type)
    int e, T* cube_l, T* cube_r) {
    using namespace final_ns;
    auto& becv = bipolar_edges_computed_vertices[e];
    for (int i = 0; i < static_cast<int>(becv.size()); i++) {
        for (int j = 0; j < 8; j++) { // NOLINT(readability-identifier-length)
            int cid = i * 8 + j;
            for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                cube_l[cid * 3 + k] = becv[i].m_c[k] + (((j >> k) & 1) * 2 - 1) * becv[i].m_l;
                cube_r[cid * 3 + k] = becv[i].m_c[k] + (((j >> k) & 1) * 2 - 1) * becv[i].m_r;
            }
        }
    }
}

void get_lr_extra_verts(                          // NOLINT(readability-identifier-naming,
                                                  // modernize-use-trailing-return-type)
    T* epos_l, T* epos_r, T* fpos_l, T* fpos_r) { // NOLINT(bugprone-easily-swappable-parameters)
    using namespace computing;
    for (int i = 0; i < static_cast<int>(edge_vertices.size()); i++) {
        for (int j = 0; j < 2; j++) { // NOLINT(readability-identifier-length)
            int cid = i * 2 + j;
            for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                epos_l[cid * 3 + k] = edge_vertices[i].second.m_c[k];
                if (k == edge_vertices[i].first)
                    epos_l[cid * 3 + k] += (j * 2 - 1) * edge_vertices[i].second.m_l;
                epos_r[cid * 3 + k] = edge_vertices[i].second.m_c[k];
                if (k == edge_vertices[i].first)
                    epos_r[cid * 3 + k] += (j * 2 - 1) * edge_vertices[i].second.m_r;
            }
        }
    }
    for (int i = 0; i < static_cast<int>(face_vertices.size()); i++) {
        for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
            int cid = i * 4 + j;
            for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                fpos_l[cid * 3 + k] = face_vertices[i].second.m_c[k];
                if (k == (face_vertices[i].first + 1) % 3)
                    fpos_l[cid * 3 + k] += (firstDigit(j) * 2 - 1) * face_vertices[i].second.m_l;
                else if (k == (face_vertices[i].first + 2) % 3)
                    fpos_l[cid * 3 + k] += (secondDigit(j) * 2 - 1) * face_vertices[i].second.m_l;
                fpos_r[cid * 3 + k] = face_vertices[i].second.m_c[k];
                if (k == (face_vertices[i].first + 1) % 3)
                    fpos_r[cid * 3 + k] += (firstDigit(j) * 2 - 1) * face_vertices[i].second.m_r;
                else if (k == (face_vertices[i].first + 2) % 3)
                    fpos_r[cid * 3 + k] += (secondDigit(j) * 2 - 1) * face_vertices[i].second.m_r;
            }
        }
    }
}

// todo consider more than corners when a vertex cube has complex side face
void finalize_verts( // NOLINT(readability-identifier-length, readability-identifier-naming,
                     // modernize-use-trailing-return-type)
    int e, sdfT* sdf_l, sdfT* sdf_r, T* verts) {
    using namespace final_ns;
    auto& becv = bipolar_edges_computed_vertices[e];
    for (int i = 0; i < static_cast<int>(becv.size()); i++) {
        T vx[3] = {0};
        int w = 0;
        for (int j = 0; j < 8; j++) { // NOLINT(readability-identifier-length)
            T sl = static_cast<T>(sdf_l[i * 8 + j]);
            T sr = static_cast<T>(sdf_r[i * 8 + j]);
            if ((sl >= 0) != (sr >= 0)) {
                w++;
                // Linear interpolation toward zero crossing (more accurate than midpoint)
                T dist = becv[i].m_l + (sl / (sl - sr)) * (becv[i].m_r - becv[i].m_l);
                for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                    vx[k] += becv[i].m_c[k] + (((j >> k) & 1) * 2 - 1) * dist;
                }
            }
        }
        if (w == 0) {
            for (int k = 0; k < 3; k++)
                verts[i * 3 + k] = becv[i].m_c[k]; // NOLINT(readability-identifier-length)
        } else {
            for (int k = 0; k < 3; k++)
                verts[i * 3 + k] = vx[k] / w; // NOLINT(readability-identifier-length)
        }
    }
    bipolar_edges_computed_vertices[e].clear();
}

void finalize_extra_verts( // NOLINT(readability-identifier-naming,
                           // modernize-use-trailing-return-type)
    sdfT* esdf_l, sdfT* esdf_r, T* everts, sdfT* fsdf_l, sdfT* fsdf_r, T* fverts) {
    using namespace computing;
    for (int i = 0; i < static_cast<int>(edge_vertices.size()); i++) {
        T vx[3] = {0};
        int w = 0;
        for (int j = 0; j < 2; j++) { // NOLINT(readability-identifier-length)
            T sl = static_cast<T>(esdf_l[i * 2 + j]);
            T sr = static_cast<T>(esdf_r[i * 2 + j]);
            if ((sl >= 0) != (sr >= 0)) {
                w++;
                T dist = edge_vertices[i].second.m_l +
                         (sl / (sl - sr)) * (edge_vertices[i].second.m_r - edge_vertices[i].second.m_l);
                for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                    vx[k] += edge_vertices[i].second.m_c[k];
                    if (k == edge_vertices[i].first)
                        vx[k] += (j * 2 - 1) * dist;
                }
            }
        }
        assert(w != 0);
        for (int k = 0; k < 3; k++)
            everts[i * 3 + k] = vx[k] / w; // NOLINT(readability-identifier-length)
    }
    edge_vertices.clear();
    for (int i = 0; i < static_cast<int>(face_vertices.size()); i++) {
        T vx[3] = {0};
        int w = 0;
        for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
            T sl = static_cast<T>(fsdf_l[i * 4 + j]);
            T sr = static_cast<T>(fsdf_r[i * 4 + j]);
            if ((sl >= 0) != (sr >= 0)) {
                w++;
                T dist = face_vertices[i].second.m_l +
                         (sl / (sl - sr)) * (face_vertices[i].second.m_r - face_vertices[i].second.m_l);
                for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                    vx[k] += face_vertices[i].second.m_c[k];
                    if (k == (face_vertices[i].first + 1) % 3)
                        vx[k] += (firstDigit(j) * 2 - 1) * dist;
                    else if (k == (face_vertices[i].first + 2) % 3)
                        vx[k] += (secondDigit(j) * 2 - 1) * dist;
                }
            }
        }
        if (w == 0) {
            for (int k = 0; k < 3; k++)
                fverts[i * 3 + k] =
                    face_vertices[i].second.m_c[k]; // NOLINT(readability-identifier-length)
        } else {
            for (int k = 0; k < 3; k++)
                fverts[i * 3 + k] = vx[k] / w; // NOLINT(readability-identifier-length)
        }
    }
    face_vertices.clear();
}

void get_in_view_tag( // NOLINT(readability-identifier-length, readability-identifier-naming,
                      // modernize-use-trailing-return-type)
    int e, bool* output) {
    using namespace final_ns;
    using namespace computing;
    int cnt = 0;
    auto& ivt = in_view_tag[e];
    for (int i = 0; i < static_cast<int>(ivt.size()); i++) { // NOLINT(modernize-loop-convert)
        output[cnt++] = ivt[i];
    }
    ivt.clear();
    int evt_n = static_cast<int>(edge_vertices_in_view_tag.size());
    for (int i = 0; i < evt_n; i++) { // NOLINT(modernize-loop-convert)
        output[cnt++] = edge_vertices_in_view_tag[i];
    }
    edge_vertices_in_view_tag.clear();
    int fvt_n = static_cast<int>(face_vertices_in_view_tag.size());
    for (int i = 0; i < fvt_n; i++) { // NOLINT(modernize-loop-convert)
        output[cnt++] = face_vertices_in_view_tag[i];
    }
    face_vertices_in_view_tag.clear();
}

void construct_faces( // NOLINT(readability-identifier-length, readability-identifier-naming,
                      // modernize-use-trailing-return-type)
    int e, T* final_vertices, int* cnt) {
    using namespace final_ns;
    using namespace computing;
    auto& edge_e = bipolar_edges[e];
    auto& vertices_ids = bipolar_edges_vindices[e];
    auto& unique_vertices = bipolar_edges_vertices_vector[e];
    faces.clear();
    edge_vertices.clear();
    edge_vertices_in_view_tag.clear();
    face_vertices.clear();
    face_vertices_in_view_tag.clear();
    face_vertices_map.clear();
    int nv = static_cast<int>(unique_vertices.size());
    for (int i = 0; i < static_cast<int>(edge_e.size()); i++) {
        T computed_edge[6];
        Cube c;
        keyToCube(c, edge_e[i].second);
        computeCoords(computed_edge, c.m_coords, c.m_l);
        int dir = std::abs(edge_e[i].first) - 1;
        c.m_coords[dir]++;
        computeCoords(computed_edge + 3, c.m_coords, c.m_l);
        T computed_faces[12 * 4];
        int computed_faces_level[4], computed_faces_dir[4];
        bool intersect[4] = {false, false, false, false};
        bool condition1 = true;
        for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
            int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4];
            if (v1 == v2)
                continue;
            Cube c1, c2;
            keyToCube(c1, unique_vertices[v1]);
            keyToCube(c2, unique_vertices[v2]);
            if (c1.m_l < c2.m_l) {
                Cube t = c1;
                c1 = c2;
                c2 = t;
            }
            bool found = false;
            for (int ax = 0; ax < 3; ax++) {
                for (int d = 0; d <= 1; d++) {
                    if (c1.m_coords[ax] + d == (c2.m_coords[ax] + 1 - d) << (c1.m_l - c2.m_l)) {
                        for (int d1 = 0; d1 <= 1; d1++)
                            for (int d2 = 0; d2 <= 1; d2++) {
                                Cube c1mod = c1;
                                c1mod.m_coords[ax] += d;
                                c1mod.m_coords[(ax + 1) % 3] += d1;
                                c1mod.m_coords[(ax + 2) % 3] += d2;
                                computeCoords(computed_faces + static_cast<ptrdiff_t>(12) * j +
                                                  static_cast<ptrdiff_t>(3) * (d1 * 2 + d2),
                                              c1mod.m_coords, c1mod.m_l);
                            }
                        computed_faces_level[j] = c1.m_l;
                        computed_faces_dir[j] = ax;
                        found = true;
                        break;
                    }
                }
                if (found)
                    break;
            }
            assert(found);
            if (c1.m_l == c2.m_l) {
                intersect[j] = true;
                continue;
            }
            // tri_seg_intersect should be strict
            bool i1 = triSegIntersect(computed_faces + static_cast<ptrdiff_t>(12) * j + 6,
                                      computed_faces + static_cast<ptrdiff_t>(12) * j,
                                      computed_faces + static_cast<ptrdiff_t>(12) * j + 3,
                                      final_vertices + static_cast<ptrdiff_t>(v1) * 3,
                                      final_vertices + static_cast<ptrdiff_t>(v2) * 3);
            bool i2 = triSegIntersect(computed_faces + static_cast<ptrdiff_t>(12) * j + 3,
                                      computed_faces + static_cast<ptrdiff_t>(12) * j + 9,
                                      computed_faces + static_cast<ptrdiff_t>(12) * j + 6,
                                      final_vertices + static_cast<ptrdiff_t>(v1) * 3,
                                      final_vertices + static_cast<ptrdiff_t>(v2) * 3);
            intersect[j] = i1 || i2;
            condition1 = condition1 && intersect[j];
        }
        if (!condition1) {
            ComputedVertex cv;
            for (int j = 0; j < 3; j++) // NOLINT(readability-identifier-length)
                cv.m_c[j] = (computed_edge[j] + computed_edge[j + 3]) / 2;
            cv.m_l = 0;
            cv.m_r = 0.5 * params::size / (1 << c.m_l);
            edge_vertices.emplace_back(dir, cv);
            bool edge_in_view = false;
            for (int j = 0; j < 4; j++)
                edge_in_view =
                    edge_in_view ||
                    in_view_tag[e]
                               [vertices_ids[i * 4 + j]]; // NOLINT(readability-identifier-length)
            edge_vertices_in_view_tag.push_back(edge_in_view);
            for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
                int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4];
                if (v1 == v2)
                    continue;
                if (!intersect[j]) {
                    int vf;
                    if (face_vertices_map.count(std::make_pair(v1, v2))) {
                        vf = face_vertices_map[std::make_pair(v1, v2)];
                    } else if (face_vertices_map.count(std::make_pair(v2, v1))) {
                        vf = face_vertices_map[std::make_pair(v2, v1)];
                    } else {
                        vf = static_cast<int>(face_vertices.size());
                        face_vertices_map[std::make_pair(v1, v2)] = vf;
                        ComputedVertex cv2;
                        for (int k = 0; k < 3; k++) // NOLINT(readability-identifier-length)
                            cv2.m_c[k] =
                                (computed_faces[12 * j + k + 3] + computed_faces[12 * j + k + 6]) /
                                2;
                        cv2.m_l = 0;
                        cv2.m_r = 0.5 * params::size / (1 << computed_faces_level[j]);
                        face_vertices.emplace_back(computed_faces_dir[j], cv2);
                        bool face_in_view = in_view_tag[e][v1] || in_view_tag[e][v2];
                        face_vertices_in_view_tag.push_back(face_in_view);
                    }
                    addFaces(faces, v1, -vf - 1, nv + static_cast<int>(edge_vertices.size()) - 1);
                    addFaces(faces, -vf - 1, v2, nv + static_cast<int>(edge_vertices.size()) - 1);
                } else {
                    addFaces(faces, v1, v2, nv + static_cast<int>(edge_vertices.size()) - 1);
                }
            }
        } else {
            bool condition2 = false;
            int start_j = -1;
            for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
                int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4],
                    v3 = vertices_ids[i * 4 + (j + 2) % 4];
                if (v1 == v2 || v2 == v3 || v1 == v3)
                    continue;
                if (triSegIntersect(final_vertices + static_cast<ptrdiff_t>(v1) * 3,
                                    final_vertices + static_cast<ptrdiff_t>(v2) * 3,
                                    final_vertices + static_cast<ptrdiff_t>(v3) * 3, computed_edge,
                                    computed_edge + 3)) {
                    start_j = j;
                }
            }
            if (start_j != -1) {
                int j = start_j; // NOLINT(readability-identifier-length)
                int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4],
                    v3 = vertices_ids[i * 4 + (j + 2) % 4];
                int v4 = vertices_ids[i * 4 + (j + 3) % 4];
                if (v4 == v3 || v4 == v1 ||
                    triSegIntersect(computed_edge, final_vertices + static_cast<ptrdiff_t>(v4) * 3,
                                    computed_edge + 3,
                                    final_vertices + static_cast<ptrdiff_t>(v1) * 3,
                                    final_vertices + static_cast<ptrdiff_t>(v3) * 3)) {
                    condition2 = true;
                }
            }
            if (!condition2) {
                ComputedVertex cv;
                for (int j = 0; j < 3; j++) // NOLINT(readability-identifier-length)
                    cv.m_c[j] = (computed_edge[j] + computed_edge[j + 3]) / 2;
                cv.m_l = 0;
                cv.m_r = 0.5 * params::size / (1 << c.m_l);
                edge_vertices.emplace_back(dir, cv);
                bool edge_in_view = false;
                for (int j = 0; j < 4; j++)
                    edge_in_view =
                        edge_in_view ||
                        in_view_tag[e][vertices_ids[i * 4 +
                                                    j]]; // NOLINT(readability-identifier-length)
                edge_vertices_in_view_tag.push_back(edge_in_view);
                for (int j = 0; j < 4; j++) { // NOLINT(readability-identifier-length)
                    int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4];
                    if (v1 == v2)
                        continue;
                    addFaces(faces, v1, v2, nv + static_cast<int>(edge_vertices.size()) - 1);
                }
            } else {
                int j = start_j; // NOLINT(readability-identifier-length)
                int v1 = vertices_ids[i * 4 + j], v2 = vertices_ids[i * 4 + (j + 1) % 4],
                    v3 = vertices_ids[i * 4 + (j + 2) % 4];
                addFaces(faces, v1, v2, v3);
                int v4 = vertices_ids[i * 4 + (j + 3) % 4];
                if (v4 == v1 || v4 == v3)
                    continue;
                addFaces(faces, v1, v3, v4);
            }
        }
    }
    face_vertices_map.clear();
    for (int i = 0; i < static_cast<int>(faces.size()); i++) { // NOLINT(modernize-loop-convert)
        if (faces[i] < 0)
            faces[i] = nv + static_cast<int>(edge_vertices.size()) + (-faces[i] - 1);
    }
    vertices_ids.clear();
    unique_vertices.clear();
    edge_e.clear();
    cnt[0] = static_cast<int>(edge_vertices.size());
    cnt[1] = static_cast<int>(face_vertices.size());
    cnt[2] = static_cast<int>(faces.size()) / 3;
}

void get_faces( // NOLINT(readability-identifier-naming, modernize-use-trailing-return-type)
    int* faces_output) {
    using namespace computing;
    memcpy(faces_output, &faces[0], sizeof(int) * faces.size());
    faces.clear();
}
}
