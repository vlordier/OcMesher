// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root
// directory of this source tree.

// Authors: Zeyu Ma

#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using T = double;
using sdfT = float; // NOLINT(readability-identifier-naming)

struct ComputedVertex {
    T m_c[3], m_l, m_r; // NOLINT(modernize-avoid-c-arrays)
};

struct Cube {
    int m_coords[3], m_l; // NOLINT(modernize-avoid-c-arrays)
};

using Vertex = Cube;
using Int3 = std::pair<int, std::pair<int, int>>;
using KeyCube = std::pair<int, std::pair<int, std::pair<int, int>>>;
using KeyEdge = std::pair<int, KeyCube>;

// Hash functor for KeyCube used by unordered containers.
struct KeyCubeHash {
    auto operator()(const KeyCube& k) const noexcept -> std::size_t {
        // Combine four ints via bit-mixing.
        std::size_t h = std::hash<int>{}(k.first);
        h ^= std::hash<int>{}(k.second.first) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.second.second.first) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.second.second.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Hash functor for std::pair<int, KeyCube> (bipolar edge keys).
struct IntKeyCubeHash {
    auto operator()(const std::pair<int, KeyCube>& k) const noexcept -> std::size_t {
        KeyCubeHash kch;
        std::size_t h = std::hash<int>{}(k.first);
        h ^= kch(k.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Node {
    Cube m_c;
    int m_nxts[8]; // NOLINT(modernize-avoid-c-arrays)
};

inline auto intLog(T x) -> int { // NOLINT(modernize-use-trailing-return-type)
    return static_cast<int>(std::max(T(0), static_cast<T>(std::ceil(std::log2(x)))));
}

inline auto cubex(int x) -> int { // NOLINT(modernize-use-trailing-return-type)
    return x * x * x;
}

inline auto cubeIndex(int x, int y, int z,
                      int s) -> int { // NOLINT(modernize-use-trailing-return-type)
    return x * s * s + y * s + z;
}

inline auto makeInt3(int x, int y, int z) -> Int3 { // NOLINT(modernize-use-trailing-return-type)
    return std::make_pair(x, std::make_pair(y, z));
}

inline auto cubeToKey(const Cube& c) -> KeyCube { // NOLINT(modernize-use-trailing-return-type)
    return std::make_pair(c.m_coords[0],
                          std::make_pair(c.m_coords[1], std::make_pair(c.m_coords[2], c.m_l)));
}

inline void keyToCube(Cube& c, const KeyCube& key) {
    c.m_coords[0] = key.first;
    c.m_coords[1] = key.second.first;
    c.m_coords[2] = key.second.second.first;
    c.m_l = key.second.second.second;
}

inline void assign(int& x, int y, int a, int b) {
    assert(y < (std::numeric_limits<int>::max() - std::max(0, b)) / a);
    x = y * a + b;
}

inline auto isLeafNode(const Node& n) -> bool { // NOLINT(modernize-use-trailing-return-type)
    return n.m_nxts[0] <= -2;
}

inline void markLeafNode(Node& n) {
    n.m_nxts[0] = -2;
}

inline void markGridNode(Node& n, int gl) {
    n.m_nxts[0] = -2 - gl;
}

inline auto gridNodeLevel(const Node& n) -> int { // NOLINT(modernize-use-trailing-return-type)
    return -n.m_nxts[0] - 2;
}

inline auto isRegular(const Cube& c) -> bool { // NOLINT(modernize-use-trailing-return-type)
    return c.m_l >= 0;
}

inline auto isBoundary(const Cube& c) -> bool { // NOLINT(modernize-use-trailing-return-type)
    return c.m_l == -1;
}

inline auto isExterior(const Cube& c) -> bool { // NOLINT(modernize-use-trailing-return-type)
    return c.m_l == -2;
}

inline void markBoundary(Cube& c) {
    c.m_l = -1;
}

inline void markExterior(Cube& c) {
    c.m_l = -2;
}

inline void addFaces(std::vector<int>& faces, int v1, int v2, int v3) {
    assert(v1 != v2 && v2 != v3 && v3 != v1);
    faces.push_back(v1);
    faces.push_back(v2);
    faces.push_back(v3);
}

inline auto firstDigit(int j) -> int { // NOLINT(readability-identifier-length)
    return j & 1;
}

inline auto secondDigit(int j) -> int { // NOLINT(readability-identifier-length)
    return (j >> 1) & 1;
}

namespace params {
int n_cams, memory_limit_mb, coarse_count,
    n_elements;   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
T *center, *cams; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
T size, pixels_per_cube, occ_scale,
    min_dist; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// Pre-computed pixel angular threshold per camera (cached to avoid atan per call).
std::vector<T> cam_pix_ang_ppc; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

inline void precompute_cam_pix_ang() {
    cam_pix_ang_ppc.resize(static_cast<std::size_t>(n_cams));
    for (int k = 0; k < n_cams; k++) {
        T* current_cam = cams + static_cast<ptrdiff_t>(k) * (12 + 9 + 2);
        T w = current_cam[22];
        T pix_ang = std::atan(w / 2 / current_cam[12]) * 2 / w;
        cam_pix_ang_ppc[static_cast<std::size_t>(k)] = pix_ang * pixels_per_cube;
    }
}
} // namespace params

void enumerateVertices(Vertex* v, const Node& n) {
    assert(isLeafNode(n));
    int s = gridNodeLevel(n), ss = 1 << s;
    for (int i = 0; i <= ss; i++)
        for (int j = 0; j <= ss; j++)       // NOLINT(readability-identifier-length)
            for (int k = 0; k <= ss; k++) { // NOLINT(readability-identifier-length)
                int vid = i + (ss + 1) * j + (ss + 1) * (ss + 1) * k;
                for (int p = 0; p < 3; p++)
                    v[vid].m_coords[p] = n.m_c.m_coords[p] * ss + (p == 0 ? i : (p == 1 ? j : k));
                v[vid].m_l = n.m_c.m_l + s;
                for (;;) {
                    if (v[vid].m_l == 0)
                        break;
                    bool flag = true;
                    for (int p = 0; p < 3; p++) // NOLINT(modernize-loop-convert)
                        if ((v[vid].m_coords[p] & 1) != 0) {
                            flag = false;
                            break;
                        }
                    if (!flag)
                        break;
                    for (int p = 0; p < 3; p++) // NOLINT(modernize-loop-convert)
                        v[vid].m_coords[p] >>= 1;
                    v[vid].m_l--;
                }
            }
}

void computeCoords(T* coords, int* icoords, int level) {
    for (int j = 0; j < 3; j++) // NOLINT(readability-identifier-length)
        coords[j] = params::center[j] - params::size / 2 + params::size * icoords[j] / (1 << level);
}

void computeCenter(T* coords, const Cube& v) {
    for (int j = 0; j < 3; j++) // NOLINT(readability-identifier-length)
        coords[j] = params::center[j] - params::size / 2 +
                    params::size * (v.m_coords[j] + 0.5) / (1 << v.m_l);
}

void projectedCoords(                         // NOLINT(readability-identifier-length)
    const Cube& c, int k, T* icoords, T* r) { // NOLINT(bugprone-easily-swappable-parameters)
    using namespace params;
    T pw[3], pc[3]; // NOLINT(modernize-avoid-c-arrays)
    T* current_cam = cams + static_cast<ptrdiff_t>(k) * (12 + 9 + 2);
    computeCenter(pw, c);
    for (int i = 0; i < 3; i++) {
        pc[i] = current_cam[static_cast<ptrdiff_t>(i) * 4 + 3];
        for (int j = 0; j < 3; j++) { // NOLINT(readability-identifier-length)
            pc[i] += pw[j] * current_cam[static_cast<ptrdiff_t>(i) * 4 + j];
        }
    }
    if (r != nullptr) {
        *r = std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
        *r = std::max(*r, min_dist);
    }
    if (icoords != nullptr) {
        for (int i = 0; i < 3; i++) {
            icoords[i] = 0;
            for (int j = 0; j < 3; j++) { // NOLINT(readability-identifier-length)
                icoords[i] += pc[j] * current_cam[12 + static_cast<ptrdiff_t>(i) * 3 + j];
            }
        }
        icoords[0] /= icoords[2];
        icoords[1] /= icoords[2];
    }
}

auto projectedSize(const Cube& c, int k)
    -> T { // NOLINT(readability-identifier-length, modernize-use-trailing-return-type)
    using namespace params;
    T r; // NOLINT(readability-identifier-length)
    projectedCoords(c, k, nullptr, &r);
    return size / (1 << c.m_l) / r / cam_pix_ang_ppc[static_cast<std::size_t>(k)];
}

auto projectedSize(const Cube& c) -> T { // NOLINT(modernize-use-trailing-return-type)
    T max_size = 0;
    for (int k = 0; k < params::n_cams; k++) { // NOLINT(readability-identifier-length)
        T size_k = projectedSize(c, k);
        max_size = std::max(max_size, size_k);
    }
    return max_size;
}

void expandOctree(std::vector<Node>& nodes, int index) {
    assert(isLeafNode(nodes[index]));
    for (int i = 0; i < 8; i++) {
        Node* current = &nodes[index];
        current->m_nxts[i] = static_cast<int>(nodes.size());
        Node node0;
        markLeafNode(node0);
        node0.m_c.m_l = current->m_c.m_l + 1;
        for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
            int offset = (i >> k) & 1;
            assign(node0.m_c.m_coords[k], current->m_c.m_coords[k], 2, offset);
        }
        nodes.push_back(node0);
    }
}

void partialExpandOctree(std::vector<Node>& nodes,
                         int index, // NOLINT(bugprone-easily-swappable-parameters)
                         int instance) {
    assert(!isLeafNode(nodes[index]));
    for (int i = 0; i < 8; i++) {
        Node* current = &nodes[index];
        if ((instance >> i) & 1) {
            if (current->m_nxts[i] == -1) {
                current->m_nxts[i] = static_cast<int>(nodes.size());
                Node node0;
                memset(node0.m_nxts, -1, 8 * sizeof(int));
                node0.m_c.m_l = current->m_c.m_l + 1;
                for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                    int offset = (i >> k) & 1;
                    assign(node0.m_c.m_coords[k], current->m_c.m_coords[k], 2, offset);
                }
                nodes.push_back(node0);
            }
        }
    }
}

auto search(Node* nodes, int* coords, int depth, // NOLINT(modernize-use-trailing-return-type)
            bool include_exterior = false) -> Cube {
    assert(depth > 0);
    Cube res;
    for (int i = 0; i < 3; i++)
        if (!(coords[i] > 0 && coords[i] < (1 << depth))) {
            markBoundary(res);
            return res;
        }
    Node current = nodes[0];
    int current_id = 0;
    for (int l = 0; l < depth - 1; l++) { // NOLINT(readability-identifier-length)
        if (isLeafNode(current))
            break;
        for (int i = 0; i < 3; i++)
            if ((coords[i] & ((1 << (depth - current.m_c.m_l - 1)) - 1)) == 0) {
                markBoundary(res);
                return res;
            }
        int nxt = 0;
        for (int p = 0; p < 3; p++)
            if (2 * current.m_c.m_coords[p] + 1 <= (coords[p] >> (depth - current.m_c.m_l - 1)))
                nxt += (1 << p);
        current_id = current.m_nxts[nxt];
        if (current_id == -1) {
            if (!include_exterior) {
                markExterior(res);
            } else {
                for (int k = 0; k < 3; k++) { // NOLINT(readability-identifier-length)
                    int offset = (nxt >> k) & 1;
                    assign(res.m_coords[k], current.m_c.m_coords[k], 2, offset);
                }
                res.m_l = current.m_c.m_l + 1;
            }
            return res;
        }
        current = nodes[current_id];
    }
    if (!isLeafNode(current)) {
        markBoundary(res);
        return res;
    }
    int gl = gridNodeLevel(current);
    if (depth <= current.m_c.m_l + gl) {
        markBoundary(res);
        return res;
    }
    for (int i = 0; i < 3; i++)
        if ((coords[i] & ((1 << (depth - current.m_c.m_l - gl)) - 1)) == 0) {
            markBoundary(res);
            return res;
        }
    res.m_l = current.m_c.m_l + gl;
    for (int p = 0; p < 3; p++)
        res.m_coords[p] = coords[p] >> (depth - res.m_l);
    return res;
}

auto divideToCube(std::vector<Node>& nodes,
                  const Cube& c) -> int { // NOLINT(modernize-use-trailing-return-type)
    Node current = nodes[0];
    int current_id = 0;
    for (;;) {
        assert(!isLeafNode(current));
        bool flag = true;
        for (int p = 0; p < 3; p++)
            flag &= (current.m_c.m_coords[p] == c.m_coords[p]);
        if (flag && current.m_c.m_l == c.m_l) {
            markLeafNode(nodes[current_id]);
            return current_id;
        }
        int nxt = 0;
        for (int p = 0; p < 3; p++)
            if (2 * current.m_c.m_coords[p] < (c.m_coords[p] >> (c.m_l - current.m_c.m_l - 1)))
                nxt += (1 << p);
        partialExpandOctree(nodes, current_id, 1 << nxt);
        current = nodes[current_id];
        current_id = current.m_nxts[nxt];
        current = nodes[current_id];
    }
}

void findEdges(const Node& n, std::unordered_map<KeyCube, int, KeyCubeHash>& vertices, sdfT* sdf,
               std::vector<std::vector<KeyEdge>>& bipolar_edges) {
    int s = gridNodeLevel(n), ss = 1 << s;
    Vertex v[cubex(ss + 1)]; // NOLINT(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    sdfT* sdf_v[cubex(ss + 1)];
    enumerateVertices(v, n);
    for (int i = 0; i < cubex(ss + 1); i++) {
        sdf_v[i] = sdf + static_cast<ptrdiff_t>(vertices[cubeToKey(v[i])]) * params::n_elements;
    }
    for (int edir = 0; edir < 3; edir++)
        for (int i = 0; i < ss; i++)
            for (int j = 0; j <= ss; j++)       // NOLINT(readability-identifier-length)
                for (int k = 0; k <= ss; k++) { // NOLINT(readability-identifier-length)
                    int coords[3];              // NOLINT(modernize-avoid-c-arrays)
                    coords[edir] = i + 1;
                    coords[(edir + 1) % 3] = j;
                    coords[(edir + 2) % 3] = k;
                    int vid = coords[0] + coords[1] * (ss + 1) + coords[2] * (ss + 1) * (ss + 1);
                    sdfT* sdf1 = sdf_v[vid];
                    coords[edir]--;
                    vid = coords[0] + coords[1] * (ss + 1) + coords[2] * (ss + 1) * (ss + 1);
                    sdfT* sdf2 = sdf_v[vid];
                    sdfT sdf1_min = std::numeric_limits<sdfT>::infinity(),
                         sdf2_min = std::numeric_limits<sdfT>::infinity();
                    for (int e = 0; e < params::n_elements;
                         e++) { // NOLINT(readability-identifier-length)
                        sdf1_min = std::min(sdf1[e], sdf1_min);
                        sdf2_min = std::min(sdf2[e], sdf2_min);
                        if ((sdf1[e] >= 0) != (sdf2[e] >= 0)) {
                            Cube c0;
                            for (int p = 0; p < 3; p++)
                                assign(c0.m_coords[p], n.m_c.m_coords[p], ss, coords[p]);
                            c0.m_l = n.m_c.m_l + s;
                            int dir = edir + 1;
                            if (sdf1[e] < 0)
                                dir *= -1;
                            bipolar_edges[e].emplace_back(dir, cubeToKey(c0));
                        }
                    }
                    if ((sdf1_min >= 0) != (sdf2_min >= 0)) {
                        Cube c0;
                        for (int p = 0; p < 3; p++)
                            assign(c0.m_coords[p], n.m_c.m_coords[p], ss, coords[p]);
                        c0.m_l = n.m_c.m_l + s;
                        int dir = edir + 1;
                        if (sdf1_min < 0)
                            dir *= -1;
                        bipolar_edges[params::n_elements].emplace_back(dir, cubeToKey(c0));
                    }
                }
}

auto computeBoundary(const Cube& c,
                     const Cube& bound) -> int { // NOLINT(modernize-use-trailing-return-type)
    int b[3][2];                                 // NOLINT(modernize-avoid-c-arrays)
    for (int i = 0; i < 3; i++)
        for (int p = 0; p < 2; p++) {
            b[i][p] = static_cast<int>((c.m_coords[i] + p) ==
                                       ((bound.m_coords[i] + p) << (c.m_l - bound.m_l)));
        }
    int instance = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 3; j++) // NOLINT(readability-identifier-length)
            if (b[j][(i >> j) & 1]) {
                instance |= 1 << i;
                break;
            }
    }
    return instance;
}

inline auto
det(T matrix[3][3]) -> T { // NOLINT(modernize-use-trailing-return-type, modernize-avoid-c-arrays)
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

auto triSegIntersect(T* t1, T* t2, T* t3, T* s1, T* s2)
    -> bool { // NOLINT(modernize-use-trailing-return-type, bugprone-easily-swappable-parameters)
    // note (t1,t3) of the tri is allowed to intersect
    T m[3][3]; // NOLINT(modernize-avoid-c-arrays)
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = t3[i] - s1[i];
    }
    T det1 = det(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s2[i];
        m[1][i] = t2[i] - s2[i];
        m[2][i] = t3[i] - s2[i];
    }
    T det2 = det(m);
    if (!((det1 > 0 && det2 < 0) || (det1 < 0 && det2 > 0)))
        return false;

    for (int i = 0; i < 3; i++) {
        m[0][i] = t1[i] - s1[i];
        m[1][i] = t2[i] - s1[i];
        m[2][i] = s2[i] - s1[i];
    }
    det1 = det(m);
    for (int i = 0; i < 3; i++) {
        m[0][i] = t2[i] - s1[i];
        m[1][i] = t3[i] - s1[i];
    }
    det2 = det(m);
    if (!((det1 > 0 && det2 > 0) || (det1 < 0 && det2 < 0)))
        return false;
    for (int i = 0; i < 3; i++) {
        m[0][i] = t3[i] - s1[i];
        m[1][i] = t1[i] - s1[i];
    }
    det1 = det(m);
    if (!((det1 >= 0 && det2 > 0) || (det1 <= 0 && det2 < 0)))
        return false;
    return true;
}
