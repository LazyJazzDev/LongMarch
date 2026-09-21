#include "sparkium/pipelines/raytracing/core/cpu_bvh.h"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <random>
#include <set>
using namespace sparkium::raytracing;

namespace {
template <class T>
T Read(const CpuBvhTree &tree, size_t offset) {
  T value{};
  if (offset + sizeof(value) > tree.bytes.size())
    throw std::out_of_range("test tree range");
  std::memcpy(&value, tree.bytes.data() + offset, sizeof(value));
  return value;
}

void Validate(const std::vector<CpuBounds> &bounds) {
  auto tree = BuildCpuBvh(bounds);
  const auto node_count = Read<uint32_t>(tree, 0), indices = Read<uint32_t>(tree, 4);
  ASSERT_EQ(Read<uint32_t>(tree, 8), bounds.size());
  if (bounds.empty())
    return;
  std::set<uint32_t> seen, nodes;
  std::function<void(uint32_t, unsigned)> visit = [&](uint32_t index, unsigned depth) {
    ASSERT_LT(index, node_count);
    ASSERT_LE(depth, 48u);
    ASSERT_TRUE(nodes.insert(index).second);
    auto node = Read<CpuBvhNode>(tree, 16 + size_t(index) * 32);
    if (node.count) {
      for (uint32_t i = 0; i < node.count; ++i) {
        auto id = Read<uint32_t>(tree, indices + size_t(node.first + i) * 4);
        ASSERT_LT(id, bounds.size());
        EXPECT_TRUE(seen.insert(id).second);
        for (int axis = 0; axis < 3; ++axis) {
          EXPECT_LE(node.lo[axis], bounds[id].lo[axis]);
          EXPECT_GE(node.hi[axis], bounds[id].hi[axis]);
        }
      }
    } else {
      for (unsigned child = node.first; child < node.first + 2; ++child) {
        auto b = Read<CpuBvhNode>(tree, 16 + size_t(child) * 32);
        for (int axis = 0; axis < 3; ++axis) {
          EXPECT_LE(node.lo[axis], b.lo[axis]);
          EXPECT_GE(node.hi[axis], b.hi[axis]);
        }
        visit(child, depth + 1);
      }
    }
  };

  visit(0, 0);
  EXPECT_EQ(seen.size(), bounds.size());
  EXPECT_EQ(nodes.size(), node_count);
  EXPECT_EQ(tree.bytes, BuildCpuBvh(bounds).bytes);
}

TEST(CpuBvh, EmptyDegenerateAndCoincidentBounds) {
  Validate({});
  Validate({CpuBounds{{0, 0, 0}, {0, 0, 0}}});
  Validate(std::vector<CpuBounds>(257, CpuBounds{{1, 2, 3}, {1, 2, 3}}));
}

TEST(CpuBvh, BuildsDirectlyFromImmutableHostMesh) {
  auto mesh = grassland::Mesh<float>::Sphere(12, 6);
  std::vector<CpuBounds> reference(mesh.NumIndices() / 3);
  for (size_t i = 0; i < mesh.NumIndices(); ++i) {
    const auto &p = mesh.Positions()[mesh.Indices()[i]];
    reference[i / 3].Extend(glm::vec3(p.x(), p.y(), p.z()));
  }
  EXPECT_EQ(BuildCpuMeshBvh(mesh).bytes, BuildCpuBvh(reference).bytes);
}

TEST(CpuBvh, RandomAndHighlyUnbalancedDistributions) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> d(-100, 100);
  std::vector<CpuBounds> bounds;
  for (int i = 0; i < 4096; ++i) {
    glm::vec3 p{d(rng), d(rng), d(rng)};
    bounds.push_back({p, p + glm::vec3(0.01f)});
  }

  Validate(bounds);
  for (unsigned i = 0; i < bounds.size(); ++i) {
    float x = float(i * i);
    bounds[i] = {{x, 0, 0}, {x + 0.1f, 1, 1}};
  }

  Validate(bounds);
}

TEST(CpuBvh, RejectsTruncatedGeometry) {
  EXPECT_THROW(BuildCpuMeshBvh({}, 1), std::out_of_range);
  std::vector<uint8_t> bytes(52);
  uint32_t index_offset = 52;
  std::memcpy(bytes.data() + 48, &index_offset, 4);
  EXPECT_THROW(BuildCpuMeshBvh(bytes, 1), std::out_of_range);
}
}  // namespace
