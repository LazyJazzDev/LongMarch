// Checks the CPU backend's BVH against an oracle.
//
// The tree is built by bvh_builder and then walked by the shader's own
// traversal (software/traversal.hlsli, compiled as C++), so this covers both
// the builder and the layout the two backends share. The reference is a
// brute force intersection over every triangle, which is independent of both.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "sparkium/pipelines/raytracing/cpu/bvh_builder.h"
#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_render.h"

using sparkium::raytracing::cpu::BuildBvh;
using sparkium::raytracing::cpu::BvhNodeCount;
using sparkium::raytracing::cpu::BvhPrimitive;
using sparkium::raytracing::cpu::SoftwareNode;
using sparkium_cpu_shaders::ByteAddressBuffer;
using sparkium_cpu_shaders::Float2;
using sparkium_cpu_shaders::Float3;
using sparkium_cpu_shaders::HitRecord;
using sparkium_cpu_shaders::InlineIntersect;
using sparkium_cpu_shaders::RayDesc;
using sparkium_cpu_shaders::SamplerState;
using sparkium_cpu_shaders::SoftwareHit;

// The shader-graph sampler is part of the same translation unit and needs an
// implementation; these tests never reach it.
namespace sparkium_cpu_shaders {
GraphSurface EvaluateShaderGraph(HitRecord, Float3, int, int, bool, ByteAddressBuffer) {
  return GraphSurface{};
}
}  // namespace sparkium_cpu_shaders

namespace {

// The geometry byte layout GeometryMesh uploads.
std::vector<uint8_t> MakeGeometryBuffer(const std::vector<float> &positions,
                                        const std::vector<uint32_t> &indices) {
  std::vector<uint8_t> data(60, 0);
  uint32_t header[15] = {};
  header[0] = static_cast<uint32_t>(positions.size() / 3);
  header[1] = static_cast<uint32_t>(indices.size());
  header[2] = 60;
  header[3] = sizeof(float) * 3;
  header[12] = 60 + static_cast<uint32_t>(positions.size() * sizeof(float));
  std::memcpy(data.data(), header, sizeof(header));
  const uint8_t *positions_bytes = reinterpret_cast<const uint8_t *>(positions.data());
  data.insert(data.end(), positions_bytes, positions_bytes + positions.size() * sizeof(float));
  const uint8_t *index_bytes = reinterpret_cast<const uint8_t *>(indices.data());
  data.insert(data.end(), index_bytes, index_bytes + indices.size() * sizeof(uint32_t));
  return data;
}

// Moller-Trumbore in double precision, used only as the reference.
double TriangleDistance(const float *a, const float *b, const float *c, const float *origin,
                        const float *direction) {
  const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double p[3] = {direction[1] * e2[2] - direction[2] * e2[1],
                       direction[2] * e2[0] - direction[0] * e2[2],
                       direction[0] * e2[1] - direction[1] * e2[0]};
  const double determinant = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (std::abs(determinant) < 1e-12)
    return -1.0;
  const double inverse = 1.0 / determinant;
  const double t[3] = {origin[0] - a[0], origin[1] - a[1], origin[2] - a[2]};
  const double u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * inverse;
  if (u < 0.0 || u > 1.0)
    return -1.0;
  const double q[3] = {t[1] * e1[2] - t[2] * e1[1], t[2] * e1[0] - t[0] * e1[2],
                       t[0] * e1[1] - t[1] * e1[0]};
  const double v = (direction[0] * q[0] + direction[1] * q[1] + direction[2] * q[2]) * inverse;
  if (v < 0.0 || u + v > 1.0)
    return -1.0;
  return (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inverse;
}

}  // namespace

TEST(SparkiumCpuBackend, BvhTraversalMatchesABruteForceOracle) {
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

  constexpr int kTriangles = 500;
  std::vector<float> positions;
  std::vector<uint32_t> indices;
  for (int i = 0; i < kTriangles; ++i) {
    const float centre[3] = {unit(generator) * 4.0f, unit(generator) * 4.0f, unit(generator) * 4.0f};
    for (int vertex = 0; vertex < 3; ++vertex) {
      for (int axis = 0; axis < 3; ++axis)
        positions.push_back(centre[axis] + unit(generator) * 0.4f);
      indices.push_back(static_cast<uint32_t>(i * 3 + vertex));
    }
  }
  const std::vector<uint8_t> geometry_bytes = MakeGeometryBuffer(positions, indices);

  std::vector<BvhPrimitive> primitives(kTriangles);
  for (int i = 0; i < kTriangles; ++i) {
    float low[3] = {1e30f, 1e30f, 1e30f};
    float high[3] = {-1e30f, -1e30f, -1e30f};
    for (int vertex = 0; vertex < 3; ++vertex)
      for (int axis = 0; axis < 3; ++axis) {
        const float value = positions[(i * 3 + vertex) * 3 + axis];
        low[axis] = std::min(low[axis], value);
        high[axis] = std::max(high[axis], value);
      }
    std::copy(low, low + 3, primitives[i].lo);
    std::copy(high, high + 3, primitives[i].hi);
    primitives[i].index = static_cast<uint32_t>(i);
  }

  // The top level tree is rooted at node 0, which InlineIntersect assumes; the
  // one mesh tree follows it, exactly as the pipeline lays them out.
  const size_t tlas_nodes = BvhNodeCount(1);
  std::vector<SoftwareNode> nodes(tlas_nodes + BvhNodeCount(primitives.size()));
  const uint32_t mesh_root = static_cast<uint32_t>(tlas_nodes);
  BuildBvh(nodes, mesh_root, primitives);

  BvhPrimitive instance{};
  std::copy(nodes[mesh_root].lo, nodes[mesh_root].lo + 3, instance.lo);
  std::copy(nodes[mesh_root].hi, nodes[mesh_root].hi + 3, instance.hi);
  instance.index = 0;
  std::vector<BvhPrimitive> instances{instance};
  BuildBvh(nodes, 0, instances);

  std::vector<uint8_t> instance_bytes(16 + 112, 0);
  const uint32_t instance_count = 1;
  std::memcpy(instance_bytes.data(), &instance_count, sizeof(instance_count));
  // glm::mat4x3 is column major, four columns of three, with the translation in
  // the fourth; this is what LoadFloat3x4 reads back.
  const float identity[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  std::memcpy(instance_bytes.data() + 16, identity, sizeof(identity));
  std::memcpy(instance_bytes.data() + 16 + 48, identity, sizeof(identity));
  uint32_t *info = reinterpret_cast<uint32_t *>(instance_bytes.data() + 16 + 96);
  info[0] = mesh_root;
  info[1] = 0;
  info[2] = 0;
  info[3] = kTriangles;

  sparkium_cpu_shaders::data_buffers.assign(7, ByteAddressBuffer{});
  sparkium_cpu_shaders::data_buffers[0] = ByteAddressBuffer(geometry_bytes.data(), geometry_bytes.size());
  sparkium_cpu_shaders::software_data_buffer_count = 1;
  sparkium_cpu_shaders::data_buffers[sparkium_cpu_shaders::software_data_buffer_count + 5] =
      ByteAddressBuffer(instance_bytes.data(), instance_bytes.size());
  sparkium_cpu_shaders::software_nodes =
      ByteAddressBuffer(nodes.data(), nodes.size() * sizeof(SoftwareNode));
  sparkium_cpu_shaders::samplers.assign(2, SamplerState{});

  int hits = 0;
  int mismatches = 0;
  constexpr int kRays = 3000;
  for (int i = 0; i < kRays; ++i) {
    const float origin[3] = {unit(generator) * 8.0f, unit(generator) * 8.0f, unit(generator) * 8.0f};
    float direction[3] = {unit(generator), unit(generator), unit(generator)};
    const float length =
        std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (length < 1e-6f)
      continue;
    for (float &value : direction)
      value /= length;

    double expected = 1e9;
    for (int triangle = 0; triangle < kTriangles; ++triangle) {
      const double distance = TriangleDistance(&positions[indices[triangle * 3 + 0] * 3],
                                               &positions[indices[triangle * 3 + 1] * 3],
                                               &positions[indices[triangle * 3 + 2] * 3], origin, direction);
      if (distance > 1e-4 && distance < expected)
        expected = distance;
    }

    RayDesc ray;
    ray.Origin = Float3(origin[0], origin[1], origin[2]);
    ray.Direction = Float3(direction[0], direction[1], direction[2]);
    ray.TMin = 1e-3f;
    ray.TMax = 1e9f;
    SoftwareHit hit;
    const bool found = InlineIntersect(ray, false, hit);

    if (expected < 1e8) {
      hits++;
      if (!found || std::abs(hit.distance - expected) > 1e-3 * (1.0 + expected))
        mismatches++;
    } else if (found && hit.distance < 1e8) {
      mismatches++;
    }
  }

  EXPECT_GT(hits, 50) << "the test rays barely hit anything, so it proves little";
  EXPECT_EQ(mismatches, 0) << "the shader traversal and the oracle disagree";
}
