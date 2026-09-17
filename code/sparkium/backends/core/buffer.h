#pragma once
// Portable port of code/sparkium/shaders/buffer_helper.hlsli.
//
// The graphics backends use ByteAddressBuffer objects; the offline backends
// read the very same byte layouts out of a flat, 4-byte aligned blob, which
// keeps the mesh/light decoders identical to the HLSL ones.

#include "sparkium/backends/core/hlsl_math.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

// Read-only view of a byte-address blob. All Sparkium layouts keep every load
// 4-byte aligned, so the loader can index uint32_t directly.
struct ByteBufferView {
  const uint8_t *data{nullptr};

  SPARKIUM_HD uint32_t Load(uint32_t offset) const {
    const uint32_t *words = reinterpret_cast<const uint32_t *>(data);
    return words[offset >> 2];
  }
  SPARKIUM_HD uint2 Load2(uint32_t offset) const {
    const uint32_t *words = reinterpret_cast<const uint32_t *>(data);
    uint32_t index = offset >> 2;
    return uint2(words[index], words[index + 1]);
  }
  SPARKIUM_HD uint3 Load3(uint32_t offset) const {
    const uint32_t *words = reinterpret_cast<const uint32_t *>(data);
    uint32_t index = offset >> 2;
    return uint3(words[index], words[index + 1], words[index + 2]);
  }
  SPARKIUM_HD uint4 Load4(uint32_t offset) const {
    const uint32_t *words = reinterpret_cast<const uint32_t *>(data);
    uint32_t index = offset >> 2;
    return uint4(words[index], words[index + 1], words[index + 2], words[index + 3]);
  }

  SPARKIUM_HD float LoadFloat(uint32_t offset) const {
    return device::asfloat(Load(offset));
  }
  SPARKIUM_HD float2 LoadFloat2(uint32_t offset) const {
    return device::asfloat(Load2(offset));
  }
  SPARKIUM_HD float3 LoadFloat3(uint32_t offset) const {
    return device::asfloat(Load3(offset));
  }
  SPARKIUM_HD float4 LoadFloat4(uint32_t offset) const {
    return device::asfloat(Load4(offset));
  }
  SPARKIUM_HD int32_t LoadInt(uint32_t offset) const {
    return static_cast<int32_t>(Load(offset));
  }

  // `LoadFloat3x4` in buffer_helper.hlsli: three 16-byte rows transposed into a
  // float3x4 matrix whose row i is (c0[i], c1[i], c2[i], c3[i]) for the four
  // 12-byte spaced columns. Sparkium's affine transforms only ever consume it
  // through mul(transform, float4(p, 1)), which is `transform_point`.
  SPARKIUM_HD Mat4x3 LoadMat4x3(uint32_t offset) const {
    Mat4x3 result;
    result.c0 = LoadFloat3(offset + 0);
    result.c1 = LoadFloat3(offset + 12);
    result.c2 = LoadFloat3(offset + 24);
    result.c3 = LoadFloat3(offset + 36);
    return result;
  }

  // `LoadFloat4x4` (four 16-byte rows transposed). Row i of the result is row i
  // of the uploaded column-major glm matrix, so mul(matrix, v) is the standard
  // matrix product.
  SPARKIUM_HD float4x4 LoadMat4(uint32_t offset) const {
    float4x4 result;
    float4 c0 = LoadFloat4(offset + 0);
    float4 c1 = LoadFloat4(offset + 16);
    float4 c2 = LoadFloat4(offset + 32);
    float4 c3 = LoadFloat4(offset + 48);
    result.r0 = float4(c0.x, c1.x, c2.x, c3.x);
    result.r1 = float4(c0.y, c1.y, c2.y, c3.y);
    result.r2 = float4(c0.z, c1.z, c2.z, c3.z);
    result.r3 = float4(c0.w, c1.w, c2.w, c3.w);
    return result;
  }
};

// Port of StreamedBufferReference: sequential typed reads.
struct StreamedBufferReference {
  ByteBufferView buffer;
  uint32_t offset{0};

  SPARKIUM_HD float LoadFloat() {
    float result = buffer.LoadFloat(offset);
    offset += 4;
    return result;
  }
  SPARKIUM_HD float2 LoadFloat2() {
    float2 result = buffer.LoadFloat2(offset);
    offset += 8;
    return result;
  }
  SPARKIUM_HD float3 LoadFloat3() {
    float3 result = buffer.LoadFloat3(offset);
    offset += 12;
    return result;
  }
  SPARKIUM_HD int32_t LoadInt() {
    int32_t result = buffer.LoadInt(offset);
    offset += 4;
    return result;
  }
  SPARKIUM_HD uint32_t LoadUint() {
    uint32_t result = buffer.Load(offset);
    offset += 4;
    return result;
  }
  SPARKIUM_HD float4 LoadFloat4() {
    float4 result = buffer.LoadFloat4(offset);
    offset += 16;
    return result;
  }
};

// Index of a memory buffer in a "data_buffers" array. The offline scene keeps
// all meshes in one blob, so a geometry reference is just a mesh index.
SPARKIUM_HD inline ByteBufferView MeshBuffer(const DeviceScene &scene, uint32_t mesh_index) {
  ByteBufferView view;
  view.data = scene.mesh_data + scene.meshes[mesh_index].offset;
  return view;
}

}  // namespace sparkium::backends
