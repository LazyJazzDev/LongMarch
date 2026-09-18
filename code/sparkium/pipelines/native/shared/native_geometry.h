#pragma once

// Ports of `geometry/mesh/hit_record.hlsli`,
// `geometry/mesh/sample_primitive.hlsli` and
// `geometry_primitive_sampler.hlsli`.

#include "native_random.h"

namespace sparkium::native {

struct GeometryHeader {
  uint32_t num_vertices;
  uint32_t num_indices;
  uint32_t position_offset;
  uint32_t position_stride;
  uint32_t normal_offset;
  uint32_t normal_stride;
  uint32_t tex_coord_offset;
  uint32_t tex_coord_stride;
  uint32_t tangent_offset;
  uint32_t tangent_stride;
  uint32_t signal_offset;
  uint32_t signal_stride;
  uint32_t index_offset;
  uint32_t color_offset;
  uint32_t color_stride;
};

LM_DEVICE_FUNC inline GeometryHeader LoadGeometryHeader(const BufferReference &geometry_buffer) {
  GeometryHeader header;
  header.num_vertices = geometry_buffer.Load(0);
  header.num_indices = geometry_buffer.Load(4);
  header.position_offset = geometry_buffer.Load(8);
  header.position_stride = geometry_buffer.Load(12);
  header.normal_offset = geometry_buffer.Load(16);
  header.normal_stride = geometry_buffer.Load(20);
  header.tex_coord_offset = geometry_buffer.Load(24);
  header.tex_coord_stride = geometry_buffer.Load(28);
  header.tangent_offset = geometry_buffer.Load(32);
  header.tangent_stride = geometry_buffer.Load(36);
  header.signal_offset = geometry_buffer.Load(40);
  header.signal_stride = geometry_buffer.Load(44);
  header.index_offset = geometry_buffer.Load(48);
  header.color_offset = geometry_buffer.Load(52);
  header.color_stride = geometry_buffer.Load(56);
  return header;
}

// `world_to_object` is passed as the float3x4 stored in the instance record;
// the HLSL code receives its transpose and uses `mul(world_to_object, v3)`,
// which equals `v3 * world_to_object` here.
LM_DEVICE_FUNC inline HitRecord MakeMeshHitRecord(const SceneView &scene,
                                                  uint32_t geometry_index,
                                                  uint32_t object_index,
                                                  uint32_t primitive_index,
                                                  const float2 &barycentric,
                                                  float distance,
                                                  const float3 &ray_direction,
                                                  const float3x4 &object_to_world,
                                                  const float3x4 &world_to_object) {
  HitRecord hit_record;
  const BufferReference geometry_buffer = MakeBufferReference(scene.DataBuffer(static_cast<int>(geometry_index)), 0);
  const GeometryHeader header = LoadGeometryHeader(geometry_buffer);

  const uint3 vid = geometry_buffer.Load3(header.index_offset + primitive_index * 3 * 4);

  float3 pos[3] = {geometry_buffer.LoadF3(header.position_offset + header.position_stride * vid[0]),
                   geometry_buffer.LoadF3(header.position_offset + header.position_stride * vid[1]),
                   geometry_buffer.LoadF3(header.position_offset + header.position_stride * vid[2])};

  const float3 barycentrics{1.0f - barycentric.x - barycentric.y, barycentric.x, barycentric.y};

  hit_record.t = distance;
  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.object_position = hit_record.position;
  hit_record.object_origin = mul(object_to_world, float4{0, 0, 0, 1});
  const float3 world_pos[3] = {mul(object_to_world, float4{pos[0], 1.0f}), mul(object_to_world, float4{pos[1], 1.0f}),
                               mul(object_to_world, float4{pos[2], 1.0f})};
  hit_record.position =
      world_pos[0] * barycentrics[0] + world_pos[1] * barycentrics[1] + world_pos[2] * barycentrics[2];
  hit_record.geom_normal =
      glm::normalize(float3{mul_transposed(world_to_object, glm::cross(pos[1] - pos[0], pos[2] - pos[0]))});
  // Light-hit MIS is evaluated in world-space solid angle. Using the local
  // triangle area here loses most of the BSDF-sampled contribution for scaled
  // emitters (for example a unit quad scaled to a 30 x 30 area light).
  hit_record.pdf =
      1.0f / (glm::length(glm::cross(world_pos[1] - world_pos[0], world_pos[2] - world_pos[0])) * 0.5f);

  if (header.normal_offset != 0) {
    hit_record.normal = geometry_buffer.LoadF3(header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
                        geometry_buffer.LoadF3(header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
                        geometry_buffer.LoadF3(header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_record.normal = glm::normalize(float3{mul_transposed(world_to_object, hit_record.normal)});
  } else {
    hit_record.normal = hit_record.geom_normal;
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  if (header.tex_coord_offset != 0) {
    hit_record.tex_coord =
        geometry_buffer.LoadF2(header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        geometry_buffer.LoadF2(header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        geometry_buffer.LoadF2(header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.tex_coord = float2{0.0f, 0.0f};
  }
  if (header.color_offset != 0) {
    hit_record.color = geometry_buffer.LoadF3(header.color_offset + header.color_stride * vid[0]) * barycentrics[0] +
                       geometry_buffer.LoadF3(header.color_offset + header.color_stride * vid[1]) * barycentrics[1] +
                       geometry_buffer.LoadF3(header.color_offset + header.color_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.color = float3{1.0f, 1.0f, 1.0f};
  }

  if (header.tangent_offset != 0) {
    hit_record.tangent =
        geometry_buffer.LoadF3(header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        geometry_buffer.LoadF3(header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        geometry_buffer.LoadF3(header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_record.tangent = glm::normalize(mul(object_to_world, float4{hit_record.tangent, 0.0f}));
  } else {
    hit_record.tangent = glm::cross(hit_record.normal, float3{0.0f, 0.0f, 1.0f});
    if (glm::length(hit_record.tangent) < 0.001f) {
      hit_record.tangent = glm::cross(hit_record.normal, float3{1.0f, 0.0f, 0.0f});
    }
    hit_record.tangent = glm::normalize(hit_record.tangent);
  }

  if (header.signal_offset != 0) {
    hit_record.signal = geometry_buffer.LoadF(header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
                        geometry_buffer.LoadF(header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
                        geometry_buffer.LoadF(header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    // Zero marks the absence of a valid tangent frame. Tangent-space normal
    // maps must then leave the interpolated surface normal unchanged.
    hit_record.signal = 0.0f;
  }

  if (glm::dot(ray_direction, hit_record.normal) > 0.0f) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = static_cast<int>(object_index);
  hit_record.primitive_index = static_cast<int>(primitive_index);

  return hit_record;
}

LM_DEVICE_FUNC inline GeometryPrimitiveSample MeshSamplePrimitive(const ByteBuffer &geometry_data,
                                                                  const float3x4 &transform,
                                                                  uint32_t primitive_id,
                                                                  float2 sample) {
  const uint32_t position_offset = geometry_data.Load(8);
  const uint32_t position_stride = geometry_data.Load(12);
  const uint32_t index_offset = geometry_data.Load(48);
  const uint32_t tex_coord_offset = geometry_data.Load(24);
  const uint32_t tex_coord_stride = geometry_data.Load(28);
  const uint3 vid = geometry_data.Load3(index_offset + primitive_id * 3 * 4);
  if (sample.x + sample.y > 1.0f) {
    // Handle case where sample is outside the triangle
    sample = float2{1.0f - sample.x, 1.0f - sample.y};
  }
  const float3 barycentrics{1.0f - sample.x - sample.y, sample.x, sample.y};
  float3 pos[3];
  pos[0] = mul(transform, float4{LoadFloat3(geometry_data, position_offset + position_stride * vid[0]), 1.0f});
  pos[1] = mul(transform, float4{LoadFloat3(geometry_data, position_offset + position_stride * vid[1]), 1.0f});
  pos[2] = mul(transform, float4{LoadFloat3(geometry_data, position_offset + position_stride * vid[2]), 1.0f});

  GeometryPrimitiveSample sample_result;
  sample_result.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  sample_result.normal = glm::normalize(glm::cross(pos[1] - pos[0], pos[2] - pos[0]));
  if (tex_coord_offset != 0) {
    sample_result.tex_coord =
        LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_data, tex_coord_offset + tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    sample_result.tex_coord = float2{0.0f, 0.0f};
  }
  sample_result.pdf = 1.0f / (glm::length(glm::cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);
  return sample_result;
}

LM_DEVICE_FUNC inline void SamplePrimitivePower(const ByteBuffer &direct_lighting_sampler_data,
                                                float &r,
                                                uint32_t &primitive_id,
                                                float &prob) {
  const uint32_t primitive_count = direct_lighting_sampler_data.Load(48);
  const BufferReference power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
  const float total_power = asfloat(power_cdf.Load(primitive_count * 4 - 4));

  uint32_t L = 0, R = primitive_count - 1;
  while (L < R) {
    const uint32_t mid = (L + R) / 2;
    const float mid_power = asfloat(power_cdf.Load(mid * 4));
    if (r <= mid_power / total_power) {
      R = mid;
    } else {
      L = mid + 1;
    }
  }

  primitive_id = L;
  const float high_prob = asfloat(power_cdf.Load(L * 4)) / total_power;
  const float low_prob = (L > 0) ? asfloat(power_cdf.Load((L - 1) * 4)) / total_power : 0.0f;
  prob = high_prob - low_prob;

  r = (r - low_prob) / prob;
}

LM_DEVICE_FUNC inline float EvaluatePrimitiveProbability(const ByteBuffer &direct_lighting_sampler_data,
                                                         uint32_t primitive_id) {
  const uint32_t primitive_count = direct_lighting_sampler_data.Load(48);
  const BufferReference power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
  const float total_power = asfloat(power_cdf.Load(primitive_count * 4 - 4));
  const float high_prob = asfloat(power_cdf.Load(primitive_id * 4));
  const float low_prob = (primitive_id > 0) ? asfloat(power_cdf.Load((primitive_id - 1) * 4)) : 0.0f;
  return (high_prob - low_prob) / total_power;
}

}  // namespace sparkium::native
