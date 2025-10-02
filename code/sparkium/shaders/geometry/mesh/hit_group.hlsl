#include "bindings.hlsli"
#include "direct_lighting.hlsli"
#include "geometry/mesh/geometry_header.hlsli"
#include "material_sampler.hlsli"
#include "random.hlsli"

[shader("closesthit")] void RenderClosestHit(inout RenderContext context,
                                             in BuiltInTriangleIntersectionAttributes attr) {
  HitRecord hit_record;
  BufferReference<ByteAddressBuffer> geometry_buffer = MakeBufferReference(data_buffers[InstanceID()], 0);
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

  uint3 vid;
  vid = geometry_buffer.Load3(header.index_offset + PrimitiveIndex() * 3 * 4);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.position = mul(ObjectToWorld3x4(), float4(hit_record.position, 1.0));
  hit_record.geom_normal = normalize(mul(WorldToObject4x3(), cross(pos[1] - pos[0], pos[2] - pos[0])).xyz);
  hit_record.pdf = 1.0f / (length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);

  if (header.normal_offset != 0) {
    hit_record.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_record.normal = normalize(mul(WorldToObject4x3(), hit_record.normal).xyz);
  } else {
    hit_record.normal = hit_record.geom_normal;
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  if (header.tex_coord_offset != 0) {
    hit_record.tex_coord =
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.tex_coord = float2(0.0, 0.0);
  }

  if (header.tangent_offset != 0) {
    hit_record.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_record.tangent = normalize(mul(ObjectToWorld3x4(), float4(hit_record.tangent, 0.0)).xyz);
  } else {
    hit_record.tangent = cross(hit_record.normal, float3(0.0, 0.0, 1.0));
    if (length(hit_record.tangent) < 0.001) {
      hit_record.tangent = cross(hit_record.normal, float3(1.0, 0.0, 0.0));
    }
    hit_record.tangent = normalize(hit_record.tangent);
  }

  if (header.signal_offset != 0) {
    hit_record.signal =
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.signal = 1.0;
  }

  if (dot(WorldRayDirection(), hit_record.normal) > 0.0) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = InstanceIndex();
  hit_record.primitive_index = PrimitiveIndex();

  SampleMaterial(context, hit_record);
}

    [shader("closesthit")] void ShadowClosestHit(inout ShadowRayPayload payload,
                                                 in BuiltInTriangleIntersectionAttributes attr) {
#if defined(SAMPLE_SHADOW_NO_HITRECORD)
  SampleShadow(payload);
#else
  HitRecord hit_record;
  BufferReference<ByteAddressBuffer> geometry_buffer = MakeBufferReference(data_buffers[InstanceID()], 0);
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

  uint3 vid;
  vid = geometry_buffer.Load3(header.index_offset + PrimitiveIndex() * 3 * 4);

  float3 pos[3] = {LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[0]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[1]),
                   LoadFloat3(geometry_buffer, header.position_offset + header.position_stride * vid[2])};

  float3 barycentrics =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  hit_record.front_facing = true;
  hit_record.position = pos[0] * barycentrics[0] + pos[1] * barycentrics[1] + pos[2] * barycentrics[2];
  hit_record.position = mul(ObjectToWorld3x4(), float4(hit_record.position, 1.0));
  hit_record.geom_normal = normalize(mul(WorldToObject4x3(), cross(pos[1] - pos[0], pos[2] - pos[0])).xyz);
  hit_record.pdf = 1.0f / (length(cross(pos[1] - pos[0], pos[2] - pos[0])) * 0.5f);

  if (header.normal_offset != 0) {
    hit_record.normal =
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.normal_offset + header.normal_stride * vid[2]) * barycentrics[2];
    hit_record.normal = normalize(mul(WorldToObject4x3(), hit_record.normal).xyz);
  } else {
    hit_record.normal = hit_record.geom_normal;
  }
  // normal transformation need to multiply inverse transpose of the object to world matrix
  if (header.tex_coord_offset != 0) {
    hit_record.tex_coord =
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[0]) * barycentrics[0] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[1]) * barycentrics[1] +
        LoadFloat2(geometry_buffer, header.tex_coord_offset + header.tex_coord_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.tex_coord = float2(0.0, 0.0);
  }

  if (header.tangent_offset != 0) {
    hit_record.tangent =
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[0]) * barycentrics[0] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[1]) * barycentrics[1] +
        LoadFloat3(geometry_buffer, header.tangent_offset + header.tangent_stride * vid[2]) * barycentrics[2];
    hit_record.tangent = normalize(mul(ObjectToWorld3x4(), float4(hit_record.tangent, 0.0)).xyz);
  } else {
    hit_record.tangent = float3(0.0, 0.0, 0.0);
  }

  if (header.signal_offset != 0) {
    hit_record.signal =
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[0]) * barycentrics[0] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[1]) * barycentrics[1] +
        LoadFloat(geometry_buffer, header.signal_offset + header.signal_stride * vid[2]) * barycentrics[2];
  } else {
    hit_record.signal = 1.0;
  }

  if (dot(WorldRayDirection(), hit_record.normal) > 0.0) {
    hit_record.front_facing = false;
    hit_record.geom_normal = -hit_record.geom_normal;
    hit_record.normal = -hit_record.normal;
    hit_record.tangent = -hit_record.tangent;
    hit_record.signal = -hit_record.signal;
  }

  hit_record.object_index = InstanceIndex();
  hit_record.primitive_index = PrimitiveIndex();

  SampleShadow(payload, hit_record);
#endif
}
