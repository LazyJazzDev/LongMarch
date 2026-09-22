struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

RaytracingAccelerationStructure scene : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);

bool TraceScene(RayDesc ray, uint flags, float scale, float3 bias, out float3 color) {
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, flags, 0xFF, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      // Intersect the unit sphere in object space. Do not normalize the direction:
      // its length preserves world-ray t under the ellipsoid's nonuniform scale.
      float3 origin = query.CandidateObjectRayOrigin();
      float3 direction = query.CandidateObjectRayDirection();
      float a = dot(direction, direction);
      float b = dot(origin, direction);
      float c = dot(origin, origin) - 1.0;
      float discriminant = b * b - a * c;
      if (discriminant >= 0.0) {
        float root = sqrt(discriminant);
        float t = (-b - root) / a;
        if (t <= ray.TMin)
          t = (-b + root) / a;
        float closest = query.CommittedStatus() == COMMITTED_NOTHING ? ray.TMax : query.CommittedRayT();
        if (t > ray.TMin && t <= closest)
          query.CommitProceduralPrimitiveHit(t);
      }
    }
  }
  color = float3(0.8, 0.7, 0.6);
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    float2 bary = query.CommittedTriangleBarycentrics();
    color = scale * float3(1.0 - bary.x - bary.y, bary.x, bary.y) + bias;
    return true;
  }
  if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
    float3 normal = query.CommittedObjectRayOrigin() + query.CommittedRayT() * query.CommittedObjectRayDirection();
    normal = normalize(mul(normal, (float3x3)query.CommittedWorldToObject3x4()));
    // Equivalent to SphereClosestHitMain followed by CallableMain in the RT demo.
    color = (max(dot(normal, normalize(float3(-3.0, 1.0, 2.0))), 0.0) * 0.5 + 0.5) * float3(0.6, 0.7, 0.8);
    return true;
  }
  return false;
}

[numthreads(8, 8, 1)] void CSMain(uint3 pixel
                                  : SV_DispatchThreadID) {
  uint width, height;
  output.GetDimensions(width, height);
  if (pixel.x >= width || pixel.y >= height)
    return;
  float2 uv = (float2(pixel.xy) + 0.5) / float2(width, height);
  uv.y = 1.0 - uv.y;
  float4 target = mul(camera_info.screen_to_camera, float4(uv * 2.0 - 1.0, 1, 1));
  RayDesc ray;
  ray.Origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1)).xyz;
  ray.Direction = mul(camera_info.camera_to_world, float4(target.xyz, 0)).xyz;
  ray.TMin = 0.001;
  ray.TMax = 10000.0;
  float3 color;
  if (!TraceScene(ray, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 1.0, float3(0, 0, 0), color))
    TraceScene(ray, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, -1.0, float3(1, 1, 1), color);
  output[pixel.xy] = float4(color, 1);
}
