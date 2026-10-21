RHI_ACCELERATION_STRUCTURE(0, 0, scene);
RHI_RW_TEXTURE2D(0, 1, float4, output_image);

struct Payload {
  float3 color;
};

struct Attributes {
  float2 bary;
};

float3 Background(float2 uv) {
  return lerp(float3(0.08, 0.12, 0.2), float3(0.6, 0.7, 0.8), uv.y);
}

RayDesc CameraRay(uint2 pixel, uint2 size) {
  float2 xy = (float2(pixel) + 0.5) / float2(size) * 2.0 - 1.0;
  xy.x *= float(size.x) / float(size.y);
  RayDesc ray;
  ray.Origin = float3(0, 0, -3);
  ray.Direction = normalize(float3(xy.x, -xy.y, 2.0));
  ray.TMin = 0.001;
  ray.TMax = 100;
  return ray;
}

float SphereHit(float3 origin, float3 direction) {
  float a = dot(direction, direction);
  float b = dot(origin, direction);
  float c = dot(origin, origin) - 0.36;
  float d = b * b - a * c;
  if (d < 0)
    return -1;
  float t = (-b - sqrt(d)) / a;
  if (t <= 0)
    t = (-b + sqrt(d)) / a;
  return t;
}
[shader("compute")][numthreads(8, 8, 1)] void Query(uint3 id
                                                    : SV_DispatchThreadID) {
  uint width, height;
  output_image.GetDimensions(width, height);
  if (id.x >= width || id.y >= height)
    return;
  RayDesc ray = CameraRay(id.xy, uint2(width, height));
  RayQuery<RAY_FLAG_NONE> query;
  query.TraceRayInline(scene, RAY_FLAG_NONE, 255, ray);
  while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      float t = SphereHit(query.CandidateObjectRayOrigin(), query.CandidateObjectRayDirection());
      if (t >= ray.TMin && t <= query.CommittedRayT())
        query.CommitProceduralPrimitiveHit(t);
    }
  }
  float3 color = Background(float2(id.xy) / float2(width, height));
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    float2 bary = query.CommittedTriangleBarycentrics();
    color = float3(1 - bary.x - bary.y, bary);
  } else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
    float3 n =
        normalize(query.CommittedObjectRayOrigin() + query.CommittedRayT() * query.CommittedObjectRayDirection());
    color = n * 0.5 + 0.5;
  }
  output_image[id.xy] = float4(color, 1);
}
