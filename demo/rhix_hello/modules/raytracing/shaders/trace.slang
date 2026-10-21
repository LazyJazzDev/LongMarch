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

[shader("raygeneration")] void RayGen() {
  uint2 pixel = DispatchRaysIndex().xy;
  uint2 size = DispatchRaysDimensions().xy;
  RayDesc ray = CameraRay(pixel, size);
  Payload p;
  p.color = Background(float2(pixel) / float2(size));
  TraceRay(scene, RAY_FLAG_NONE, 255, 0, 1, 0, ray, p);
  output_image[pixel] = float4(p.color, 1);
}[shader("miss")] void Miss(inout Payload
                                p){}[shader("closesthit")] void TriangleHit(inout Payload p,
                                                                            BuiltInTriangleIntersectionAttributes
                                                                                attr) {
  p.color = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics);
}

[shader("intersection")] void SphereIntersection() {
  float t = SphereHit(ObjectRayOrigin(), ObjectRayDirection());
  if (t >= RayTMin() && t <= RayTCurrent()) {
    Attributes attr;
    attr.bary = float2(0, 0);
    ReportHit(t, 0, attr);
  }
}[shader("closesthit")] void SphereClosest(inout Payload p, Attributes attr) {
  float3 n = normalize(ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection());
  p.color = n * 0.5 + 0.5;
}
