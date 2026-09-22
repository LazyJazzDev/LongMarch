
struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

RaytracingAccelerationStructure as : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);

struct RayPayload {
  float3 color;
  bool hit;
  float a;
  float3 b;
};

[shader("raygeneration")] void RayGenMain() {
  float2 pixel_center = (float2)DispatchRaysIndex() + float2(0.5, 0.5);
  float2 uv = pixel_center / float2(DispatchRaysDimensions().xy);
  uv.y = 1.0 - uv.y;
  float2 d = uv * 2.0 - 1.0;
  float4 origin = mul(camera_info.camera_to_world, float4(0, 0, 0, 1));
  float4 target = mul(camera_info.screen_to_camera, float4(d, 1, 1));
  float4 direction = mul(camera_info.camera_to_world, float4(target.xyz, 0));

  float t_min = 0.001;
  float t_max = 10000.0;

  RayPayload payload;
  payload.color = float3(0, 0, 0);

  RayDesc ray;
  ray.Origin = origin.xyz;
  ray.Direction = direction.xyz;
  ray.TMin = t_min;
  ray.TMax = t_max;

  payload.hit = false;
  payload.a = 1.0;
  payload.b = float3(0, 0, 0);
  TraceRay(as, RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, 1, 0, ray,
           payload);

  payload.a = -1.0;
  payload.b = float3(1, 1, 1);
  TraceRay(as, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 0, 1, 0, ray,
           payload);

  output[DispatchRaysIndex().xy] = float4(payload.color, 1);
}

    [shader("miss")] void MissMain(inout RayPayload payload) {
  if (!payload.hit) {
    payload.color = float3(0.8, 0.7, 0.6);
  }
}

[shader("closesthit")] void ClosestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
  float3 barycentrics =
      float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  if (!payload.hit) {
    payload.color = payload.a * barycentrics + payload.b;
    payload.hit = true;
  }
}

struct SphereIntersectionAttributes {
  float3 normal;
};

[shader("intersection")] void SphereIntersectionMain() {
  float3 origin = WorldRayOrigin();
  float3 dir = WorldRayDirection();
  float3 sphere_center = float3(0, 0, 0);
  float sphere_radius = 1.0;

  origin = mul(WorldToObject3x4(), float4(origin, 1));
  dir = mul(WorldToObject3x4(), float4(dir, 0));

  float3 oc = origin - sphere_center;
  float a = dot(dir, dir);
  float b = 2.0 * dot(oc, dir);
  float c = dot(oc, oc) - sphere_radius * sphere_radius;
  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return;  // No intersection
  }

  float t = (-b - sqrt(discriminant)) / (2.0 * a);
  if (t > RayTMin()) {
    SphereIntersectionAttributes attr;
    attr.normal = (origin + t * dir) - sphere_center;
    attr.normal = normalize(mul(WorldToObject4x3(), attr.normal).xyz);
    ReportHit(t, 0, attr);
  }

  t = (-b + sqrt(discriminant)) / (2.0 * a);
  if (t > RayTMin()) {
    SphereIntersectionAttributes attr;
    attr.normal = (origin + t * dir) - sphere_center;
    attr.normal = normalize(mul(WorldToObject4x3(), attr.normal).xyz);
    ReportHit(t, 0, attr);
  }
}

struct ShadingColor {
  float3 color;
};

[shader("closesthit")] void SphereClosestHitMain(inout RayPayload payload, in SphereIntersectionAttributes attr) {
  if (!payload.hit) {
    payload.color = float3(1.0, 1.0, 1.0) * (max(dot(attr.normal, normalize(float3(-3.0, 1.0, 2.0))), 0.0) * 0.5 + 0.5);
    payload.hit = true;
    ShadingColor shading_color;
    shading_color.color = payload.color;
    CallShader(0, shading_color);
    payload.color = shading_color.color;
  }
}

    [shader("callable")] void CallableMain(inout ShadingColor payload) {
  payload.color = payload.color * float3(0.6, 0.7, 0.8);
}
