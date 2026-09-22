struct CameraInfo {
  float4x4 screen_to_camera;
  float4x4 camera_to_world;
};

RaytracingAccelerationStructure scene : register(t0, space0);
RWTexture2D<float4> output : register(u0, space1);
ConstantBuffer<CameraInfo> camera_info : register(b0, space2);

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
  ray.Direction = normalize(mul(camera_info.camera_to_world, float4(target.xyz, 0)).xyz);
  ray.TMin = 0.001;
  ray.TMax = 10000.0;

  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
  while (query.Proceed()) {
  }
  float3 color = float3(0.8, 0.7, 0.6);
  if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    float2 bary = query.CommittedTriangleBarycentrics();
    color = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
  }
  output[pixel.xy] = float4(color, 1);
}
