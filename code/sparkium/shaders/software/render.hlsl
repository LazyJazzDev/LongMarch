#define SPARKIUM_SOFTWARE_RT
#include "bindings.hlsli"
#include "bsdf/lambertian.hlsli"
#include "bsdf/principled_material.hlsli"
#include "bsdf/specular.hlsli"
#include "camera.hlsl"
#include "direct_lighting.hlsli"
#include "geometry/mesh/hit_record.hlsli"
#include "random.hlsli"
#ifdef SPARKIUM_RAY_QUERY
#include "ray_query/traversal.hlsli"
#else
#include "software/traversal.hlsli"
#endif

#include "software_materials.hlsli"

HitRecord SoftwareHitRecord(SoftwareHit hit, float3 direction) {
  SoftwareInstance instance = LoadSoftwareInstance(software_instances, hit.instance);
  return MakeMeshHitRecord(instance.geometry, hit.instance, hit.primitive, hit.barycentric, hit.distance, direction,
                           instance.object_to_world, transpose(instance.world_to_object));
}
void ApplyPathMiss(inout RenderContext context);
void SoftwareTracePath(RayDesc ray, inout RenderContext context) {
  SoftwareHit hit;
  if (!InlineIntersect(ray, false, hit)) {
    ApplyPathMiss(context);
    return;
  }
  HitRecord record = SoftwareHitRecord(hit, ray.Direction);
  SoftwareSampleMaterial(LoadSoftwareInstance(software_instances, hit.instance).material, context, record);
}
#include "raygen.hlsl"
#include "software/shadow.hlsli"
[numthreads(8, 8, 1)] void Main(uint3 id : SV_DispatchThreadID) {
  uint width, height;
  accumulated_color.GetDimensions(width, height);
  if (id.x < width && id.y < height)
    RenderPixel(id.xy, uint2(width, height));
}
