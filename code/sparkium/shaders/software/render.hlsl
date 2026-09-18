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
#include "subsurface_random_walk.hlsli"

#ifdef SPARKIUM_SHADER_GRAPHS
#include "material/shader_graph/surface_sampler.hlsli"
#endif
#ifndef SPARKIUM_CPU_SHADER
// The GPU paths assemble the material dispatch at runtime and write it into the
// shader VFS. The CPU backend knows its material set at compile time and
// supplies SoftwareSampleMaterial / SoftwareShadowTransmission itself.
#include "software_materials.hlsli"
#endif

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
  if (!ContinueSubsurfaceRandomWalk(context, record))
    SoftwareSampleMaterial(LoadSoftwareInstance(software_instances, hit.instance).material, context, record);
}
#include "raygen.hlsl"
#include "software/shadow.hlsli"

// The dispatch is per-backend; the per-pixel work is shared. The CPU backend
// calls this once for each pixel it owns.
void RenderDispatch(uint2 id, uint2 extent) {
  if (id.x < extent.x && id.y < extent.y)
    RenderPixel(id, extent);
}

#ifndef SPARKIUM_CPU_SHADER
[numthreads(8, 8, 1)] void Main(uint3 id : SV_DispatchThreadID) {
  uint width, height;
  accumulated_color.GetDimensions(width, height);
  RenderDispatch(id.xy, uint2(width, height));
}
#endif
