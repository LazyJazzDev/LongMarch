#define SPARKIUM_SOFTWARE_RT
#include "bindings.slang"
#include "bsdf/lambertian.slang"
#include "bsdf/principled_material.slang"
#include "bsdf/specular.slang"
#include "camera.slang"
#include "direct_lighting.slang"
#include "geometry/mesh/hit_record.slang"
#include "random.slang"
#ifdef SPARKIUM_RAY_QUERY
#include "ray_query/traversal.slang"
#else
#include "software/traversal.slang"
#endif
#include "subsurface_random_walk.slang"

#ifdef SPARKIUM_SHADER_GRAPHS
#include "material/shader_graph/surface_sampler.slang"
#endif
#include "software_materials.slang"

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

#include "raygen.slang"
#include "software/shadow.slang"
[numthreads(8, 8, 1)] void Main(uint3 id
                                : SV_DispatchThreadID) {
  uint width, height;
  accumulated_color.GetDimensions(width, height);
  if (id.x < width && id.y < height)
    RenderPixel(id.xy, uint2(width, height));
}
