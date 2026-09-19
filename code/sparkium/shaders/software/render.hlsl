#include "native_contract.hlsli"
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
#elif defined(SPARKIUM_CPU_SAH)
#include "software/cpu_traversal.hlsli"
#else
#include "software/traversal.hlsli"
#endif
#include "subsurface_random_walk.hlsli"

#ifdef SPARKIUM_SHADER_GRAPHS
#include "material/shader_graph/surface_sampler.hlsli"
#endif
#include "software_materials.hlsli"

HitRecord SoftwareHitRecord(SP_CONTEXT SoftwareHit hit, float3 direction) {
  SoftwareInstance instance = LoadSoftwareInstance(SP_BINDING_software_instances, hit.instance);
  return MakeMeshHitRecord(SP_CONTEXT_ARG instance.geometry, hit.instance, hit.primitive, hit.barycentric, hit.distance,
                           direction, instance.object_to_world, transpose(instance.world_to_object));
}

void ApplyPathMiss(SP_CONTEXT inout RenderContext context);

void SoftwareTracePath(SP_CONTEXT SP_RAY ray, inout RenderContext context) {
  SoftwareHit hit;
  if (!InlineIntersect(SP_CONTEXT_ARG ray, false, hit)) {
    ApplyPathMiss(SP_CONTEXT_ARG context);
    return;
  }

  HitRecord record = SoftwareHitRecord(SP_CONTEXT_ARG hit, ray.Direction);
  if (!ContinueSubsurfaceRandomWalk(SP_CONTEXT_ARG context, record))
    SoftwareSampleMaterial(SP_CONTEXT_ARG LoadSoftwareInstance(SP_BINDING_software_instances, hit.instance).material,
                           context, record);
}

#include "raygen.hlsl"
#include "software/shadow.hlsli"

SP_NUMTHREADS(8, 8, 1) void Main(SP_CONTEXT uint3 id : SV_DispatchThreadID) {
  uint width, height;
  SP_BINDING_accumulated_color.GetDimensions(width, height);
  if (id.x < width && id.y < height)
    RenderPixel(SP_CONTEXT_ARG id.xy, uint2(width, height));
}
