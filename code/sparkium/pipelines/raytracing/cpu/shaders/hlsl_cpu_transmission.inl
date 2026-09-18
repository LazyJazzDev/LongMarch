// Shadow-transmission wrapper for one material namespace.
//
// This is the text SoftwarePipeline::CompileRenderer appends after each
// material source, kept in one place so the two backends cannot drift. It has
// no include guard on purpose: it is included once per material namespace, so
// the enclosing namespace changes each time.
//
// The SAMPLE_SHADOW_* macros are defined by the material source that precedes
// this include and are undefined again right after the namespace closes.
float Transmission(HitRecord hit, float3 direction) {
#ifdef SAMPLE_SHADOW_ANY_HIT
  return 1.0f - saturate(SampleShadowOpacity(hit, direction));
#else
  ShadowRayPayload payload;
  payload.shadow = 1.0f;
#ifdef SAMPLE_SHADOW_NO_HITRECORD
  SampleShadow(payload);
#else
  SampleShadow(payload, hit);
#endif
  return payload.shadow;
#endif
}
