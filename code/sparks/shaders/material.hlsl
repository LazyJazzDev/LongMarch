#include "bindings.hlsli"
#include "common.hlsli"

[shader("callable")] void CallableMain(inout RenderContext context) {
  ByteAddressBuffer material_buffer = material_data[context.material_buffer_index];
  float3 color = LoadFloat3(material_buffer, 0);
  float3 emission = LoadFloat3(material_buffer, 12);
  context.radiance += emission * context.throughput;
  float3 omega_in;
  float pdf;
  SampleCosHemisphere(context.rd, context.payload.normal, omega_in, pdf);
  float NdotL = dot(context.payload.normal, omega_in);
  if (NdotL > 0) {
    context.throughput *= color * NdotL * INV_PI / pdf;
  } else {
    context.throughput = float3(0.0, 0.0, 0.0);
  }
  context.origin = context.payload.position;
  context.direction = omega_in;
}
