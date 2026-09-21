#include "compute_contract.hlsli"
#include "tone_mapping.hlsli"
// readonly rgba32f accumulated_color image
SP_RESOURCE(SP_TEXTURE(float4), accumulated_color, t0, 0);
#define SP_BINDING_accumulated_color SP_RESOURCE_ACCESS(SP_TEXTURE(float4), accumulated_color, 0)
// readonly r32i accumulated_samples
SP_RESOURCE(SP_TEXTURE(float), accumulated_samples, t0, 1);
#define SP_BINDING_accumulated_samples SP_RESOURCE_ACCESS(SP_TEXTURE(float), accumulated_samples, 1)

SP_RESOURCE(SP_RW_TEXTURE(float4), output, u0, 2);
#define SP_BINDING_output SP_RESOURCE_ACCESS(SP_RW_TEXTURE(float4), output, 2)

// Compute shader convert accumulated color to image
SP_NUMTHREADS(8, 8, 1) void Main(SP_CONTEXT uint3 dispatch_thread_id : SV_DispatchThreadID) {
  // Get the pixel coordinates
  uint2 pixel_coords = dispatch_thread_id.xy;

  // edge check
  uint width, height;
  SP_BINDING_accumulated_color.GetDimensions(width, height);
  if (pixel_coords.x >= width || pixel_coords.y >= height) {
    return;  // Out of bounds
  }

  // Read the accumulated color and samples
  float4 color = SP_BINDING_accumulated_color.Load(int3(pixel_coords, 0));
  int samples = SP_BINDING_accumulated_samples.Load(int3(pixel_coords, 0));

  // If no samples were taken, output black
  if (samples == 0) {
    SP_BINDING_output[pixel_coords] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Compute the average color
  float4 average_color = color / (float)samples;

  // Write the result to the output image
  SP_BINDING_output[pixel_coords] = average_color;
}
