#include "tone_mapping_functions.slang"
// readonly rgba32f accumulated_color image
RHI_TEXTURE2D(0, 0, float4, accumulated_color);
// readonly r32f accumulated_samples
RHI_TEXTURE2D(0, 1, float, accumulated_samples);

RHI_RW_TEXTURE2D(0, 2, float4, output);

// Compute shader convert accumulated color to image
[numthreads(8, 8, 1)] void Main(uint3 dispatch_thread_id
                                : SV_DispatchThreadID) {
  // Get the pixel coordinates
  uint2 pixel_coords = dispatch_thread_id.xy;

  // edge check
  uint width, height;
  accumulated_color.GetDimensions(width, height);
  if (pixel_coords.x >= width || pixel_coords.y >= height) {
    return;  // Out of bounds
  }

  // Read the accumulated color and samples
  float4 color = accumulated_color.Load(int3(pixel_coords, 0));
  float samples = accumulated_samples.Load(int3(pixel_coords, 0));

  // If no samples were taken, output black
  if (samples == 0) {
    output[pixel_coords] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Compute the average color
  float4 average_color = color / samples;

  // Write the result to the output image
  output[pixel_coords] = average_color;
}
