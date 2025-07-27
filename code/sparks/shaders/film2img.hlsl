// readonly rgba32f accumulated_color image
Texture2D<float4> accumulated_color : register(t0, space0);
// readonly r32i accumulated_samples
Texture2D<int> accumulated_samples : register(t0, space1);
// output rgba8 image
[[vk::image_format("rgba8")]] RWTexture2D<float4> output : register(u0, space2);

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
  int samples = accumulated_samples.Load(int3(pixel_coords, 0));

  // If no samples were taken, output black
  if (samples == 0) {
    output[pixel_coords] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Compute the average color
  float4 average_color = color / (float)samples;

  // Write the result to the output image
  output[pixel_coords] = pow(average_color, 1.0 / 2.2);  // Apply gamma correction
}
