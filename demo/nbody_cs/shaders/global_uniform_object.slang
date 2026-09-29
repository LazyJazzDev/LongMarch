struct GlobalUniformObject {
  float4x4 world_to_screen;
  float4x4 camera_to_world;
  float particle_size;
  // An int like the C++ uniform: WGSL uniforms cannot hold bool.
  int hdr;
};
