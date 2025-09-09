#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_tex_coord;
layout(location = 2) in vec4 in_color;
layout(location = 3) in uint in_instance_index;

layout(location = 0) out vec2 out_tex_coord;
layout(location = 1) out vec4 out_color;

struct DrawMetadata {
  mat4 model;
  vec4 color;
};

layout(std430, set = 0, binding = 0) buffer DrawMetadataBuffer {
  DrawMetadata draw_metadata[];
};

void main() {
  out_tex_coord = in_tex_coord;
  out_color = in_color * draw_metadata[in_instance_index].color;
  gl_Position = draw_metadata[in_instance_index].model * vec4(in_position, 0.0, 1.0);
}
