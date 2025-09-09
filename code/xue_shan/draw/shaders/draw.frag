#version 450

layout(location = 0) in vec2 in_tex_coord;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 frag_color;

layout(set = 1, binding = 0) uniform texture2D tex;
layout(set = 2, binding = 0) uniform sampler samp;

void main() {
  vec4 tex_color = texture(sampler2D(tex, samp), in_tex_coord);
  frag_color = tex_color * in_color;
}
