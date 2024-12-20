#version 450

layout(set = 0, binding = 0) uniform texture2D tex;

layout(set = 1, binding = 0) uniform sampler samp;

layout(location = 0) in vec2 in_tex_coord;

layout(location = 0) out vec4 frag_color;

void main() {
  frag_color = texture(sampler2D(tex, samp), in_tex_coord);
}
