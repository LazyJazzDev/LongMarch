#include "scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace rhi_pcf_demo {
namespace {
#include "shaders/pcf_shader.slang.h"

constexpr uint32_t kShadowSize = 512;
const glm::vec3 kLight(0, 6, -7);

struct Vertex {
  glm::vec3 position, normal;
  glm::vec4 color;
};

struct Uniform {
  glm::mat4 view_projection, light_projection;
  glm::vec4 light, options;
};

void AddBox(std::vector<Vertex> &vertices, glm::vec3 center, glm::vec3 size, glm::vec4 color) {
  const glm::vec3 normals[] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  const glm::vec3 tangents[] = {{0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}};
  const glm::vec2 corners[] = {{-1, -1}, {1, -1}, {1, 1}, {-1, -1}, {1, 1}, {-1, 1}};
  for (int face = 0; face < 6; ++face) {
    const auto normal = normals[face], tangent = tangents[face], bitangent = glm::cross(normal, tangent);
    for (auto corner : corners)
      vertices.push_back({center + (normal + tangent * corner.x + bitangent * corner.y) * size * .5f, normal, color});
  }
}
}  // namespace

using namespace grassland::rhi;

bool Scene::Initialize(IDevice *device, ImageFormat target_format) {
  device_ = device;
  const std::string source(reinterpret_cast<const char *>(shaders_pcf_shader_slang), shaders_pcf_shader_slang_len);
  const std::string shadow_source = "#define SHADOW_PASS 1\n" + source;
  ObjPtr<IShader> shadow_vs = device->CreateShaderFromSource(shadow_source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> shadow_ps = device->CreateShaderFromSource(shadow_source, "PSMain", SHADER_STAGE_FRAGMENT);
  ObjPtr<IShader> scene_vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> scene_ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!shadow_vs || !shadow_ps || !scene_vs || !scene_ps)
    return false;
  ProgramDesc desc;
  desc.vertex_shader = shadow_vs.Get();
  desc.fragment_shader = shadow_ps.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}};
  desc.color_formats.clear();
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = desc.depth_stencil.depth_write = true;
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_ALL}}}};
  shadow_program_ = device->CreateProgram(desc);
  shadow_group_ = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  desc.vertex_shader = scene_vs.Get();
  desc.fragment_shader = scene_ps.Get();
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT3, 12}, {2, 0, INPUT_TYPE_FLOAT4, 24}};
  desc.color_formats = {target_format};
  desc.pipeline_layout.groups[0].bindings.insert(desc.pipeline_layout.groups[0].bindings.end(),
                                                 {{1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                                  {2, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT},
                                                  {3, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}});
  scene_program_ = device->CreateProgram(desc);
  shadow_ = device->CreateImage({kShadowSize, kShadowSize, IMAGE_FORMAT_D32_SFLOAT});
  uniforms_ = device->CreateBuffer({256});
  SamplerDesc sampler{SAMPLER_FILTER_NEAREST, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_CLAMP_TO_EDGE};
  hard_sampler_ = device->CreateSampler(sampler);
  sampler.min_filter = sampler.mag_filter = SAMPLER_FILTER_LINEAR;
  sampler.compare_enable = true;
  sampler.compare_op = COMPARE_LESS_EQUAL;
  pcf_sampler_ = device->CreateSampler(sampler);
  if (!shadow_program_ || !scene_program_ || !shadow_group_ || !shadow_ || !uniforms_ || !hard_sampler_ ||
      !pcf_sampler_ || !shadow_group_->BindResource(0, BufferView{uniforms_.Get(), 0, 256}))
    return false;
  scene_group_ = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!scene_group_ || !scene_group_->BindResource(0, BufferView{uniforms_.Get(), 0, 256}) ||
      !scene_group_->BindResource(1, shadow_.Get()) || !scene_group_->BindResource(2, hard_sampler_.Get()) ||
      !scene_group_->BindResource(3, pcf_sampler_.Get()))
    return false;
  std::vector<Vertex> vertices;
  AddBox(vertices, {0, -.15f, 0}, {14, .3f, 18}, {.38f, .43f, .48f, 0});
  AddBox(vertices, {-1.4f, 1.5f, 0}, {.75f, 3, 1.1f}, {.72f, .35f, .16f, 0});
  AddBox(vertices, {1.4f, 1.5f, 0}, {.75f, 3, 1.1f}, {.72f, .35f, .16f, 0});
  shadow_vertices_ = uint32_t(vertices.size());
  AddBox(vertices, kLight, {.3f, .3f, .3f}, {1, 1, 1, 1});
  scene_vertices_ = uint32_t(vertices.size());
  vertices_ = device->CreateBuffer({vertices.size() * sizeof(Vertex)});
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!vertices_ || !queue)
    return false;
  ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
  if (!upload || !upload->CmdUploadBuffer(vertices_.Get(), 0, vertices.data(), vertices.size() * sizeof(Vertex)))
    return false;
  ObjPtr<ISubmissionToken> done = upload->Submit();
  return done && done->WaitHost();
}

bool Scene::Draw(ICommandContext *commands, IImage *target, bool visibility_only) {
  const auto extent = target->GetDesc();
  if (extent.width < 2 || extent.height < 96)
    return false;
  if (!depth_ || depth_->GetDesc().width != extent.width || depth_->GetDesc().height != extent.height) {
    depth_ = device_->CreateImage({extent.width, extent.height, IMAGE_FORMAT_D32_SFLOAT});
    if (!depth_)
      return false;
  }
  Uniform uniform;
  uniform.view_projection =
      glm::perspectiveZO(glm::radians(52.f), float(extent.width) / (extent.height - 72), .1f, 60.f) *
      glm::lookAt(glm::vec3(0, 7, 13), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0));
  uniform.light_projection = glm::perspectiveZO(glm::radians(80.f), 1.f, 1.f, 30.f) *
                             glm::lookAt(kLight, glm::vec3(0, 0, 1), glm::vec3(0, 1, 0));
  uniform.light = glm::vec4(kLight, 1);
  uniform.options = {float(extent.width) * .5f, float(kShadowSize), .0006f, visibility_only ? 1.f : 0.f};
  if (!commands->CmdUploadBuffer(uniforms_.Get(), 0, &uniform, sizeof(uniform)) ||
      !commands->CmdBindProgram(shadow_program_.Get()) || !commands->CmdBindGroup(0, shadow_group_.Get()) ||
      !commands->CmdBindVertexBuffer(0, vertices_.Get()))
    return false;
  RenderingInfo shadow_pass;
  shadow_pass.depth_stencil.image = shadow_.Get();
  shadow_pass.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  if (!commands->CmdBeginRendering(shadow_pass) || !commands->CmdSetViewport({0, 0, kShadowSize, kShadowSize, 0, 1}) ||
      !commands->CmdSetScissor({0, 0, kShadowSize, kShadowSize}) || !commands->CmdDraw(shadow_vertices_) ||
      !commands->CmdEndRendering() || !commands->CmdBindProgram(scene_program_.Get()) ||
      !commands->CmdBindGroup(0, scene_group_.Get()) || !commands->CmdBindVertexBuffer(0, vertices_.Get()))
    return false;
  RenderingInfo scene_pass;
  scene_pass.color_attachments = {
      {target, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {.035f, .045f, .065f, 1}}};
  scene_pass.depth_stencil.image = depth_.Get();
  scene_pass.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  if (!commands->CmdBeginRendering(scene_pass))
    return false;
  if (!commands->CmdSetViewport({0, 72, float(extent.width), float(extent.height - 72), 0, 1}) ||
      !commands->CmdSetScissor({0, 72, extent.width, extent.height - 72}) || !commands->CmdDraw(scene_vertices_))
    return false;
  return commands->CmdEndRendering();
}
}  // namespace rhi_pcf_demo
