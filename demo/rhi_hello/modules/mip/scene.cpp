#include "scene.h"

#include <algorithm>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace rhi_mip_demo {
namespace {
#include "shaders/mip.slang.h"

struct Vertex {
  float position[3];
  float uv[2];
};

struct Uniform {
  glm::mat4 model, view, projection;
};
}  // namespace

using namespace grassland::rhi;

bool Scene::Initialize(IDevice *device, ImageFormat target_format) {
  device_ = device;
  const auto &features = device->GetPhysicalDevice().mip_feature;
  if (!device->GetPhysicalDevice().format_features[IMAGE_FORMAT_R8G8B8A8_UNORM].mip_compute_generation ||
      features.max_anisotropy < 2)
    return false;
  anisotropy_ = std::min(16u, features.max_anisotropy);
  std::string source(reinterpret_cast<const char *>(shaders_mip_slang), shaders_mip_slang_len);
  ObjPtr<IShader> vs = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!vs || !ps)
    return false;
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT2, 12}};
  desc.color_formats = {target_format};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX},
                                   {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT},
                                   {2, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  program_ = device->CreateProgram(desc);
  if (!program_)
    return false;
  // Separate vertices give each face the complete, identically oriented texture.
  const Vertex vertices[] = {
      {{-1, -1, 1}, {0, 1}},  {{1, -1, 1}, {1, 1}},   {{1, 1, 1}, {1, 0}},   {{-1, 1, 1}, {0, 0}},
      {{1, -1, -1}, {0, 1}},  {{-1, -1, -1}, {1, 1}}, {{-1, 1, -1}, {1, 0}}, {{1, 1, -1}, {0, 0}},
      {{1, -1, 1}, {0, 1}},   {{1, -1, -1}, {1, 1}},  {{1, 1, -1}, {1, 0}},  {{1, 1, 1}, {0, 0}},
      {{-1, -1, -1}, {0, 1}}, {{-1, -1, 1}, {1, 1}},  {{-1, 1, 1}, {1, 0}},  {{-1, 1, -1}, {0, 0}},
      {{-1, 1, 1}, {0, 1}},   {{1, 1, 1}, {1, 1}},    {{1, 1, -1}, {1, 0}},  {{-1, 1, -1}, {0, 0}},
      {{-1, -1, -1}, {0, 1}}, {{1, -1, -1}, {1, 1}},  {{1, -1, 1}, {1, 0}},  {{-1, -1, 1}, {0, 0}}};
  uint32_t indices[36];
  for (uint32_t face = 0; face < 6; ++face) {
    const uint32_t local[] = {0, 1, 2, 0, 2, 3};
    for (uint32_t j = 0; j < 6; ++j)
      indices[face * 6 + j] = face * 4 + local[j];
  }
  vertices_ = device->CreateBuffer({sizeof(vertices)});
  indices_ = device->CreateBuffer({sizeof(indices)});
  uniforms_ = device->CreateBuffer({9 * 256});
  constexpr uint32_t size = 2048;
  texture_ = device->CreateImage({size, size, IMAGE_FORMAT_R8G8B8A8_UNORM, 0});
  if (!vertices_ || !indices_ || !uniforms_ || !texture_)
    return false;
  for (uint32_t col = 0; col < 3; ++col) {
    SamplerDesc sampling{SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_LINEAR, SAMPLER_FILTER_LINEAR};
    if (col == 0)
      sampling.max_lod = 0;
    if (col == 2)
      sampling.max_anisotropy = anisotropy_;
    samplers_[col] = device->CreateSampler(sampling);
    if (!samplers_[col])
      return false;
    for (uint32_t row = 0; row < 3; ++row) {
      const uint32_t index = row * 3 + col;
      groups_[index] = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
      if (!groups_[index] || !groups_[index]->BindResource(0, BufferView{uniforms_.Get(), index * 256, 256}) ||
          !groups_[index]->BindResource(1, texture_.Get()) || !groups_[index]->BindResource(2, samplers_[col].Get()))
        return false;
    }
  }
  std::vector<uint8_t> pixels(size * size * 4);
  for (uint32_t y = 0; y < size; ++y)
    for (uint32_t x = 0; x < size; ++x) {
      const uint8_t fine = ((x / 2) ^ (y / 2)) & 1 ? 48 : 0;
      const uint8_t vertical = (x / 16) & 1 ? 130 : 30;
      const uint8_t horizontal = (y / 16) & 1 ? 130 : 30;
      const bool grid = (x % 128 < 3 || y % 128 < 3);
      const auto offset = (y * size + x) * 4;
      pixels[offset] = grid ? 235 : vertical + fine;
      pixels[offset + 1] = grid ? 195 : horizontal + fine;
      pixels[offset + 2] = grid ? 70 : uint8_t(60 + ((x ^ y) & 63) + fine);
      pixels[offset + 3] = 255;
    }
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!queue)
    return false;
  ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
  if (!upload || !upload->CmdUploadBuffer(vertices_.Get(), 0, vertices, sizeof(vertices)) ||
      !upload->CmdUploadBuffer(indices_.Get(), 0, indices, sizeof(indices)) ||
      !upload->CmdUploadImage(texture_.Get(), pixels.data(), size * 4) || !upload->CmdGenerateMipmaps(texture_.Get()))
    return false;
  ObjPtr<ISubmissionToken> done = upload->Submit();
  return done && done->WaitHost();
}

bool Scene::Draw(ICommandContext *commands, IImage *target, float seconds) {
  const auto extent = target->GetDesc();
  if (extent.width < 3 || extent.height < 96)
    return false;
  if (!depth_ || depth_->GetDesc().width != extent.width || depth_->GetDesc().height != extent.height) {
    depth_ = device_->CreateImage({extent.width, extent.height, IMAGE_FORMAT_D32_SFLOAT});
    if (!depth_)
      return false;
  }
  const float width = float(extent.width) / 3, height = float(extent.height - 72) / 3;
  std::array<uint8_t, 9 * 256> data{};
  for (uint32_t row = 0; row < 3; ++row) {
    const float scale[] = {0.95f, 0.55f, 0.28f};
    Uniform uniform;
    uniform.model = glm::rotate(glm::mat4(1), seconds * 0.35f + 0.65f, glm::vec3(0, 1, 0));
    uniform.model = glm::rotate(uniform.model, 0.42f, glm::vec3(1, 0, 0));
    uniform.model = glm::scale(uniform.model, glm::vec3(scale[row]));
    uniform.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
    uniform.projection = glm::perspectiveZO(glm::radians(45.0f), width / height, 1.0f, 10.0f);
    for (uint32_t col = 0; col < 3; ++col)
      std::memcpy(data.data() + (row * 3 + col) * 256, &uniform, sizeof(uniform));
  }
  if (!commands->CmdUploadBuffer(uniforms_.Get(), 0, data.data(), data.size()) ||
      !commands->CmdBindProgram(program_.Get()) || !commands->CmdBindGroup(0, groups_[0].Get()))
    return false;
  RenderingInfo rendering;
  rendering.color_attachments = {
      {target, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.035f, 0.045f, 0.07f, 1}}};
  rendering.depth_stencil.image = depth_.Get();
  rendering.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  if (!commands->CmdBeginRendering(rendering) || !commands->CmdBindVertexBuffer(0, vertices_.Get()) ||
      !commands->CmdBindIndexBuffer(indices_.Get()))
    return false;
  for (uint32_t row = 0; row < 3; ++row)
    for (uint32_t col = 0; col < 3; ++col) {
      const uint32_t x = col * extent.width / 3, right = (col + 1) * extent.width / 3;
      const uint32_t y = 72 + (2 - row) * (extent.height - 72) / 3, bottom = 72 + (3 - row) * (extent.height - 72) / 3;
      if (!commands->CmdBindGroup(0, groups_[row * 3 + col].Get()) ||
          !commands->CmdSetViewport({float(x), float(y), float(right - x), float(bottom - y), 0, 1}) ||
          !commands->CmdSetScissor({int32_t(x), int32_t(y), right - x, bottom - y}) || !commands->CmdDrawIndexed(36))
        return false;
    }
  return commands->CmdEndRendering();
}
}  // namespace rhi_mip_demo
