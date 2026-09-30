#include "snowberg/gui/world_panel.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace snowberg::gui {
namespace {
#include "built_in_shaders.inl"
}

WorldPanel::WorldPanel(grassland::graphics::Core *core,
                       grassland::graphics::Window *window,
                       int width,
                       int height,
                       const std::string &font_file)
    : core_(core),
      controls_(core, window, font_file),
      width_(width),
      height_(height) {
  if (width <= 0 || height <= 0)
    throw std::invalid_argument("world panel dimensions must be positive");
  controls_.SetLogicalSize(width, height);
  controls_.Style().backdrop_blur = false;
  core_->CreateImage(width, height, grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image_);
  core_->CreateBuffer(4 * sizeof(Vertex), grassland::graphics::BUFFER_TYPE_STATIC, &vertices_);
  core_->CreateBuffer(6 * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_STATIC, &indices_);
  core_->CreateBuffer(sizeof(glm::mat4), grassland::graphics::BUFFER_TYPE_DYNAMIC, &uniform_);
  const Vertex vertices[] = {{{0, 0}, {0, 0}}, {{0, 1}, {0, 1}}, {{1, 0}, {1, 0}}, {{1, 1}, {1, 1}}};
  const uint32_t indices[] = {0, 1, 2, 2, 1, 3};
  vertices_->UploadData(vertices, sizeof(vertices));
  indices_->UploadData(indices, sizeof(indices));
  core_->CreateSampler({grassland::graphics::FILTER_MODE_LINEAR, grassland::graphics::ADDRESS_MODE_CLAMP_TO_EDGE},
                       &sampler_);
  core_->CreateShader(GetShaderCode("shaders/world_panel.slang"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/world_panel.slang"), "PSMain", "ps_6_0", &pixel_shader_);
}

WorldPanel::~WorldPanel() = default;

std::optional<WorldPanel::Hit> WorldPanel::RayHit(glm::vec3 origin, glm::vec3 direction) const {
  const glm::mat4 world_to_local = glm::inverse(local_to_world_);
  const glm::vec3 local_origin = glm::vec3(world_to_local * glm::vec4(origin, 1.0f));
  const glm::vec3 local_direction = glm::vec3(world_to_local * glm::vec4(direction, 0.0f));
  if (std::abs(local_direction.z) < 1e-6f)
    return std::nullopt;
  const float distance = -local_origin.z / local_direction.z;
  if (distance < 0.0f)
    return std::nullopt;
  const glm::vec3 point = local_origin + local_direction * distance;
  if (point.x < 0 || point.x >= 1 || point.y < 0 || point.y >= 1)
    return std::nullopt;
  const glm::vec3 world_point = glm::vec3(local_to_world_ * glm::vec4(point, 1.0f));
  return Hit{{point.x * width_, point.y * height_}, glm::length(world_point - origin)};
}

void WorldPanel::BeginFrame(std::optional<Hit> hit, bool down) {
  controls_.SetPointerInput(hit ? hit->pixel.x : -1.0f, hit ? hit->pixel.y : -1.0f, down);
  controls_.BeginFrame();
}

grassland::graphics::Program *WorldPanel::GetProgram(grassland::graphics::ImageFormat color,
                                                     grassland::graphics::ImageFormat depth) {
  const auto key = std::make_pair(color, depth);
  auto found = programs_.find(key);
  if (found != programs_.end())
    return found->second.get();
  std::unique_ptr<grassland::graphics::Program> program;
  core_->CreateProgram({color}, depth, &program);
  program->AddInputBinding(sizeof(Vertex));
  program->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT2, offsetof(Vertex, position));
  program->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT2, offsetof(Vertex, uv));
  program->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_IMAGE, 1);
  program->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_SAMPLER, 1);
  program->SetBlendState(0, true);
  program->SetCullMode(grassland::graphics::CULL_MODE_NONE);
  program->BindShader(vertex_shader_.get(), grassland::graphics::SHADER_TYPE_VERTEX);
  program->BindShader(pixel_shader_.get(), grassland::graphics::SHADER_TYPE_PIXEL);
  program->Finalize();
  return programs_.emplace(key, std::move(program)).first->second.get();
}

void WorldPanel::Render(grassland::graphics::CommandContext *commands,
                        grassland::graphics::Image *scene,
                        grassland::graphics::Image *depth,
                        const glm::mat4 &view_projection) {
  commands->CmdClearImage(image_.get(), {{0, 0, 0, 0}});
  auto *panel_image = controls_.EndFrame(commands, image_.get());
  const glm::mat4 local_to_clip = view_projection * local_to_world_;
  uniform_->UploadData(&local_to_clip, sizeof(local_to_clip));
  commands->CmdBeginRendering({scene}, depth);
  commands->CmdBindProgram(
      GetProgram(scene->Format(), depth ? depth->Format() : grassland::graphics::IMAGE_FORMAT_UNDEFINED));
  commands->CmdBindVertexBuffers(0, {vertices_.get()}, {0});
  commands->CmdBindIndexBuffer(indices_.get(), 0);
  commands->CmdBindResources(0, {uniform_.get()});
  commands->CmdBindResources(1, {panel_image});
  commands->CmdBindResources(2, {sampler_.get()});
  const auto extent = scene->Extent();
  commands->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1});
  commands->CmdSetScissor({0, 0, extent.width, extent.height});
  commands->CmdSetPrimitiveTopology(grassland::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  commands->CmdDrawIndexed(6, 1, 0, 0, 0);
  commands->CmdEndRendering();
}
}  // namespace snowberg::gui
