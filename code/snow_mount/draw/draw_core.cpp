#include "snow_mount/draw/draw_core.h"

#include "snow_mount/draw/draw_commands.h"
#include "snow_mount/draw/draw_model.h"
#include "snow_mount/draw/draw_texture.h"

namespace snow_mount::draw {

namespace {
#include "built_in_shaders.inl"
}

Core::Core(graphics::Core *core) : core_(core) {
  if (core_->API() == graphics::BACKEND_API_VULKAN) {
    core_->CreateShader(
        vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/draw.vert"),
                                   VK_SHADER_STAGE_VERTEX_BIT),
        &vertex_shader_);
    core_->CreateShader(
        vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/draw.frag"),
                                   VK_SHADER_STAGE_FRAGMENT_BIT),
        &fragment_shader_);
  }
#if defined(WIN32)
  else if (core_->API() == graphics::BACKEND_API_D3D12) {
    core_->CreateShader(d3d12::CompileShader(GetShaderCode("shaders/draw.hlsl"),
                                             "VSMain", "vs_6_0"),
                        &vertex_shader_);
    core_->CreateShader(d3d12::CompileShader(GetShaderCode("shaders/draw.hlsl"),
                                             "PSMain", "ps_6_0"),
                        &fragment_shader_);
  }
#endif

  core_->CreateBuffer(sizeof(DrawMetadata), graphics::BUFFER_TYPE_DYNAMIC,
                      &metadata_buffer_);

  core_->CreateBuffer(1024, graphics::BUFFER_TYPE_DYNAMIC,
                      &instance_index_buffer_);
  std::vector<uint32_t> instance_indices(1024 / 4);
  for (size_t i = 0; i < instance_indices.size(); i++) {
    instance_indices[i] = i;
  }
  instance_index_buffer_->UploadData(instance_indices.data(), 1024);

  core_->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM,
                     &pure_white_texture_);
  uint32_t white = 0xFFFFFFFF;
  pure_white_texture_->UploadData(&white);
  core_->CreateSampler(graphics::FILTER_MODE_LINEAR, &linear_sampler_);
}

void Core::BeginDraw() {
}

void Core::EndDraw() {
}

void Core::Render(graphics::CommandContext *context, graphics::Image *image) {
  if (draw_metadata_.size() * sizeof(uint32_t) >
      instance_index_buffer_->Size()) {
    instance_index_buffer_->Resize(draw_metadata_.size() * sizeof(uint32_t));
    std::vector<uint32_t> instance_indices(instance_index_buffer_->Size() / 4);
    for (size_t i = 0; i < instance_indices.size(); i++) {
      instance_indices[i] = i;
    }
    instance_index_buffer_->UploadData(
        instance_indices.data(), instance_indices.size() * sizeof(uint32_t));
  }

  if (draw_metadata_.size() * sizeof(DrawMetadata) > metadata_buffer_->Size()) {
    metadata_buffer_->Resize(draw_metadata_.size() * sizeof(DrawMetadata));
  }
  metadata_buffer_->UploadData(draw_metadata_.data(),
                               draw_metadata_.size() * sizeof(DrawMetadata));
  draw_metadata_.clear();

  context->CmdBeginRendering({image}, nullptr);
  context->CmdBindProgram(GetProgram(image->Format()));
  context->CmdBindVertexBuffers(1, {instance_index_buffer_.get()}, {0});
  graphics::Extent2D extent = image->Extent();
  graphics::Viewport viewport;
  graphics::Scissor scissor;
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = extent.width;
  viewport.height = extent.height;
  viewport.min_depth = 0.0f;
  viewport.max_depth = 1.0f;
  scissor.offset = {0, 0};
  scissor.extent = extent;
  context->CmdSetViewport(viewport);
  context->CmdSetScissor(scissor);
  context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  context->CmdBindResources(0, {metadata_buffer_.get()});
  context->CmdBindResources(1, {pure_white_texture_.get()});
  context->CmdBindResources(2, {linear_sampler_.get()});

  for (auto &command : commands_) {
    command->Execute(context);
  }
  context->CmdEndRendering();
  commands_.clear();
}

void Core::CmdSetDrawRegion(int x, int y, int width, int height) {
  commands_.push_back(
      std::make_unique<DrawCmdSetDrawRegion>(x, y, width, height));
}

void Core::CmdDrawInstance(Model *model,
                           Texture *texture,
                           const Transform &transform,
                           glm::vec4 color) {
  CmdDrawInstance(model, texture->Image(), transform, color);
}

void Core::CmdDrawInstance(Model *model,
                           const Transform &transform,
                           glm::vec4 color) {
  CmdDrawInstance(model, pure_white_texture_.get(), transform, color);
}

void Core::CmdDrawInstance(Model *model, glm::vec4 color) {
  CmdDrawInstance(model, pure_white_texture_.get(), Transform{1.0f}, color);
}

void Core::CreateModel(double_ptr<Model> model) {
  model.construct(this);
}

void Core::CreateTexture(int width, int height, double_ptr<Texture> texture) {
  texture.construct(this, width, height);
}

graphics::Program *Core::GetProgram(graphics::ImageFormat format) {
  if (programs_.find(format) == programs_.end()) {
    std::vector<graphics::ImageFormat> color_formats = {format};
    graphics::ImageFormat depth_format = graphics::IMAGE_FORMAT_UNDEFINED;
    core_->CreateProgram(color_formats, depth_format, &programs_[format]);

    auto &program = programs_[format];
    program->BindShader(vertex_shader_.get(), graphics::SHADER_TYPE_VERTEX);
    program->BindShader(fragment_shader_.get(), graphics::SHADER_TYPE_FRAGMENT);
    program->AddInputBinding(sizeof(Vertex));
    program->AddInputBinding(sizeof(uint32_t), true);
    program->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT2,
                               offsetof(Vertex, position));
    program->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT2,
                               offsetof(Vertex, tex_coord));
    program->AddInputAttribute(0, graphics::INPUT_TYPE_FLOAT4,
                               offsetof(Vertex, color));
    program->AddInputAttribute(1, graphics::INPUT_TYPE_UINT4, 0);
    program->SetBlendState(0, true);
    program->SetCullMode(graphics::CULL_MODE_NONE);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_STORAGE_BUFFER, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_TEXTURE, 1);
    program->AddResourceBinding(graphics::RESOURCE_TYPE_SAMPLER, 1);
    program->Finalize();
  }
  return programs_[format].get();
}

void Core::CmdDrawInstance(Model *model,
                           graphics::Image *texture,
                           const Transform &transform,
                           glm::vec4 color) {
  DrawMetadata metadata;
  metadata.transform = transform;
  metadata.color = color;
  uint32_t instance_base = draw_metadata_.size();
  uint32_t instance_count = 1;
  draw_metadata_.push_back(metadata);
  commands_.push_back(std::make_unique<DrawCmdDrawInstance>(
      model, texture, instance_base, instance_count));
}

void CreateCore(graphics::Core *core, double_ptr<Core> draw_core) {
  draw_core.construct(core);
}

}  // namespace snow_mount::draw
