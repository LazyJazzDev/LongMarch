#include "snow_mount/draw/draw_core.h"

#include "snow_mount/draw/draw_commands.h"
#include "snow_mount/draw/draw_font.h"
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
  core_->CreateSampler(
      {graphics::FILTER_MODE_LINEAR,
       grassland::graphics::AddressMode::ADDRESS_MODE_CLAMP_TO_EDGE},
      &linear_sampler_);
  font_core_ = std::make_unique<FontCore>(this);

  CreateModel(&text_model_);
  std::vector<Vertex> vertices = {
      {{-1.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
      {{-1.0f, -1.0f}, {0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
      {{1.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
      {{1.0f, -1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
  };
  std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3};
  text_model_->SetModelData(vertices, indices);
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
  context->CmdSetPrimitiveTopology(graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  context->CmdBindResources(0, {metadata_buffer_.get()});
  context->CmdBindResources(1, {pure_white_texture_.get()});
  context->CmdBindResources(
      2, std::vector<graphics::Sampler *>{linear_sampler_.get()});

  for (auto &command : commands_) {
    command->Execute(context);
  }
  context->CmdEndRendering();
  commands_.clear();
}

void Core::CmdSetDrawRegion(int x, int y, int width, int height) {
  commands_.push_back(
      std::make_unique<DrawCmdSetDrawRegion>(x, y, width, height));
  pixel_transform_ = PixelCoordToNDC(width, height);
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

void Core::CmdDrawText(glm::vec2 origin,
                       const std::string &text,
                       glm::vec4 color) {
  float base_x_ = origin.x;
  float base_y_ = origin.y;
  std::wstring wtext = StringToWString(text);

  for (size_t i = 0; i < wtext.size(); i++) {
    auto char_model = font_core_->GetCharModel(wtext[i]);
    if (char_model.char_tex_) {
      float left = base_x_ + char_model.bearing_x_;
      float top = base_y_ - char_model.bearing_y_;
      float right = char_model.width_ + left;
      float bottom = char_model.height_ + top;
      Transform transform{1.0f};
      transform[0][0] = (right - left) / 2.0f;
      transform[1][1] = (top - bottom) / 2.0f;
      transform[3][0] = left + (right - left) / 2.0f;
      transform[3][1] = top + (bottom - top) / 2.0f;
      CmdDrawInstance(text_model_.get(), char_model.char_tex_,
                      pixel_transform_ * transform, color);
    }
    base_x_ += char_model.advance_x_;
  }
}

void Core::SetFontTypeFile(const std::string &filename) {
  font_core_->SetFontTypeFile(filename);
}

void Core::SetASCIIFontTypeFile(const std::string &filename) {
  font_core_->SetASCIIFontTypeFile(filename);
}

void Core::SetFontSize(uint32_t size) {
  font_core_->SetFontSize(size);
}

float Core::GetTextWidth(const std::string &text) {
  auto wtext = StringToWString(text);
  float width = 0.0f;
  for (size_t i = 0; i < wtext.size(); i++) {
    auto char_model = font_core_->GetCharModel(wtext[i]);
    width += char_model.advance_x_;
  }
  return width;
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
