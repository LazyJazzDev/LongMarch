#include "module.h"

namespace graphics_hello::blend {

ModuleBlend::ModuleBlend(grassland::graphics::BackendAPI api, bool headless) : Module(headless) {
  InitializeGraphicsHello(api, core_, false, headless ? 3 : 2);
}

ModuleBlend::~ModuleBlend() = default;

void ModuleBlend::OnInit() {
  alive_ = true;
  if (!headless_)
    core_->CreateWindowObject(1280, 720, GraphicsHelloTitle(core_->API()) + std::string(" Graphics Hello Blending"),
                              &window_);

  std::vector<Vertex> vertices = {
      {{0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 1.0f, 0.5f}},    {{-0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 0.0f, 0.5f}},
      {{0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 1.0f, 0.5f}},   {{0.0f, 0.25f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
      {{-0.3f, -0.35f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}}, {{0.3f, -0.35f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex),
                      headless_ ? grassland::graphics::BUFFER_TYPE_STATIC : grassland::graphics::BUFFER_TYPE_DYNAMIC,
                      &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t),
                      headless_ ? grassland::graphics::BUFFER_TYPE_STATIC : grassland::graphics::BUFFER_TYPE_DYNAMIC,
                      &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  for (int i = 0; i < (headless_ ? 3 : 1); ++i)
    core_->CreateImage(1280, 720,
                       headless_ ? grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM
                                 : grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                       &color_images_[i]);

  core_->CreateShader(LoadShader("modules/blend/shaders/shader.slang"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(LoadShader("modules/blend/shaders/shader.slang"), "PSMain", "ps_6_0", &fragment_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateProgram({color_images_[0]->Format()}, grassland::graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->AddInputBinding(sizeof(Vertex), false);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT4, sizeof(float) * 3);
  program_->BindShader(vertex_shader_.get(), grassland::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), grassland::graphics::SHADER_TYPE_PIXEL);
  program_->SetBlendState(0, true);
  if (headless_)
    program_->SetCullMode(grassland::graphics::CULL_MODE_NONE);
  program_->Finalize();
}

void ModuleBlend::OnClose() {
  core_->WaitGPU();
  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  for (auto &image : color_images_)
    image.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void ModuleBlend::OnUpdate() {
  if (window_ && window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void ModuleBlend::OnRender() {
  auto *color_image = color_images_[headless_ ? core_->CurrentFrame() : 0].get();
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image, {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdBeginRendering({color_image}, nullptr);
  command_context->CmdBindProgram(program_.get());
  command_context->CmdBindVertexBuffers(0, {vertex_buffer_.get()}, {0});
  command_context->CmdBindIndexBuffer(index_buffer_.get(), 0);
  command_context->CmdSetViewport({0, 0, 1280, 720, 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, 1280, 720});
  command_context->CmdSetPrimitiveTopology(grassland::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  command_context->CmdDrawIndexed(3, 1, 0, 3, 0);
  command_context->CmdDrawIndexed(3, 1, 0, 0, 0);
  command_context->CmdEndRendering();
  if (!headless_)
    command_context->CmdPresent(window_.get(), color_image);
  core_->SubmitCommandContext(command_context.get());
}

}  // namespace graphics_hello::blend
