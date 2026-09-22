#include "module.h"

namespace graphics_hello::hdr {

ModuleHDR::ModuleHDR(grassland::graphics::BackendAPI api) {
  InitializeGraphicsHello(api, core_);
}

ModuleHDR::~ModuleHDR() = default;

void ModuleHDR::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(
      1280, 720, GraphicsHelloTitle(core_->API()) + std::string(" Graphics Hello HDR [H: toggle HDR/SDR]"), &window_);
  window_->SetHDR(true);
  window_->KeyEvent().RegisterCallback([this](int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_H && action == GLFW_PRESS) {
      hdr_enabled_ = !hdr_enabled_;
      window_->SetHDR(hdr_enabled_);
      grassland::LogInfo("HDR demo presentation: {}", hdr_enabled_ ? "HDR" : "SDR");
    }
  });

  std::vector<Vertex> vertices = {
      {{-0.5f, 0.05f, 0.0f}, {0.0f, 0.0f, 0.0f}},
      {{0.5f, 0.05f, 0.0f}, {3.0f, 3.0f, 3.0f}},
      {{0.5f, 0.25f, 0.0f}, {3.0f, 3.0f, 3.0f}},
      {{-0.5f, 0.25f, 0.0f}, {0.0f, 0.0f, 0.0f}},
      // SDR reference white below the HDR gradient.
      {{-0.5f, -0.25f, 0.0f}, {1.0f, 1.0f, 1.0f}},
      {{0.5f, -0.25f, 0.0f}, {1.0f, 1.0f, 1.0f}},
      {{0.5f, -0.05f, 0.0f}, {1.0f, 1.0f, 1.0f}},
      {{-0.5f, -0.05f, 0.0f}, {1.0f, 1.0f, 1.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex), grassland::graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);

  core_->CreateShader(LoadShader("modules/hdr/shaders/shader.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(LoadShader("modules/hdr/shaders/shader.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateProgram({grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT},
                       grassland::graphics::IMAGE_FORMAT_UNDEFINED, &program_);
  program_->AddInputBinding(sizeof(Vertex), false);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, 0);
  program_->AddInputAttribute(0, grassland::graphics::INPUT_TYPE_FLOAT3, sizeof(float) * 3);
  program_->BindShader(vertex_shader_.get(), grassland::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), grassland::graphics::SHADER_TYPE_PIXEL);
  program_->Finalize();
}

void ModuleHDR::OnClose() {
  core_->WaitGPU();
  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  color_image_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void ModuleHDR::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void ModuleHDR::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.0, 0.0, 0.0, 1.0}});
  command_context->CmdBeginRendering({color_image_.get()}, nullptr);
  command_context->CmdBindProgram(program_.get());
  command_context->CmdBindVertexBuffers(0, {vertex_buffer_.get()}, {0});
  command_context->CmdBindIndexBuffer(index_buffer_.get(), 0);
  command_context->CmdSetViewport({0, 0, 1280, 720, 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, 1280, 720});
  command_context->CmdSetPrimitiveTopology(grassland::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  command_context->CmdDrawIndexed(12, 1, 0, 0, 0);
  command_context->CmdEndRendering();
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}

}  // namespace graphics_hello::hdr
