#include "app.h"

namespace {
#include "built_in_shaders.inl"
}

Application::Application(CD::graphics::BackendAPI api) {
  CD::graphics::CreateCore(api, CD::graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  CD::LogInfo("Device Name: {}", core_->DeviceName());
  CD::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(1280, 720,
                            ((core_->API() == CD::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") +
                                std::string(" Graphics Hello SDR Sample"),
                            &window_);
  // window_->SetHDR(true);

  core_->CreateImage(1280, 720, CD::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);

  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "VSMain", "vs_6_0", &vertex_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "PSMain", "ps_6_0", &fragment_shader_);
  CD::LogInfo("Shader compiled successfully");

  core_->CreateProgram({CD::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT}, CD::graphics::IMAGE_FORMAT_UNDEFINED,
                       &program_);
  program_->BindShader(vertex_shader_.get(), CD::graphics::SHADER_TYPE_VERTEX);
  program_->BindShader(fragment_shader_.get(), CD::graphics::SHADER_TYPE_FRAGMENT);
  program_->Finalize();
}

void Application::OnClose() {
  program_.reset();
  vertex_shader_.reset();
  fragment_shader_.reset();
  color_image_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void Application::OnRender() {
  std::unique_ptr<CD::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdBeginRendering({color_image_.get()}, nullptr);
  command_context->CmdBindProgram(program_.get());
  command_context->CmdSetViewport({0, 0, 1280, 720, 0.0f, 1.0f});
  command_context->CmdSetScissor({0, 0, 1280, 720});
  command_context->CmdSetPrimitiveTopology(CD::graphics::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  command_context->CmdDraw(6, 1, 0, 0);
  command_context->CmdEndRendering();
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
