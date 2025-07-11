#include "app.h"

#include <glm/gtc/matrix_transform.hpp>

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(
      1280, 720,
      ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") + std::string(" ImGui Demo"),
      false, true, &window_);
  window_->InitImGui();
  core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R8G8B8A8_UNORM,
                     &frame_image_);
}

void Application::OnClose() {
  frame_image_.reset();
  window_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }

  if (alive_) {
    window_->BeginImGuiFrame();
    ImGui::ShowDemoWindow();
    window_->EndImGuiFrame();
  }
}

void Application::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(frame_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdPresent(window_.get(), frame_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
