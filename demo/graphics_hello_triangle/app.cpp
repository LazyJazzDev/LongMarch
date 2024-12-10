#include "app.h"

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{},
                                  &core_);
  core_->InitializeLogicalDeviceAutoSelect(false);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}",
                     core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  core_->CreateWindowObject(1280, 720, "Graphics Hello Triangle", &window_);
}

void Application::OnClose() {
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void Application::OnRender() {
}
