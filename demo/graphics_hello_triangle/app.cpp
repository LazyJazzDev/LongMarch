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

  std::vector<Vertex> vertices = {
      {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };
  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateBuffer(vertices.size() * sizeof(Vertex),
                      grassland::graphics::BUFFER_TYPE_STATIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t),
                      grassland::graphics::BUFFER_TYPE_STATIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(Vertex));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));
}

void Application::OnClose() {
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
}

void Application::OnRender() {
}
