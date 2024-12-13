#include "app.h"

namespace {
#include "built_in_shaders.inl"
}

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

  if (core_->API() == grassland::graphics::BACKEND_API_VULKAN) {
    core_->CreateShader(grassland::vulkan::CompileGLSLToSPIRV(
                            GetShaderCode("shaders/vulkan/shader.vert"),
                            VK_SHADER_STAGE_VERTEX_BIT),
                        &vertex_shader_);
    core_->CreateShader(grassland::vulkan::CompileGLSLToSPIRV(
                            GetShaderCode("shaders/vulkan/shader.frag"),
                            VK_SHADER_STAGE_FRAGMENT_BIT),
                        &fragment_shader_);
    grassland::LogInfo("[Vulkan] Shader compiled successfully");
  }
#ifdef WIN32
  else if (core_->API() == grassland::graphics::BACKEND_API_D3D12) {
    core_->CreateShader(
        grassland::d3d12::CompileShader(
            GetShaderCode("shaders/d3d12/shader.hlsl"), "VSMain", "vs_6_0"),
        &vertex_shader_);
    core_->CreateShader(
        grassland::d3d12::CompileShader(
            GetShaderCode("shaders/d3d12/shader.hlsl"), "PSMain", "ps_6_0"),
        &fragment_shader_);
    grassland::LogInfo("[D3D12] Shader compiled successfully");
  }
#endif
}

void Application::OnClose() {
  vertex_buffer_.reset();
  fragment_shader_.reset();
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
