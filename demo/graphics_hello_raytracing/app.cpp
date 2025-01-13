#include "app.h"

namespace {
#include "built_in_shaders.inl"
}

Application::Application(grassland::graphics::BackendAPI api) {
  grassland::graphics::CreateCore(api, grassland::graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);

  grassland::LogInfo("Device Name: {}", core_->DeviceName());
  grassland::LogInfo("- Ray Tracing Support: {}", core_->DeviceRayTracingSupport());
}

Application::~Application() {
  core_.reset();
}

void Application::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(1280, 720,
                            ((core_->API() == grassland::graphics::BACKEND_API_VULKAN) ? "[Vulkan]" : "[D3D12]") +
                                std::string(" Graphics Hello Ray Tracing"),
                            &window_);

  std::vector<glm::vec3> vertices = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateBuffer(vertices.size() * sizeof(glm::vec3), grassland::graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(glm::vec3));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateImage(1280, 720, grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &color_image_);

  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "RayGenMain", "lib_6_3", &raygen_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "MissMain", "lib_6_3", &miss_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "ClosestHitMain", "lib_6_3", &closest_hit_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateBottomLevelAccelerationStructure(vertex_buffer_.get(), index_buffer_.get(), sizeof(glm::vec3), &blas_);
  core_->CreateTopLevelAccelerationStructure({{blas_.get(), glm::mat4(1.0f)}}, &tlas_);
}

void Application::OnClose() {
  program_.reset();
  raygen_shader_.reset();
  miss_shader_.reset();
  closest_hit_shader_.reset();

  blas_.reset();
  tlas_.reset();

  color_image_.reset();
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
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
