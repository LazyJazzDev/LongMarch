#include "app.h"

#include "glm/gtc/matrix_transform.hpp"

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

  core_->CreateBuffer(sizeof(CameraObject), grassland::graphics::BUFFER_TYPE_DYNAMIC, &camera_object_buffer_);
  CameraObject camera_object{};
  camera_object.screen_to_camera = glm::inverse(
      glm::perspectiveLH(glm::radians(90.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
  camera_object.camera_to_world = glm::inverse(
      glm::lookAtLH(glm::vec3{0.0f, 0.0f, -2.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}));
  camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));

  core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     &color_image_);

  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "RayGenMain", "lib_6_3", &raygen_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "MissMain", "lib_6_3", &miss_shader_);
  core_->CreateShader(GetShaderCode("shaders/shader.hlsl"), "ClosestHitMain", "lib_6_3", &closest_hit_shader_);
  grassland::LogInfo("Shader compiled successfully");

  core_->CreateBottomLevelAccelerationStructure(vertex_buffer_.get(), index_buffer_.get(), sizeof(glm::vec3), &blas_);
  core_->CreateTopLevelAccelerationStructure({{blas_.get(), glm::mat4(1.0f)}}, &tlas_);

  core_->CreateRayTracingProgram(raygen_shader_.get(), miss_shader_.get(), closest_hit_shader_.get(), &program_);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_IMAGE, 1);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->Finalize();
}

void Application::OnClose() {
  program_.reset();
  raygen_shader_.reset();
  miss_shader_.reset();
  closest_hit_shader_.reset();

  blas_.reset();
  tlas_.reset();

  color_image_.reset();
  camera_object_buffer_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void Application::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
  if (alive_) {
    static float theta = 0.0f;
    theta += glm::radians(0.1f);

    tlas_->UpdateInstances(std::vector<std::pair<grassland::graphics::AccelerationStructure *, glm::mat4>>{
        {blas_.get(), glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 1.0f, 0.0f})}});
  }
}

void Application::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdBindRayTracingProgram(program_.get());
  command_context->CmdBindResources(0, tlas_.get(), grassland::graphics::BIND_POINT_RAYTRACING);
  command_context->CmdBindResources(1, {color_image_.get()}, grassland::graphics::BIND_POINT_RAYTRACING);
  command_context->CmdBindResources(2, {camera_object_buffer_.get()}, grassland::graphics::BIND_POINT_RAYTRACING);
  command_context->CmdDispatchRays(window_->GetWidth(), window_->GetHeight(), 1);
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}
