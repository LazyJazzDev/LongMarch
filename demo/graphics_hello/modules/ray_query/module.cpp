#include "module.h"

#include "glm/gtc/matrix_transform.hpp"

namespace graphics_hello::ray_query {

ModuleRayQuery::ModuleRayQuery(grassland::graphics::BackendAPI api) {
  InitializeGraphicsHello(api, core_);
  if (!core_->DeviceRayQuerySupport())
    throw std::runtime_error("Ray queries are unavailable on the selected device/backend");
}

ModuleRayQuery::~ModuleRayQuery() = default;

void ModuleRayQuery::OnInit() {
  alive_ = true;
  core_->CreateWindowObject(1280, 720, GraphicsHelloTitle(core_->API()) + std::string(" Graphics Hello Ray Query"),
                            &window_);

  std::vector<glm::vec3> vertices = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateBuffer(vertices.size() * sizeof(glm::vec3), grassland::graphics::BUFFER_TYPE_DYNAMIC, &vertex_buffer_);
  core_->CreateBuffer(indices.size() * sizeof(uint32_t), grassland::graphics::BUFFER_TYPE_DYNAMIC, &index_buffer_);
  vertex_buffer_->UploadData(vertices.data(), vertices.size() * sizeof(glm::vec3));
  index_buffer_->UploadData(indices.data(), indices.size() * sizeof(uint32_t));

  core_->CreateBuffer(sizeof(CameraObject), grassland::graphics::BUFFER_TYPE_DYNAMIC, &camera_object_buffer_);
  CameraObject camera_object{};
  camera_object.screen_to_camera = glm::inverse(
      glm::perspective(glm::radians(60.0f), (float)window_->GetWidth() / (float)window_->GetHeight(), 0.1f, 10.0f));
  camera_object.camera_to_world =
      glm::inverse(glm::lookAt(glm::vec3{0.0f, 0.0f, 5.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}));
  camera_object_buffer_->UploadData(&camera_object, sizeof(CameraObject));

  core_->CreateImage(window_->GetWidth(), window_->GetHeight(), grassland::graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                     &color_image_);

  core_->CreateShader(LoadShader("modules/ray_query/shaders/shader.hlsl"), "CSMain", "cs_6_5", &compute_shader_);

  core_->CreateBottomLevelAccelerationStructure(vertex_buffer_.get(), index_buffer_.get(), sizeof(glm::vec3),
                                                &triangle_blas_);
  grassland::graphics::RayTracingAABB aabb{-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  std::unique_ptr<grassland::graphics::Buffer> aabb_buffer;
  core_->CreateBuffer(sizeof(aabb), grassland::graphics::BUFFER_TYPE_STATIC, &aabb_buffer);
  aabb_buffer->UploadData(&aabb, sizeof(aabb));
  core_->CreateBottomLevelAccelerationStructure(aabb_buffer->Range(), sizeof(aabb), 1,
                                                grassland::graphics::RAYTRACING_GEOMETRY_FLAG_OPAQUE, &sphere_blas_);
  core_->CreateTopLevelAccelerationStructure(
      {triangle_blas_->MakeInstance(glm::mat4{1.0f}, 0, 0xFF, 0, grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE),
       sphere_blas_->MakeInstance(glm::mat4{1.0f}, 0, 0xFF, 1, grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE)},
      &tlas_);

  core_->CreateComputeProgram(compute_shader_.get(), &program_);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_WRITABLE_IMAGE, 1);
  program_->AddResourceBinding(grassland::graphics::RESOURCE_TYPE_UNIFORM_BUFFER, 1);
  program_->Finalize();
}

void ModuleRayQuery::OnClose() {
  core_->WaitGPU();
  program_.reset();
  compute_shader_.reset();

  tlas_.reset();
  sphere_blas_.reset();
  triangle_blas_.reset();

  color_image_.reset();
  camera_object_buffer_.reset();
  index_buffer_.reset();
  vertex_buffer_.reset();
}

void ModuleRayQuery::OnUpdate() {
  if (window_->ShouldClose()) {
    window_->CloseWindow();
    alive_ = false;
  }
  if (alive_) {
    static float theta = 0.0f;
    theta += glm::radians(0.1f);

    tlas_->UpdateInstances(
        std::vector{triangle_blas_->MakeInstance(glm::translate(glm::mat4{1.0f}, glm::vec3{-2.0f, 0.0f, 0.0f}) *
                                                     glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 1.0f, 0.0f}),
                                                 0, 0xFF, 0, grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE),
                    sphere_blas_->MakeInstance(glm::translate(glm::mat4{1.0f}, glm::vec3{2.0f, 0.0f, 0.0f}) *
                                                   glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 1.0f, 0.0f}) *
                                                   glm::scale(glm::mat4{1.0f}, glm::vec3{1.0f, 1.0f, 0.5f}),
                                               0, 0xFF, 1, grassland::graphics::RAYTRACING_INSTANCE_FLAG_NONE)});
  }
}

void ModuleRayQuery::OnRender() {
  std::unique_ptr<grassland::graphics::CommandContext> command_context;
  core_->CreateCommandContext(&command_context);
  command_context->CmdClearImage(color_image_.get(), {{0.6, 0.7, 0.8, 1.0}});
  command_context->CmdBindComputeProgram(program_.get());
  command_context->CmdBindResources(0, tlas_.get(), grassland::graphics::BIND_POINT_COMPUTE);
  command_context->CmdBindResources(1, {color_image_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
  command_context->CmdBindResources(2, {camera_object_buffer_.get()}, grassland::graphics::BIND_POINT_COMPUTE);
  command_context->CmdDispatch((color_image_->Extent().width + 7) / 8, (color_image_->Extent().height + 7) / 8, 1);
  command_context->CmdPresent(window_.get(), color_image_.get());
  core_->SubmitCommandContext(command_context.get());
}

}  // namespace graphics_hello::ray_query
