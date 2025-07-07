#include "rtapp.h"

#include "vector"

namespace {
#include "built_in_shaders.inl"
}

RayTracingApp::RayTracingApp() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(1920, 1080, "Ray Tracing", nullptr, nullptr);

  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("Failed to create window");
  }
}

RayTracingApp::~RayTracingApp() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void RayTracingApp::Run() {
  OnInit();

  while (!glfwWindowShouldClose(window_)) {
    OnUpdate();
    OnRender();
    glfwPollEvents();
  }

  if (core_) {
    core_->Device()->WaitIdle();
  }

  OnClose();
}

void RayTracingApp::OnInit() {
  CreateCoreObjects();
  CreateFrameAssets();
  CreateObjectAssets();
  CreateRayTracingPipeline();
}

void RayTracingApp::OnClose() {
  DestroyRayTracingPipeline();
  DestroyObjectAssets();
  DestroyFrameAssets();
  DestroyCoreObjects();
}

void RayTracingApp::OnUpdate() {
  static float theta = 0.0f;
  theta += glm::radians(0.1f);

  tlas_->UpdateInstances(
      std::vector<std::pair<grassland::vulkan::AccelerationStructure *, glm::mat4>>{
          {blas_.get(), glm::rotate(glm::mat4{1.0f}, theta, glm::vec3{0.0f, 1.0f, 0.0f})}},
      core_->GraphicsCommandPool(), core_->GraphicsQueue());
}

void RayTracingApp::OnRender() {
  core_->BeginFrame();
  frame_image_->ClearColor(core_->CommandBuffer()->Handle(), {0.6f, 0.7f, 0.8f, 1.0f});
  VkCommandBuffer command_buffer = core_->CommandBuffer()->Handle();
  vulkan::TransitImageLayout(command_buffer, frame_image_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_->Handle());
  VkDescriptorSet descriptor_sets[] = {descriptor_set_->Handle()};
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout_->Handle(), 0, 1,
                          descriptor_sets, 0, nullptr);

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties =
      core_->Device()->PhysicalDevice().GetPhysicalDeviceRayTracingPipelineProperties();

  auto aligned_size = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1) & ~(alignment - 1); };
  const uint32_t handle_size_aligned = aligned_size(ray_tracing_pipeline_properties.shaderGroupHandleSize,
                                                    ray_tracing_pipeline_properties.shaderGroupHandleAlignment);

  VkStridedDeviceAddressRegionKHR ray_gen_shader_sbt_entry{};
  ray_gen_shader_sbt_entry.deviceAddress = sbt_->GetRayGenDeviceAddress();
  ray_gen_shader_sbt_entry.stride = handle_size_aligned;
  ray_gen_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR miss_shader_sbt_entry{};
  miss_shader_sbt_entry.deviceAddress = sbt_->GetMissDeviceAddress();
  miss_shader_sbt_entry.stride = handle_size_aligned;
  miss_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR hit_shader_sbt_entry{};
  hit_shader_sbt_entry.deviceAddress = sbt_->GetHitGroupDeviceAddress();
  hit_shader_sbt_entry.stride = handle_size_aligned;
  hit_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR callable_shader_sbt_entry{};

  core_->Device()->Procedures().vkCmdTraceRaysKHR(command_buffer, &ray_gen_shader_sbt_entry, &miss_shader_sbt_entry,
                                                  &hit_shader_sbt_entry, &callable_shader_sbt_entry,
                                                  frame_image_->Width(), frame_image_->Height(), 1);

  vulkan::TransitImageLayout(command_buffer, frame_image_->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_NONE, VK_ACCESS_TRANSFER_READ_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);

  core_->OutputFrame(frame_image_.get());
  core_->EndFrame();
}

void RayTracingApp::CreateCoreObjects() {
  vulkan::CoreSettings core_settings;
  core_settings.window = window_;
  core_settings.enable_ray_tracing = true;
  core_settings.device_index = -1;
  core_ = std::make_unique<vulkan::Core>(core_settings);
}

void RayTracingApp::DestroyCoreObjects() {
  core_.reset();
}

void RayTracingApp::CreateFrameAssets() {
  core_->Device()->CreateImage(VK_FORMAT_B8G8R8A8_UNORM, core_->Swapchain()->Extent(), &frame_image_);
}

void RayTracingApp::DestroyFrameAssets() {
  frame_image_.reset();
}

void RayTracingApp::CreateObjectAssets() {
  //  std::vector<glm::vec3> vertices = {
  //      {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f},
  //      {-0.5f, 0.5f, 0.5f},   {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f},
  //      {0.5f, 0.5f, -0.5f},   {0.5f, 0.5f, 0.5f},
  //  };
  //  std::vector<uint32_t> indices = {
  //      0, 1, 2, 2, 1, 3, 4, 6, 5, 5, 6, 7, 0, 2, 4, 4, 2, 6,
  //      1, 5, 3, 3, 5, 7, 0, 4, 1, 1, 4, 5, 2, 3, 6, 6, 3, 7,
  //  };

  std::vector<glm::vec3> vertices = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
  std::vector<uint32_t> indices = {0, 1, 2};

  core_->CreateStaticBuffer<glm::vec3>(
      vertices.size(),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      &vertex_buffer_);
  core_->CreateStaticBuffer<uint32_t>(
      indices.size(),
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      &index_buffer_);
  core_->CreateBottomLevelAccelerationStructure(vertices, indices, &blas_);

  core_->CreateTopLevelAccelerationStructure({{blas_.get(), glm::mat4{1.0f}}}, &tlas_);

  core_->Device()->CreateDescriptorSetLayout(
      {{0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
       {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
       {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr}},
      &descriptor_set_layout_);

  vulkan::DescriptorPoolSize pool_size = descriptor_set_layout_->GetPoolSize();
  core_->Device()->CreateDescriptorPool(pool_size, 1, &descriptor_pool_);
  descriptor_pool_->AllocateDescriptorSet(descriptor_set_layout_->Handle(), &descriptor_set_);

  core_->CreateStaticBuffer<CameraObject>(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &camera_object_buffer_);
  CameraObject camera_object{};
  camera_object.screen_to_camera = glm::inverse(glm::perspectiveLH(
      glm::radians(90.0f), (float)core_->Swapchain()->Extent().width / (float)core_->Swapchain()->Extent().height, 0.1f,
      10.0f));
  camera_object.camera_to_world = glm::inverse(
      glm::lookAtLH(glm::vec3{0.0f, 0.0f, -2.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f}));
  camera_object_buffer_->UploadContents(&camera_object, 1);

  {
    std::vector<VkWriteDescriptorSet> writes;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_->Handle();
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    write.pNext = nullptr;
    VkAccelerationStructureKHR as = tlas_->Handle();
    VkWriteDescriptorSetAccelerationStructureKHR as_info{};
    as_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_info.pNext = nullptr;
    as_info.accelerationStructureCount = 1;
    as_info.pAccelerationStructures = &as;
    write.pNext = &as_info;
    writes.push_back(write);

    VkDescriptorImageInfo image_info{};
    image_info.sampler = VK_NULL_HANDLE;
    image_info.imageView = frame_image_->ImageView();
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    write.dstBinding = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &image_info;
    writes.push_back(write);

    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = camera_object_buffer_->GetBuffer()->Handle();
    buffer_info.offset = 0;
    buffer_info.range = sizeof(CameraObject);
    write.dstBinding = 2;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &buffer_info;
    writes.push_back(write);

    vkUpdateDescriptorSets(core_->Device()->Handle(), writes.size(), writes.data(), 0, nullptr);
  }
}

void RayTracingApp::DestroyObjectAssets() {
  camera_object_buffer_.reset();
  descriptor_set_.reset();
  descriptor_pool_.reset();
  descriptor_set_layout_.reset();
  tlas_.reset();
  blas_.reset();
  vertex_buffer_.reset();
  index_buffer_.reset();
}

void RayTracingApp::CreateRayTracingPipeline() {
  core_->Device()->CreatePipelineLayout({descriptor_set_layout_->Handle()}, &pipeline_layout_);
  core_->Device()->CreateShaderModule(
      vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/raytracing.rgen"), VK_SHADER_STAGE_RAYGEN_BIT_KHR),
      &raygen_shader_);
  core_->Device()->CreateShaderModule(
      vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/raytracing.rmiss"), VK_SHADER_STAGE_MISS_BIT_KHR),
      &miss_shader_);
  core_->Device()->CreateShaderModule(
      vulkan::CompileGLSLToSPIRV(GetShaderCode("shaders/raytracing.rchit"), VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
      &hit_shader_);

  core_->Device()->CreateRayTracingPipeline(pipeline_layout_.get(), raygen_shader_.get(), miss_shader_.get(),
                                            hit_shader_.get(), &pipeline_);

  core_->Device()->CreateShaderBindingTable(pipeline_.get(), &sbt_);
}

void RayTracingApp::DestroyRayTracingPipeline() {
  sbt_.reset();
  pipeline_.reset();
  hit_shader_.reset();
  miss_shader_.reset();
  raygen_shader_.reset();
  pipeline_layout_.reset();
}
