#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "long_march.h"

using namespace long_march;

struct CameraObject {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
};

class RayTracingApp {
 public:
  RayTracingApp();
  ~RayTracingApp();
  void Run();

 private:
  void OnInit();
  void OnUpdate();
  void OnRender();
  void OnClose();

  void CreateCoreObjects();
  void CreateFrameAssets();
  void CreateObjectAssets();
  void CreateRayTracingPipeline();

  void DestroyCoreObjects();
  void DestroyFrameAssets();
  void DestroyObjectAssets();
  void DestroyRayTracingPipeline();

  GLFWwindow *window_;

  std::unique_ptr<vulkan::Core> core_;
  std::unique_ptr<vulkan::Image> frame_image_;

  std::unique_ptr<vulkan::StaticBuffer<glm::vec3>> vertex_buffer_;
  std::unique_ptr<vulkan::StaticBuffer<uint32_t>> index_buffer_;

  std::unique_ptr<vulkan::AccelerationStructure> blas_;
  std::unique_ptr<vulkan::AccelerationStructure> tlas_;

  std::unique_ptr<vulkan::DescriptorSetLayout> descriptor_set_layout_;
  std::unique_ptr<vulkan::StaticBuffer<CameraObject>> camera_object_buffer_;

  std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
  std::unique_ptr<vulkan::DescriptorSet> descriptor_set_;

  std::unique_ptr<vulkan::ShaderModule> raygen_shader_;
  std::unique_ptr<vulkan::ShaderModule> miss_shader_;
  std::unique_ptr<vulkan::ShaderModule> hit_shader_;
  std::unique_ptr<vulkan::PipelineLayout> pipeline_layout_;
  std::unique_ptr<vulkan::Pipeline> pipeline_;
  std::unique_ptr<vulkan::ShaderBindingTable> sbt_;
};
