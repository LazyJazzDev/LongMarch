#pragma once
#include "long_march.h"

struct CameraObject {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
};

class Application {
 public:
  Application(CD::graphics::BackendAPI api = CD::graphics::BACKEND_API_VULKAN);

  ~Application();

  void OnInit();
  void OnClose();
  void OnUpdate();
  void OnRender();

  bool IsAlive() const {
    return alive_;
  }

 private:
  std::shared_ptr<CD::graphics::Core> core_;
  std::unique_ptr<CD::graphics::Window> window_;
  std::unique_ptr<CD::graphics::Buffer> vertex_buffer_;
  std::unique_ptr<CD::graphics::Buffer> index_buffer_;

  std::unique_ptr<CD::graphics::Buffer> camera_object_buffer_;

  std::unique_ptr<CD::graphics::Shader> raygen_shader_;
  std::unique_ptr<CD::graphics::Shader> miss_shader_;
  std::unique_ptr<CD::graphics::Shader> closest_hit_shader_;

  std::unique_ptr<CD::graphics::AccelerationStructure> blas_;
  std::unique_ptr<CD::graphics::AccelerationStructure> tlas_;

  std::unique_ptr<CD::graphics::Image> color_image_;
  std::unique_ptr<CD::graphics::RayTracingProgram> program_;
  bool alive_{false};
};
