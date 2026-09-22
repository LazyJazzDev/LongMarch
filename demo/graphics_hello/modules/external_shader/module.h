#pragma once
#include "../../module.h"

namespace graphics_hello::external_shader {

struct CameraObject {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
};

class ModuleExternalShader final : public Module {
 public:
  ModuleExternalShader(grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_DEFAULT);

  ~ModuleExternalShader() override;

  void OnInit() override;
  void OnClose() override;
  void OnUpdate() override;
  void OnRender() override;

  grassland::graphics::Window *GetWindow() const override {
    return window_.get();
  }

  bool IsAlive() const override {
    return alive_;
  }

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Window> window_;

  std::unique_ptr<grassland::graphics::Buffer> camera_object_buffer_;

  std::unique_ptr<grassland::graphics::Shader> raygen_shader_;
  std::unique_ptr<grassland::graphics::Shader> miss_shader_;
  std::unique_ptr<grassland::graphics::Shader> closest_hit_shader_;

  std::unique_ptr<grassland::graphics::Shader> sphere_intersection_shader_;
  std::unique_ptr<grassland::graphics::Shader> sphere_closest_hit_shader_;

  std::unique_ptr<grassland::graphics::Shader> callable_shader_;

  std::unique_ptr<grassland::graphics::AccelerationStructure> triangle_blas_;
  std::unique_ptr<grassland::graphics::AccelerationStructure> sphere_blas_;
  std::unique_ptr<grassland::graphics::AccelerationStructure> tlas_;

  std::unique_ptr<grassland::graphics::Image> color_image_;
  std::unique_ptr<grassland::graphics::RayTracingProgram> program_;
  bool alive_{false};
};

}  // namespace graphics_hello::external_shader
