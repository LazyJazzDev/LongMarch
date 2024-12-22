#pragma once
#include "long_march.h"

struct Vertex {
  glm::vec3 pos;
  glm::vec2 tex_coord;
};

struct GlobalUniformBuffer {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class Application {
 public:
  Application(grassland::graphics::BackendAPI api =
                  grassland::graphics::BACKEND_API_VULKAN);

  ~Application();

  void OnInit();
  void OnClose();
  void OnUpdate();
  void OnRender();

  bool IsAlive() const {
    return alive_;
  }

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Window> window_;
  std::unique_ptr<grassland::graphics::Buffer> vertex_buffer_;
  std::unique_ptr<grassland::graphics::Buffer> index_buffer_;
  std::unique_ptr<grassland::graphics::Shader> vertex_shader_;
  std::unique_ptr<grassland::graphics::Shader> fragment_shader_;
  std::unique_ptr<grassland::graphics::Image> color_image_;
  std::unique_ptr<grassland::graphics::Image> depth_image_;
  std::unique_ptr<grassland::graphics::Image> texture_image_;
  std::unique_ptr<grassland::graphics::Sampler> sampler_;
  std::unique_ptr<grassland::graphics::Program> program_;
  bool alive_{false};
};