#pragma once
#include "grassland/graphics/graphics.h"
#include "long_march.h"

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

class Application {
 public:
  Application(grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_DEFAULT);

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
  std::unique_ptr<grassland::graphics::Program> program_;
  bool alive_{false};
};
