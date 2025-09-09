#pragma once
#include "cao_di/graphics/graphics.h"
#include "chang_zheng.h"

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
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
  std::unique_ptr<CD::graphics::Shader> vertex_shader_;
  std::unique_ptr<CD::graphics::Shader> fragment_shader_;
  std::unique_ptr<CD::graphics::Image> color_image_;
  std::unique_ptr<CD::graphics::Program> program_;
  bool alive_{false};
  bool first_frame_{true};
};
