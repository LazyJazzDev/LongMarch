#pragma once
#include "chang_zheng.h"

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
};

struct GlobalUniformBuffer {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
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
  std::unique_ptr<CD::graphics::Image> frame_image_;
  bool alive_{false};
};
