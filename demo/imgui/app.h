#pragma once
#include "long_march.h"

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
  Application(grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_VULKAN);

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
  std::unique_ptr<grassland::graphics::Image> frame_image_;
  bool alive_{false};
};
