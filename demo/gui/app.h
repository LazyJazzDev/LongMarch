#pragma once
#include "long_march.h"
#include "snowberg/gui/gui.h"
#include "snowberg/gui/world_panel.h"

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
  std::unique_ptr<grassland::graphics::Image> frame_image_;
  std::unique_ptr<snowberg::gui::Context> ui_;
  std::unique_ptr<snowberg::gui::WorldPanel> world_panel_;
  std::unique_ptr<grassland::graphics::Image> depth_image_;
  glm::mat4 view_projection_{1.0f};
  bool animated_{true};
  bool particles_{true};
  float intensity_{0.5f};
  bool alive_{false};
};
