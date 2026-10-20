#pragma once
#include "../../module.h"
#include "long_march.h"

namespace graphics_hello::texture {

struct Vertex {
  glm::vec3 pos;
  glm::vec2 tex_coord;
};

struct GlobalUniformBuffer {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class ModuleTexture final : public Module {
 public:
  ModuleTexture(grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_DEFAULT, bool headless = false);

  ~ModuleTexture() override;

  void OnInit() override;
  void OnClose() override;
  void OnUpdate() override;
  void OnRender() override;

  grassland::graphics::Window *GetWindow() const override {
    return window_.get();
  }

  grassland::graphics::Core *GetCore() const override {
    return core_.get();
  }

  bool IsAlive() const override {
    return alive_;
  }

 private:
  std::shared_ptr<grassland::graphics::Core> core_;
  std::unique_ptr<grassland::graphics::Window> window_;
  std::unique_ptr<grassland::graphics::Buffer> vertex_buffer_;
  std::unique_ptr<grassland::graphics::Buffer> index_buffer_;
  std::unique_ptr<grassland::graphics::Shader> vertex_shader_;
  std::unique_ptr<grassland::graphics::Shader> fragment_shader_;
  std::array<std::unique_ptr<grassland::graphics::Image>, 3> color_images_;
  std::unique_ptr<grassland::graphics::Image> depth_image_;
  std::unique_ptr<grassland::graphics::Image> texture_image_;
  std::unique_ptr<grassland::graphics::Sampler> sampler_;
  std::unique_ptr<grassland::graphics::Program> program_;
  bool alive_{false};
};

}  // namespace graphics_hello::texture
