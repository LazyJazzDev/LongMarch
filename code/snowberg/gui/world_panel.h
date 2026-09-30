#pragma once

#include <map>
#include <optional>

#include "snowberg/gui/gui.h"

namespace snowberg::gui {

// A GUI surface rendered on a world-space quad. RayHit returns local UI
// coordinates; the caller chooses the nearest hit across all world panels.
class WorldPanel {
 public:
  struct Hit {
    glm::vec2 pixel;
    float distance{};
  };

  WorldPanel(grassland::graphics::Core *core,
             grassland::graphics::Window *window,
             int width,
             int height,
             const std::string &font_file);
  ~WorldPanel();
  WorldPanel(const WorldPanel &) = delete;
  WorldPanel &operator=(const WorldPanel &) = delete;

  Context &Controls() {
    return controls_;
  }

  void SetTransform(const glm::mat4 &local_to_world) {
    local_to_world_ = local_to_world;
  }

  std::optional<Hit> RayHit(glm::vec3 origin, glm::vec3 direction) const;
  void BeginFrame(std::optional<Hit> hit, bool down);
  void Render(grassland::graphics::CommandContext *commands,
              grassland::graphics::Image *scene,
              grassland::graphics::Image *depth,
              const glm::mat4 &view_projection);

 private:
  struct Vertex {
    glm::vec2 position, uv;
  };

  grassland::graphics::Program *GetProgram(grassland::graphics::ImageFormat color,
                                           grassland::graphics::ImageFormat depth);
  grassland::graphics::Core *core_{};
  Context controls_;
  glm::mat4 local_to_world_{1.0f};
  int width_{}, height_{};
  std::unique_ptr<grassland::graphics::Image> image_;
  std::unique_ptr<grassland::graphics::Buffer> vertices_, indices_, uniform_;
  std::unique_ptr<grassland::graphics::Sampler> sampler_;
  std::unique_ptr<grassland::graphics::Shader> vertex_shader_, pixel_shader_;
  std::map<std::pair<grassland::graphics::ImageFormat, grassland::graphics::ImageFormat>,
           std::unique_ptr<grassland::graphics::Program>>
      programs_;
};

}  // namespace snowberg::gui
