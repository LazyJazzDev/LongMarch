#pragma once
#include "snow_mount/draw/draw_commands.h"
#include "snow_mount/draw/draw_model.h"
#include "snow_mount/draw/draw_texture.h"
#include "snow_mount/draw/draw_util.h"

namespace snow_mount::draw {

class Core {
 public:
  Core(graphics::Core *core);

  void BeginDraw();
  void EndDraw();
  void Render(graphics::CommandContext *context, graphics::Image *image);

  void CmdSetDrawRegion(int x, int y, int width, int height);
  void CmdDrawInstance(Model *model,
                       Texture *texture,
                       const Transform &transform,
                       glm::vec4 color);
  void CmdDrawInstance(Model *model,
                       const Transform &transform = Transform{1.0f},
                       glm::vec4 color = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f});
  void CmdDrawInstance(Model *model, glm::vec4 color);
  void CmdDrawText(glm::vec2 origin, const std::string &text, glm::vec4 color);

  void CreateModel(double_ptr<Model> model);
  void CreateTexture(int width, int height, double_ptr<Texture> texture);

  graphics::Core *GraphicsCore() const {
    return core_;
  }

  graphics::Program *GetProgram(graphics::ImageFormat format);

  FontCore *GetFontCore() const {
    return font_core_.get();
  }

 private:
  void CmdDrawInstance(Model *model,
                       graphics::Image *texture,
                       const Transform &transform,
                       glm::vec4 color);
  graphics::Core *core_;
  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> fragment_shader_;
  std::map<graphics::ImageFormat, std::unique_ptr<graphics::Program>> programs_;
  std::unique_ptr<graphics::Buffer> metadata_buffer_;
  std::unique_ptr<graphics::Buffer> instance_index_buffer_;

  std::vector<DrawMetadata> draw_metadata_;
  std::vector<std::unique_ptr<DrawCommand>> commands_;

  std::unique_ptr<graphics::Image> pure_white_texture_;
  std::unique_ptr<graphics::Sampler> linear_sampler_;

  std::unique_ptr<Model> text_model_;

  std::unique_ptr<FontCore> font_core_;
  Transform pixel_transform_;
};

void CreateCore(graphics::Core *core, double_ptr<Core> draw_core);

}  // namespace snow_mount::draw
