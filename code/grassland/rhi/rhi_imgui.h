#pragma once
#include "grassland/rhi/rhi.h"

struct ImGuiContext;
struct ImDrawData;

namespace grassland::rhi {
// Window-independent ImGui context and renderer. The host supplies input events,
// DisplaySize, DisplayFramebufferScale and DeltaTime before Update. Calls and
// context access must be serialized on the host's rendering thread.
// Device is borrowed. Texture IDs are borrowed IImage pointers during Record;
// recorded commands retain the resources they use.
class ImGuiRenderer final {
 public:
  static std::unique_ptr<ImGuiRenderer> Create(IDevice *device,
                                               const char *font_file_path = nullptr,
                                               float font_size = 18.0f);
  ~ImGuiRenderer();

  ImGuiContext *Context() const {
    return context_;
  }

  bool Update(const std::function<void()> &callback);
  bool Record(ICommandContext *commands,
              IImage *image,
              ColorSpace color_space = COLOR_SPACE_SRGB_NONLINEAR,
              float reference_white_scale = 1.0f);

 private:
  explicit ImGuiRenderer(IDevice *device);
  bool Initialize(const char *font_file_path, float font_size);
  bool Render(ICommandContext *commands,
              IImage *image,
              ImDrawData *data,
              ColorSpace color_space,
              float reference_white_scale);
  IDevice *device_;
  ImGuiContext *context_{};
  bool draw_data_ready_{};
  ObjPtr<IImage> font_;
  ObjPtr<ISampler> sampler_;
  ObjPtr<IProgram> program_;
  ObjPtr<IBuffer> vertex_buffer_;
  ObjPtr<IBuffer> index_buffer_;
  ObjPtr<IBuffer> display_settings_;
  ColorSpace program_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
};
}  // namespace grassland::rhi
