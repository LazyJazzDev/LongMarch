#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalWindow : public Window {
 public:
  MetalWindow(MetalCore *core, int width, int height, const std::string &title, bool fullscreen, bool resizable);
  ~MetalWindow() override;
  void CloseWindow() override;
  void SetHDR(bool enable_hdr) override;
  void InitImGui(const char *font_file_path = nullptr, float font_size = 13) override;
  void TerminateImGui() override;
  void BeginImGuiFrame() override;
  void EndImGuiFrame() override;

  ImGuiContext *GetImGuiContext() const override {
    return imgui_;
  }

  void Present(MTL::CommandBuffer *command, MetalImage *image);

 private:
  void ConfigurePresentation(bool enable_hdr);
  MetalCore *core_;
  NS::SharedPtr<CA::MetalLayer> layer_;
  NS::SharedPtr<MTL::RenderPipelineState> pipeline_;
  ImGuiContext *imgui_ = nullptr;
};

}  // namespace grassland::graphics::backend
