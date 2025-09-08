#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_imgui_assets.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace CD::graphics::backend {

class D3D12Window : public Window {
 public:
  D3D12Window(D3D12Core *core,
              int width,
              int height,
              const std::string &title,
              bool fullscreen,
              bool resizable,
              bool enable_hdr);
  ~D3D12Window();

  virtual void CloseWindow() override;

  d3d12::SwapChain *SwapChain() const;

  ID3D12Resource *CurrentBackBuffer() const;

  void InitImGui(const char *font_file_path, float font_size) override;
  void TerminateImGui() override;
  void BeginImGuiFrame() override;
  void EndImGuiFrame() override;
  ImGuiContext *GetImGuiContext() const override;

  D3D12ImGuiAssets &ImGuiAssets();
  void SetupImGuiContext();

 private:
  D3D12Core *core_;
  std::unique_ptr<d3d12::SwapChain> swap_chain_;
  uint32_t swap_chain_recreate_event_id_;
  D3D12ImGuiAssets imgui_assets_{};
};

}  // namespace CD::graphics::backend
