#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_imgui_assets.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

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

  IDXGISwapChain3 *SwapChain() const;

  DXGI_FORMAT BackBufferFormat() const;

  Extent2D BackBufferExtent() const;

  ID3D12Resource *CurrentBackBuffer() const;

  void InitImGui(const char *font_file_path, float font_size) override;
  void TerminateImGui() override;
  void BeginImGuiFrame() override;
  void EndImGuiFrame() override;
  ImGuiContext *GetImGuiContext() const override;

  D3D12ImGuiAssets &ImGuiAssets();
  void SetupImGuiContext();

 private:
  void RecreateSwapChain();

  D3D12Core *core_;
  Microsoft::WRL::ComPtr<IDXGISwapChain3> swap_chain_;
  std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> back_buffers_;
  uint32_t swap_chain_recreate_event_id_;
  D3D12ImGuiAssets imgui_assets_{};
};

}  // namespace grassland::graphics::backend
