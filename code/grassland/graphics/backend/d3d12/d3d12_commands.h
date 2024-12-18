#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

class D3D12Command {
 public:
  virtual ~D3D12Command() = default;
  virtual void CompileCommand(D3D12CommandContext *context,
                              ID3D12GraphicsCommandList *command_list) = 0;
};

class D3D12CmdClearImage : public D3D12Command {
 public:
  D3D12CmdClearImage(D3D12Image *image, const ClearValue &clear_value);
  void CompileCommand(D3D12CommandContext *context,
                      ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Image *image_;
  ClearValue clear_value_;
};

class D3D12CmdPresent : public D3D12Command {
 public:
  D3D12CmdPresent(D3D12Window *window, D3D12Image *image);
  void CompileCommand(D3D12CommandContext *context,
                      ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Image *image_;
  D3D12Window *window_;
};

}  // namespace grassland::graphics::backend
