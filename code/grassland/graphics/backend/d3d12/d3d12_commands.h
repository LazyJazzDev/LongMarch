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

class D3D12CmdSetViewport : public D3D12Command {
 public:
  D3D12CmdSetViewport(const Viewport &viewport);
  void CompileCommand(D3D12CommandContext *context,
                      ID3D12GraphicsCommandList *command_list) override;

 private:
  Viewport viewport_;
};

class D3D12CmdSetScissor : public D3D12Command {
 public:
  D3D12CmdSetScissor(const Scissor &scissor);
  void CompileCommand(D3D12CommandContext *context,
                      ID3D12GraphicsCommandList *command_list) override;

 private:
  Scissor scissor_;
};

class D3D12CmdDrawIndexed : public D3D12Command {
 public:
  D3D12CmdDrawIndexed(D3D12Program *program,
                      const std::vector<D3D12Buffer *> &vertex_buffers,
                      D3D12Buffer *index_buffer,
                      const std::vector<D3D12Image *> &color_targets,
                      D3D12Image *depth_target,
                      uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      uint32_t vertex_offset,
                      uint32_t first_instance);

  void CompileCommand(D3D12CommandContext *context,
                      ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Program *program_;
  std::vector<D3D12Buffer *> vertex_buffers_;
  D3D12Buffer *index_buffer_;
  std::vector<D3D12Image *> color_targets_;
  D3D12Image *depth_target_;
  uint32_t index_count_;
  uint32_t instance_count_;
  uint32_t first_index_;
  uint32_t vertex_offset_;
  uint32_t first_instance_;
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
