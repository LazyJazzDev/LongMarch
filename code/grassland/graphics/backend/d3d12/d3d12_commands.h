#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

class D3D12Command {
 public:
  virtual ~D3D12Command() = default;
  virtual void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) = 0;
};

class D3D12CmdBindProgram : public D3D12Command {
 public:
  D3D12CmdBindProgram(D3D12Program *program);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Program *program_;
};

class D3D12CmdBindRayTracingProgram : public D3D12Command {
 public:
  D3D12CmdBindRayTracingProgram(D3D12RayTracingProgram *program);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12RayTracingProgram *program_;
};

class D3D12CmdBindComputeProgram : public D3D12Command {
 public:
  D3D12CmdBindComputeProgram(D3D12ComputeProgram *program);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12ComputeProgram *program_;
};

class D3D12CmdBindVertexBuffers : public D3D12Command {
 public:
  D3D12CmdBindVertexBuffers(uint32_t first_binding,
                            const std::vector<D3D12Buffer *> &buffers,
                            const std::vector<uint64_t> &offsets,
                            D3D12Program *program);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  uint32_t first_binding_;
  std::vector<D3D12Buffer *> buffers_;
  std::vector<uint64_t> offsets_;
  D3D12Program *program_;
};

class D3D12CmdBindIndexBuffer : public D3D12Command {
 public:
  D3D12CmdBindIndexBuffer(D3D12Buffer *buffer, uint64_t offset);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Buffer *buffer_;
  uint64_t offset_;
};

class D3D12CmdBindResourceBuffers : public D3D12Command {
 public:
  D3D12CmdBindResourceBuffers(int slot,
                              const std::vector<D3D12BufferRange> &buffers,
                              D3D12ProgramBase *program,
                              BindPoint bind_point);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  int slot_;
  std::vector<D3D12BufferRange> buffers_;
  D3D12ProgramBase *program_;
  BindPoint bind_point_;
};

class D3D12CmdBindResourceImages : public D3D12Command {
 public:
  D3D12CmdBindResourceImages(int slot,
                             const std::vector<D3D12Image *> &images,
                             D3D12ProgramBase *program,
                             BindPoint bind_point);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  int slot_;
  std::vector<D3D12Image *> images_;
  D3D12ProgramBase *program_;
  BindPoint bind_point_;
};

class D3D12CmdBindResourceSamplers : public D3D12Command {
 public:
  D3D12CmdBindResourceSamplers(int slot,
                               const std::vector<D3D12Sampler *> &samplers,
                               D3D12ProgramBase *program,
                               BindPoint bind_point);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  int slot_;
  std::vector<D3D12Sampler *> samplers_;
  D3D12ProgramBase *program_;
  BindPoint bind_point_;
};

class D3D12CmdBindResourceAccelerationStructure : public D3D12Command {
 public:
  D3D12CmdBindResourceAccelerationStructure(int slot,
                                            D3D12AccelerationStructure *acceleration_structure,
                                            D3D12ProgramBase *program,
                                            BindPoint bind_point);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  int slot_;
  D3D12AccelerationStructure *acceleration_structure_;
  D3D12ProgramBase *program_;
  BindPoint bind_point_;
};

class D3D12CmdBeginRendering : public D3D12Command {
 public:
  D3D12CmdBeginRendering(const std::vector<D3D12Image *> &color_targets, D3D12Image *depth_target);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  std::vector<D3D12Image *> color_targets_;
  D3D12Image *depth_target_;
};

class D3D12CmdClearImage : public D3D12Command {
 public:
  D3D12CmdClearImage(D3D12Image *image, const ClearValue &clear_value);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Image *image_;
  ClearValue clear_value_;
};

class D3D12CmdSetViewport : public D3D12Command {
 public:
  D3D12CmdSetViewport(const Viewport &viewport);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  Viewport viewport_;
};

class D3D12CmdSetScissor : public D3D12Command {
 public:
  D3D12CmdSetScissor(const Scissor &scissor);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  Scissor scissor_;
};

class D3D12CmdSetPrimitiveTopology : public D3D12Command {
 public:
  D3D12CmdSetPrimitiveTopology(PrimitiveTopology topology);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  PrimitiveTopology topology_;
};

class D3D12CmdDraw : public D3D12Command {
 public:
  D3D12CmdDraw(uint32_t index_count, uint32_t instance_count, int32_t vertex_offset, uint32_t first_instance);

  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  uint32_t index_count_;
  uint32_t instance_count_;
  int32_t vertex_offset_;
  uint32_t first_instance_;
};

class D3D12CmdDrawIndexed : public D3D12Command {
 public:
  D3D12CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t vertex_offset,
                      uint32_t first_instance);

  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  uint32_t index_count_;
  uint32_t instance_count_;
  uint32_t first_index_;
  int32_t vertex_offset_;
  uint32_t first_instance_;
};

class D3D12CmdPresent : public D3D12Command {
 public:
  D3D12CmdPresent(D3D12Window *window, D3D12Image *image);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Image *image_;
  D3D12Window *window_;
};

class D3D12CmdDispatchRays : public D3D12Command {
 public:
  D3D12CmdDispatchRays(D3D12RayTracingProgram *program, uint32_t width, uint32_t height, uint32_t depth);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12RayTracingProgram *program_;
  uint32_t width_;
  uint32_t height_;
  uint32_t depth_;
};

class D3D12CmdDispatch : public D3D12Command {
 public:
  D3D12CmdDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  uint32_t group_count_x_;
  uint32_t group_count_y_;
  uint32_t group_count_z_;
};

class D3D12CmdCopyBuffer : public D3D12Command {
 public:
  D3D12CmdCopyBuffer(D3D12Buffer *dst_buffer,
                     D3D12Buffer *src_buffer,
                     uint64_t size,
                     uint64_t dst_offset,
                     uint64_t src_offset);
  void CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) override;

 private:
  D3D12Buffer *dst_buffer_;
  D3D12Buffer *src_buffer_;
  uint64_t size_;
  uint64_t dst_offset_;
  uint64_t src_offset_;
};

}  // namespace grassland::graphics::backend
