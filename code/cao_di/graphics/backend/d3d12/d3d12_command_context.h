#pragma once
#include "cao_di/graphics/backend/d3d12/d3d12_buffer.h"
#include "cao_di/graphics/backend/d3d12/d3d12_command_context.h"
#include "cao_di/graphics/backend/d3d12/d3d12_commands.h"
#include "cao_di/graphics/backend/d3d12/d3d12_core.h"
#include "cao_di/graphics/backend/d3d12/d3d12_image.h"
#include "cao_di/graphics/backend/d3d12/d3d12_util.h"
#include "cao_di/graphics/backend/d3d12/d3d12_window.h"

namespace CD::graphics::backend {

class D3D12CommandContext : public CommandContext {
 public:
  D3D12CommandContext(D3D12Core *core);

  D3D12Core *Core() const;

  graphics::Core *GetCore() const override;

  void CmdBindProgram(Program *program) override;
  void CmdBindRayTracingProgram(RayTracingProgram *program) override;
  void CmdBindComputeProgram(ComputeProgram *program) override;
  void CmdBindVertexBuffers(uint32_t first_binding,
                            const std::vector<Buffer *> &buffers,
                            const std::vector<uint64_t> &offsets) override;
  void CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) override;
  void CmdBindResources(int slot, const std::vector<BufferRange> &buffers, BindPoint bind_point) override;
  void CmdBindResources(int slot, const std::vector<Image *> &images, BindPoint bind_point) override;
  void CmdBindResources(int slot, const std::vector<Sampler *> &samplers, BindPoint bind_point) override;
  void CmdBindResources(int slot, AccelerationStructure *acceleration_structure, BindPoint bind_point) override;

  void CmdBeginRendering(const std::vector<Image *> &color_targets, Image *depth_target) override;
  void CmdEndRendering() override;

  void CmdSetViewport(const Viewport &viewport) override;
  void CmdSetScissor(const Scissor &scissor) override;
  void CmdSetPrimitiveTopology(PrimitiveTopology topology) override;
  void CmdDraw(uint32_t index_count, uint32_t instance_count, int32_t vertex_offset, uint32_t first_instance) override;
  void CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      int32_t vertex_offset,
                      uint32_t first_instance) override;
  void CmdClearImage(Image *image, const ClearValue &color) override;
  void CmdPresent(Window *window, Image *image) override;
  void CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth) override;
  void CmdDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) override;
  void CmdCopyBuffer(Buffer *src_buffer,
                     Buffer *dst_buffer,
                     uint64_t size,
                     uint64_t src_offset,
                     uint64_t dst_offset) override;

  void RecordRTVImage(const D3D12Image *image);
  void RecordDSVImage(const D3D12Image *image);
  void RecordRTVImage(ID3D12Resource *resource);
  void RecordDSVImage(ID3D12Resource *resource);

  void RequireResourceState(ID3D12GraphicsCommandList *command_list,
                            ID3D12Resource *resource,
                            D3D12_RESOURCE_STATES state);

  CD3DX12_CPU_DESCRIPTOR_HANDLE RTVHandle(ID3D12Resource *resource) const;
  CD3DX12_CPU_DESCRIPTOR_HANDLE DSVHandle(ID3D12Resource *resource) const;

  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteUAVDescriptor(D3D12Image *image);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteSRVDescriptor(D3D12Image *image);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteSRVDescriptor(D3D12BufferRange buffer);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteSRVDescriptor(D3D12AccelerationStructure *acceleration_structure);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteCBVDescriptor(D3D12BufferRange buffer);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteUAVDescriptor(D3D12BufferRange buffer);
  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteSamplerDescriptor(const D3D12_SAMPLER_DESC &desc);

  void RecordDynamicBuffer(D3D12Buffer *buffer);

 private:
  friend D3D12Core;
  D3D12Core *core_;

  D3D12ProgramBase *program_bases_[BIND_POINT_COUNT]{};

  std::map<ID3D12Resource *, int> rtv_index_;
  std::map<ID3D12Resource *, int> dsv_index_;

  std::vector<std::unique_ptr<D3D12Command>> commands_;
  std::set<D3D12Window *> windows_;

  std::set<D3D12DynamicBuffer *> dynamic_buffers_;

  std::map<ID3D12Resource *, D3D12_RESOURCE_STATES> resource_states_;
  int resource_descriptor_count_{0};
  int resource_descriptor_size_{};
  CD3DX12_CPU_DESCRIPTOR_HANDLE resource_descriptor_base_{};
  CD3DX12_GPU_DESCRIPTOR_HANDLE resource_descriptor_gpu_base_{};

  int sampler_descriptor_count_{0};
  int sampler_descriptor_size_{};
  CD3DX12_CPU_DESCRIPTOR_HANDLE sampler_descriptor_base_{};
  CD3DX12_GPU_DESCRIPTOR_HANDLE sampler_descriptor_gpu_base_{};
};

}  // namespace CD::graphics::backend
