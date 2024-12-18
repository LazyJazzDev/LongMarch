#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"
#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"
#include "grassland/graphics/backend/d3d12/d3d12_commands.h"
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

class D3D12CommandContext : public CommandContext {
 public:
  D3D12CommandContext(D3D12Core *core);

  D3D12Core *Core() const;

  void BindColorTargets(const std::vector<Image *> &images) override;
  void BindDepthTarget(Image *image) override;
  void BindVertexBuffers(const std::vector<Buffer *> &buffers) override;
  void BindIndexBuffer(Buffer *buffer) override;
  void BindProgram(Program *program) override;

  void CmdClearImage(Image *image, const ClearValue &color) override;
  void CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count,
                      uint32_t first_index,
                      uint32_t vertex_offset,
                      uint32_t first_instance) override;
  void CmdPresent(Window *window, Image *image) override;

  void RecordRTVImage(const D3D12Image *image);
  void RecordDSVImage(const D3D12Image *image);
  void RecordRTVImage(ID3D12Resource *resource);
  void RecordDSVImage(ID3D12Resource *resource);

  void RequireImageState(ID3D12GraphicsCommandList *command_list,
                         ID3D12Resource *resource,
                         D3D12_RESOURCE_STATES state);

  CD3DX12_CPU_DESCRIPTOR_HANDLE RTVHandle(ID3D12Resource *resource) const;
  CD3DX12_CPU_DESCRIPTOR_HANDLE DSVHandle(ID3D12Resource *resource) const;

  CD3DX12_GPU_DESCRIPTOR_HANDLE WriteDescriptor(D3D12Image *image);

 private:
  friend D3D12Core;
  D3D12Core *core_;

  D3D12Program *program_{nullptr};
  std::vector<D3D12Image *> color_targets_;
  D3D12Image *depth_target_{nullptr};
  std::vector<D3D12Buffer *> vertex_buffers_;
  D3D12Buffer *index_buffer_{nullptr};

  std::map<ID3D12Resource *, int> rtv_index_;
  std::map<ID3D12Resource *, int> dsv_index_;

  std::vector<std::unique_ptr<D3D12Command>> commands_;
  std::set<D3D12Window *> windows_;

  std::map<ID3D12Resource *, D3D12_RESOURCE_STATES> resource_states_;
  int resource_descriptor_count_{0};
  int descriptor_size_{};
  CD3DX12_CPU_DESCRIPTOR_HANDLE resource_descriptor_base_{};
  CD3DX12_GPU_DESCRIPTOR_HANDLE resource_descriptor_gpu_base_{};
};

inline CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteDescriptor(
    D3D12Image *image) {
  D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
  desc.Format = ImageFormatToDXGIFormat(image->Format());
  desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  desc.Texture2D.MostDetailedMip = 0;
  desc.Texture2D.MipLevels = 1;
  desc.Texture2D.PlaneSlice = 0;
  desc.Texture2D.ResourceMinLODClamp = 0.0f;

  core_->Device()->Handle()->CreateShaderResourceView(
      image->Image()->Handle(), &desc, resource_descriptor_base_);

  resource_descriptor_base_.Offset(descriptor_size_);
  auto result = resource_descriptor_gpu_base_;
  resource_descriptor_gpu_base_.Offset(descriptor_size_);
  return result;
}

}  // namespace grassland::graphics::backend
