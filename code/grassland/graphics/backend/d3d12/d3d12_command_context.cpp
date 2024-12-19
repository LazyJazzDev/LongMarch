#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"

#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_program.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

D3D12CommandContext::D3D12CommandContext(D3D12Core *core) : core_(core) {
}

D3D12Core *D3D12CommandContext::Core() const {
  return core_;
}

void D3D12CommandContext::BindColorTargets(const std::vector<Image *> &images) {
  color_targets_.resize(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    auto d3d12_image = dynamic_cast<D3D12Image *>(images[i]);
    color_targets_[i] = d3d12_image;
    RecordRTVImage(d3d12_image);
  }
}

void D3D12CommandContext::BindDepthTarget(Image *image) {
  auto d3d12_image = dynamic_cast<D3D12Image *>(image);
  depth_target_ = d3d12_image;
  RecordDSVImage(d3d12_image);
}

void D3D12CommandContext::BindVertexBuffers(
    const std::vector<Buffer *> &buffers) {
  vertex_buffers_.resize(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    auto d3d12_buffer = dynamic_cast<D3D12Buffer *>(buffers[i]);
    vertex_buffers_[i] = d3d12_buffer;
    if (d3d12_buffer) {
      auto dynamic_buffer = dynamic_cast<D3D12DynamicBuffer *>(d3d12_buffer);
      if (dynamic_buffer) {
        dynamic_buffers_.insert(dynamic_buffer);
      }
    }
  }
}

void D3D12CommandContext::BindIndexBuffer(Buffer *buffer) {
  index_buffer_ = dynamic_cast<D3D12Buffer *>(buffer);
  if (index_buffer_) {
    auto dynamic_buffer = dynamic_cast<D3D12DynamicBuffer *>(index_buffer_);
    if (dynamic_buffer) {
      dynamic_buffers_.insert(dynamic_buffer);
    }
  }
}

void D3D12CommandContext::BindProgram(Program *program) {
  program_ = dynamic_cast<D3D12Program *>(program);
}

void D3D12CommandContext::CmdSetViewport(const Viewport &viewport) {
  commands_.push_back(std::make_unique<D3D12CmdSetViewport>(viewport));
}

void D3D12CommandContext::CmdSetScissor(const Scissor &scissor) {
  commands_.push_back(std::make_unique<D3D12CmdSetScissor>(scissor));
}

void D3D12CommandContext::CmdDrawIndexed(uint32_t index_count,
                                         uint32_t instance_count,
                                         uint32_t first_index,
                                         uint32_t vertex_offset,
                                         uint32_t first_instance) {
  commands_.push_back(std::make_unique<D3D12CmdDrawIndexed>(
      program_, vertex_buffers_, index_buffer_, color_targets_, depth_target_,
      index_count, instance_count, first_index, vertex_offset, first_instance));
}

void D3D12CommandContext::CmdClearImage(Image *image, const ClearValue &color) {
  auto d3d12_image = dynamic_cast<D3D12Image *>(image);
  commands_.push_back(std::make_unique<D3D12CmdClearImage>(d3d12_image, color));
  if (IsDepthFormat(d3d12_image->Format())) {
    RecordDSVImage(d3d12_image);
  } else {
    RecordRTVImage(d3d12_image);
  }
}

void D3D12CommandContext::CmdPresent(Window *window, Image *image) {
  auto d3d12_window = dynamic_cast<D3D12Window *>(window);
  auto d3d12_image = dynamic_cast<D3D12Image *>(image);
  commands_.push_back(
      std::make_unique<D3D12CmdPresent>(d3d12_window, d3d12_image));
  windows_.insert(d3d12_window);
  resource_descriptor_count_++;
}

void D3D12CommandContext::RecordRTVImage(const D3D12Image *image) {
  RecordRTVImage(image->Image()->Handle());
}

void D3D12CommandContext::RecordDSVImage(const D3D12Image *image) {
  RecordDSVImage(image->Image()->Handle());
}

void D3D12CommandContext::RecordRTVImage(ID3D12Resource *resource) {
  if (rtv_index_.count(resource) == 0) {
    const int index = rtv_index_.size();
    rtv_index_[resource] = index;
  }
}

void D3D12CommandContext::RecordDSVImage(ID3D12Resource *resource) {
  if (dsv_index_.count(resource) == 0) {
    const int index = dsv_index_.size();
    dsv_index_[resource] = index;
  }
}

void D3D12CommandContext::RequireImageState(
    ID3D12GraphicsCommandList *command_list,
    ID3D12Resource *resource,
    const D3D12_RESOURCE_STATES state) {
  if (resource_states_.count(resource) == 0) {
    resource_states_[resource] = D3D12_RESOURCE_STATE_GENERIC_READ;
  }

  if (state != resource_states_[resource]) {
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resource, resource_states_[resource], state);
    command_list->ResourceBarrier(1, &barrier);
    resource_states_[resource] = state;
  }
}

CD3DX12_CPU_DESCRIPTOR_HANDLE D3D12CommandContext::RTVHandle(
    ID3D12Resource *resource) const {
  return core_->RTVDescriptorHeap()->CPUHandle(rtv_index_.at(resource));
}

CD3DX12_CPU_DESCRIPTOR_HANDLE D3D12CommandContext::DSVHandle(
    ID3D12Resource *resource) const {
  return core_->DSVDescriptorHeap()->CPUHandle(dsv_index_.at(resource));
}

CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteDescriptor(
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
