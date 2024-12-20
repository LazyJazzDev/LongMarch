#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"

#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_program.h"
#include "grassland/graphics/backend/d3d12/d3d12_sampler.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

D3D12CommandContext::D3D12CommandContext(D3D12Core *core) : core_(core) {
}

D3D12Core *D3D12CommandContext::Core() const {
  return core_;
}

void D3D12CommandContext::CmdBindProgram(Program *program) {
  auto d3d12_program = dynamic_cast<D3D12Program *>(program);
  commands_.push_back(std::make_unique<D3D12CmdBindProgram>(d3d12_program));
  program_ = d3d12_program;
}

void D3D12CommandContext::CmdBindVertexBuffers(
    uint32_t first_binding,
    const std::vector<Buffer *> &buffers,
    const std::vector<uint64_t> &offsets) {
  std::vector<D3D12Buffer *> vertex_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    vertex_buffers[i] = dynamic_cast<D3D12Buffer *>(buffers[i]);
    RecordDynamicBuffer(vertex_buffers[i]);
  }
  commands_.push_back(std::make_unique<D3D12CmdBindVertexBuffers>(
      first_binding, vertex_buffers, offsets, program_));
}

void D3D12CommandContext::CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) {
  auto index_buffer = dynamic_cast<D3D12Buffer *>(buffer);
  commands_.push_back(
      std::make_unique<D3D12CmdBindIndexBuffer>(index_buffer, offset));
  RecordDynamicBuffer(index_buffer);
}

void D3D12CommandContext::CmdBindResources(
    int slot,
    const std::vector<Buffer *> &buffers) {
  auto descriptor_range = program_->DescriptorRange(slot);
  resource_descriptor_count_ += descriptor_range->NumDescriptors;
  std::vector<D3D12Buffer *> d3d12_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    d3d12_buffers[i] = dynamic_cast<D3D12Buffer *>(buffers[i]);
    RecordDynamicBuffer(d3d12_buffers[i]);
  }
  commands_.push_back(std::make_unique<D3D12CmdBindResourceBuffers>(
      slot, d3d12_buffers, program_));
}

void D3D12CommandContext::CmdBindResources(int slot,
                                           const std::vector<Image *> &images) {
  auto descriptor_range = program_->DescriptorRange(slot);
  resource_descriptor_count_ += descriptor_range->NumDescriptors;
  std::vector<D3D12Image *> d3d12_images(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    d3d12_images[i] = dynamic_cast<D3D12Image *>(images[i]);
  }
  commands_.push_back(std::make_unique<D3D12CmdBindResourceImages>(
      slot, d3d12_images, program_));
}

void D3D12CommandContext::CmdBindResources(
    int slot,
    const std::vector<Sampler *> &samplers) {
  auto descriptor_range = program_->DescriptorRange(slot);
  sampler_descriptor_count_ += descriptor_range->NumDescriptors;
  std::vector<D3D12Sampler *> d3d12_samplers(samplers.size());
  for (size_t i = 0; i < samplers.size(); ++i) {
    d3d12_samplers[i] = dynamic_cast<D3D12Sampler *>(samplers[i]);
  }
  commands_.push_back(std::make_unique<D3D12CmdBindResourceSamplers>(
      slot, d3d12_samplers, program_));
}

void D3D12CommandContext::CmdBeginRendering(
    const std::vector<Image *> &color_targets,
    Image *depth_target) {
  std::vector<D3D12Image *> d3d12_color_targets(color_targets.size());
  D3D12Image *d3d12_depth_target{nullptr};
  for (size_t i = 0; i < color_targets.size(); ++i) {
    d3d12_color_targets[i] = dynamic_cast<D3D12Image *>(color_targets[i]);
    RecordRTVImage(d3d12_color_targets[i]);
  }
  if (depth_target) {
    d3d12_depth_target = dynamic_cast<D3D12Image *>(depth_target);
    RecordDSVImage(d3d12_depth_target);
  }
  commands_.push_back(std::make_unique<D3D12CmdBeginRendering>(
      d3d12_color_targets, d3d12_depth_target));
}

void D3D12CommandContext::CmdEndRendering() {
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
                                         int32_t vertex_offset,
                                         uint32_t first_instance) {
  commands_.push_back(std::make_unique<D3D12CmdDrawIndexed>(
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

CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteUAVDescriptor(
    D3D12Image *image) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
  desc.Format = ImageFormatToDXGIFormat(image->Format());
  desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  desc.Texture2D.MipSlice = 0;
  desc.Texture2D.PlaneSlice = 0;

  core_->Device()->Handle()->CreateUnorderedAccessView(
      image->Image()->Handle(), nullptr, &desc, resource_descriptor_base_);
  resource_descriptor_base_.Offset(resource_descriptor_size_);
  auto result = resource_descriptor_gpu_base_;
  resource_descriptor_gpu_base_.Offset(resource_descriptor_size_);
  return result;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteSRVDescriptor(
    D3D12Image *image) {
  D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
  desc.Format = ImageFormatToDXGIFormat(image->Format());
  if (desc.Format == DXGI_FORMAT_D32_FLOAT) {
    desc.Format = DXGI_FORMAT_R32_FLOAT;
  }
  desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  desc.Texture2D.MostDetailedMip = 0;
  desc.Texture2D.MipLevels = 1;
  desc.Texture2D.PlaneSlice = 0;
  desc.Texture2D.ResourceMinLODClamp = 0.0f;

  core_->Device()->Handle()->CreateShaderResourceView(
      image->Image()->Handle(), &desc, resource_descriptor_base_);

  resource_descriptor_base_.Offset(resource_descriptor_size_);
  auto result = resource_descriptor_gpu_base_;
  resource_descriptor_gpu_base_.Offset(resource_descriptor_size_);
  return result;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteSRVDescriptor(
    D3D12Buffer *buffer) {
  // D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
  // desc.Format = DXGI_FORMAT_UNKNOWN;
  // desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  // desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  // desc.Buffer.FirstElement = 0;
  // desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
  // desc.Buffer.NumElements = static_cast<UINT>(buffer->Size());
  // desc.Buffer.StructureByteStride = 0;

  core_->Device()->Handle()->CreateShaderResourceView(
      buffer->Buffer()->Handle(), nullptr, resource_descriptor_base_);

  resource_descriptor_base_.Offset(resource_descriptor_size_);
  auto result = resource_descriptor_gpu_base_;
  resource_descriptor_gpu_base_.Offset(resource_descriptor_size_);
  return result;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE D3D12CommandContext::WriteCBVDescriptor(
    D3D12Buffer *buffer) {
  D3D12_CONSTANT_BUFFER_VIEW_DESC desc = {};
  desc.BufferLocation = buffer->Buffer()->Handle()->GetGPUVirtualAddress();
  desc.SizeInBytes = static_cast<UINT>(buffer->Size());

  core_->Device()->Handle()->CreateConstantBufferView(
      &desc, resource_descriptor_base_);

  resource_descriptor_base_.Offset(resource_descriptor_size_);
  auto result = resource_descriptor_gpu_base_;
  resource_descriptor_gpu_base_.Offset(resource_descriptor_size_);
  return result;
}

CD3DX12_GPU_DESCRIPTOR_HANDLE
D3D12CommandContext::WriteSamplerDescriptor(const D3D12_SAMPLER_DESC &desc) {
  core_->Device()->Handle()->CreateSampler(&desc, sampler_descriptor_base_);

  sampler_descriptor_base_.Offset(sampler_descriptor_size_);
  auto result = sampler_descriptor_gpu_base_;
  sampler_descriptor_gpu_base_.Offset(sampler_descriptor_size_);
  return result;
}

void D3D12CommandContext::RecordDynamicBuffer(D3D12Buffer *buffer) {
  auto *dynamic_buffer = dynamic_cast<D3D12DynamicBuffer *>(buffer);
  if (dynamic_buffer) {
    dynamic_buffers_.insert(dynamic_buffer);
  }
}

}  // namespace grassland::graphics::backend
