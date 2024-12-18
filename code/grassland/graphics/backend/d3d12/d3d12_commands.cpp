#include "grassland/graphics/backend/d3d12/d3d12_commands.h"

#include "grassland/graphics/backend/d3d12/d3d12_command_context.h"
#include "grassland/graphics/backend/d3d12/d3d12_image.h"
#include "grassland/graphics/backend/d3d12/d3d12_program.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

D3D12CmdClearImage::D3D12CmdClearImage(D3D12Image *image,
                                       const ClearValue &clear_value)
    : image_(image), clear_value_(clear_value) {
}

void D3D12CmdClearImage::CompileCommand(
    D3D12CommandContext *context,
    ID3D12GraphicsCommandList *command_list) {
  if (IsDepthFormat(image_->Format())) {
    context->RequireImageState(command_list, image_->Image()->Handle(),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.DepthStencil.Depth = clear_value_.depth.depth;
    clear_value.DepthStencil.Stencil = 0;
    command_list->ClearDepthStencilView(
        context->DSVHandle(image_->Image()->Handle()), D3D12_CLEAR_FLAG_DEPTH,
        clear_value.DepthStencil.Depth, clear_value.DepthStencil.Stencil, 0,
        nullptr);
  } else {
    context->RequireImageState(command_list, image_->Image()->Handle(),
                               D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.Color[0] = clear_value_.color.r;
    clear_value.Color[1] = clear_value_.color.g;
    clear_value.Color[2] = clear_value_.color.b;
    clear_value.Color[3] = clear_value_.color.a;
    command_list->ClearRenderTargetView(
        context->RTVHandle(image_->Image()->Handle()), clear_value.Color, 0,
        nullptr);
  }
}

D3D12CmdSetViewport::D3D12CmdSetViewport(const Viewport &viewport)
    : viewport_(viewport) {
}

void D3D12CmdSetViewport::CompileCommand(
    D3D12CommandContext *context,
    ID3D12GraphicsCommandList *command_list) {
  D3D12_VIEWPORT viewport;
  viewport.TopLeftX = viewport_.x;
  viewport.TopLeftY = viewport_.y;
  viewport.Width = viewport_.width;
  viewport.Height = viewport_.height;
  viewport.MinDepth = viewport_.min_depth;
  viewport.MaxDepth = viewport_.max_depth;
  command_list->RSSetViewports(1, &viewport);
}

D3D12CmdSetScissor::D3D12CmdSetScissor(const Scissor &scissor)
    : scissor_(scissor) {
}

void D3D12CmdSetScissor::CompileCommand(
    D3D12CommandContext *context,
    ID3D12GraphicsCommandList *command_list) {
  D3D12_RECT scissor_rect;
  scissor_rect.left = scissor_.offset.x;
  scissor_rect.top = scissor_.offset.y;
  scissor_rect.right = scissor_.offset.x + scissor_.extent.width;
  scissor_rect.bottom = scissor_.offset.y + scissor_.extent.height;
  command_list->RSSetScissorRects(1, &scissor_rect);
}

D3D12CmdDrawIndexed::D3D12CmdDrawIndexed(
    D3D12Program *program,
    const std::vector<D3D12Buffer *> &vertex_buffers,
    D3D12Buffer *index_buffer,
    const std::vector<D3D12Image *> &color_targets,
    D3D12Image *depth_target,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    uint32_t vertex_offset,
    uint32_t first_instance)
    : program_(program),
      vertex_buffers_({}),
      index_buffer_(index_buffer),
      color_targets_({}),
      depth_target_(nullptr),
      index_count_(index_count),
      instance_count_(instance_count),
      first_index_(first_index),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
  vertex_buffers_.resize(program_->NumInputBindings());
  for (size_t i = 0; i < vertex_buffers_.size(); ++i) {
    vertex_buffers_[i] = vertex_buffers[i];
  }
  color_targets_.resize(program_->PipelineStateDesc()->NumRenderTargets);
  for (size_t i = 0; i < color_targets_.size(); ++i) {
    color_targets_[i] = color_targets[i];
  }
  if (program_->PipelineStateDesc()->DSVFormat != DXGI_FORMAT_UNKNOWN) {
    depth_target_ = depth_target;
  }
}

void D3D12CmdDrawIndexed::CompileCommand(
    D3D12CommandContext *context,
    ID3D12GraphicsCommandList *command_list) {
  command_list->SetGraphicsRootSignature(program_->RootSignature()->Handle());
  command_list->SetPipelineState(program_->PipelineState()->Handle());
  CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handles[8]{};
  CD3DX12_CPU_DESCRIPTOR_HANDLE dsv_handle{};
  for (size_t i = 0; i < color_targets_.size(); ++i) {
    context->RequireImageState(command_list,
                               color_targets_[i]->Image()->Handle(),
                               D3D12_RESOURCE_STATE_RENDER_TARGET);
    rtv_handles[i] = context->RTVHandle(color_targets_[i]->Image()->Handle());
  }
  if (depth_target_) {
    context->RequireImageState(command_list, depth_target_->Image()->Handle(),
                               D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dsv_handle = context->DSVHandle(depth_target_->Image()->Handle());
    command_list->OMSetRenderTargets(color_targets_.size(), rtv_handles, FALSE,
                                     &dsv_handle);
  } else {
    command_list->OMSetRenderTargets(color_targets_.size(), rtv_handles, FALSE,
                                     nullptr);
  }
  if (!vertex_buffers_.empty()) {
    std::vector<D3D12_VERTEX_BUFFER_VIEW> vertex_buffer_views;
    for (size_t i = 0; i < vertex_buffers_.size(); ++i) {
      D3D12_VERTEX_BUFFER_VIEW vertex_buffer_view = {};
      vertex_buffer_view.BufferLocation =
          vertex_buffers_[i]->Buffer()->Handle()->GetGPUVirtualAddress();
      vertex_buffer_view.StrideInBytes = program_->InputBindingStride(i);
      vertex_buffer_view.SizeInBytes = vertex_buffers_[i]->Size();
      vertex_buffer_views.push_back(vertex_buffer_view);
    }
    command_list->IASetVertexBuffers(0, vertex_buffers_.size(),
                                     vertex_buffer_views.data());
  }

  D3D12_INDEX_BUFFER_VIEW index_buffer_view = {};
  index_buffer_view.BufferLocation =
      index_buffer_->Buffer()->Handle()->GetGPUVirtualAddress();
  index_buffer_view.Format = DXGI_FORMAT_R32_UINT;
  index_buffer_view.SizeInBytes = index_buffer_->Size();
  command_list->IASetIndexBuffer(&index_buffer_view);
  command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  command_list->DrawIndexedInstanced(index_count_, instance_count_,
                                     first_index_, vertex_offset_,
                                     first_instance_);
}

D3D12CmdPresent::D3D12CmdPresent(D3D12Window *window, D3D12Image *image)
    : image_(image), window_(window) {
}

void D3D12CmdPresent::CompileCommand(D3D12CommandContext *context,
                                     ID3D12GraphicsCommandList *command_list) {
  auto gpu_descriptor = context->WriteDescriptor(image_);

  context->RequireImageState(command_list, image_->Image()->Handle(),
                             D3D12_RESOURCE_STATE_GENERIC_READ);

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT,
      D3D12_RESOURCE_STATE_RENDER_TARGET);

  command_list->ResourceBarrier(1, &barrier);

  auto root_signature =
      context->Core()->BlitPipeline()->root_signature->Handle();
  auto pso = context->Core()->BlitPipeline()->GetPipelineState(
      window_->SwapChain()->BackBufferFormat());
  Extent2D extent;
  extent.width = window_->GetWidth();
  extent.height = window_->GetHeight();
  command_list->SetPipelineState(pso->Handle());
  command_list->SetGraphicsRootSignature(root_signature);
  command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  D3D12_RECT scissor_rect = {0, 0, LONG(extent.width), LONG(extent.height)};
  command_list->RSSetScissorRects(1, &scissor_rect);
  D3D12_VIEWPORT viewport = {
      0.0f, 0.0f, FLOAT(extent.width), FLOAT(extent.height), 0.0f, 1.0f};
  command_list->RSSetViewports(1, &viewport);
  const auto rtv = context->RTVHandle(window_->CurrentBackBuffer());
  command_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
  command_list->SetGraphicsRootDescriptorTable(0, gpu_descriptor);
  command_list->DrawInstanced(6, 1, 0, 0);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET,
      D3D12_RESOURCE_STATE_PRESENT);
  command_list->ResourceBarrier(1, &barrier);
}

}  // namespace grassland::graphics::backend
