#include "cao_di/graphics/backend/d3d12/d3d12_commands.h"

#include "cao_di/graphics/backend/d3d12/d3d12_command_context.h"
#include "cao_di/graphics/backend/d3d12/d3d12_image.h"
#include "cao_di/graphics/backend/d3d12/d3d12_program.h"
#include "cao_di/graphics/backend/d3d12/d3d12_sampler.h"
#include "cao_di/graphics/backend/d3d12/d3d12_window.h"

namespace CD::graphics::backend {

D3D12CmdBindProgram::D3D12CmdBindProgram(D3D12Program *program) : program_(program) {
}

void D3D12CmdBindProgram::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  command_list->SetGraphicsRootSignature(program_->RootSignature()->Handle());
  command_list->SetPipelineState(program_->PipelineState()->Handle());
}

D3D12CmdBindRayTracingProgram::D3D12CmdBindRayTracingProgram(D3D12RayTracingProgram *program) : program_(program) {
}

void D3D12CmdBindRayTracingProgram::CompileCommand(D3D12CommandContext *context,
                                                   ID3D12GraphicsCommandList *command_list) {
  command_list->SetComputeRootSignature(program_->RootSignature()->Handle());
  d3d12::ComPtr<ID3D12GraphicsCommandList4> command_list4;
  if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&command_list4)))) {
    command_list4->SetPipelineState1(program_->PipelineState()->Handle());
  }
}

D3D12CmdBindComputeProgram::D3D12CmdBindComputeProgram(D3D12ComputeProgram *program) : program_(program) {
}

void D3D12CmdBindComputeProgram::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  command_list->SetComputeRootSignature(program_->RootSignature()->Handle());
  command_list->SetPipelineState(program_->PipelineState().Get());
}

D3D12CmdBindVertexBuffers::D3D12CmdBindVertexBuffers(uint32_t first_binding,
                                                     const std::vector<D3D12Buffer *> &buffers,
                                                     const std::vector<uint64_t> &offsets,
                                                     D3D12Program *program)
    : first_binding_(first_binding), buffers_(buffers), program_(program) {
  offsets_.resize(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (i < offsets.size()) {
      offsets_[i] = offsets[i];
    } else {
      offsets_[i] = 0;
    }
  }
}

void D3D12CmdBindVertexBuffers::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  std::vector<D3D12_VERTEX_BUFFER_VIEW> vertex_buffer_views(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    vertex_buffer_views[i].BufferLocation = buffers_[i]->Buffer()->Handle()->GetGPUVirtualAddress() + offsets_[i];
    vertex_buffer_views[i].StrideInBytes = program_->InputBindingStride(first_binding_ + i);
    vertex_buffer_views[i].SizeInBytes = buffers_[i]->Size() - offsets_[i];

    context->RequireResourceState(command_list, buffers_[i]->Buffer()->Handle(),
                                  D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
  }
  command_list->IASetVertexBuffers(first_binding_, buffers_.size(), vertex_buffer_views.data());
}

D3D12CmdBindIndexBuffer::D3D12CmdBindIndexBuffer(D3D12Buffer *buffer, uint64_t offset)
    : buffer_(buffer), offset_(offset) {
}

void D3D12CmdBindIndexBuffer::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  D3D12_INDEX_BUFFER_VIEW index_buffer_view;
  index_buffer_view.BufferLocation = buffer_->Buffer()->Handle()->GetGPUVirtualAddress() + offset_;
  index_buffer_view.SizeInBytes = buffer_->Size() - offset_;
  index_buffer_view.Format = DXGI_FORMAT_R32_UINT;

  context->RequireResourceState(command_list, buffer_->Buffer()->Handle(), D3D12_RESOURCE_STATE_INDEX_BUFFER);
  command_list->IASetIndexBuffer(&index_buffer_view);
}

D3D12CmdBindResourceBuffers::D3D12CmdBindResourceBuffers(int slot,
                                                         const std::vector<D3D12BufferRange> &buffers,
                                                         D3D12ProgramBase *program,
                                                         BindPoint bind_point)
    : slot_(slot), buffers_(buffers), program_(program), bind_point_(bind_point) {
}

void D3D12CmdBindResourceBuffers::CompileCommand(D3D12CommandContext *context,
                                                 ID3D12GraphicsCommandList *command_list) {
  auto descriptor_range = program_->DescriptorRange(slot_);
  CD3DX12_GPU_DESCRIPTOR_HANDLE first_descriptor;
  first_descriptor.ptr = 0;
  switch (descriptor_range->RangeType) {
    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
      for (size_t i = 0; i < buffers_.size(); ++i) {
        auto desc = context->WriteCBVDescriptor(buffers_[i]);
        if (i == 0) {
          first_descriptor = desc;
        }
        context->RequireResourceState(command_list, buffers_[i].buffer->Buffer()->Handle(),
                                      D3D12_RESOURCE_STATE_GENERIC_READ);
      }
      break;
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
      for (size_t i = 0; i < buffers_.size(); ++i) {
        auto desc = context->WriteSRVDescriptor(buffers_[i]);
        if (i == 0) {
          first_descriptor = desc;
        }
        context->RequireResourceState(command_list, buffers_[i].buffer->Buffer()->Handle(),
                                      D3D12_RESOURCE_STATE_GENERIC_READ);
      }
      break;
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
      for (size_t i = 0; i < buffers_.size(); ++i) {
        auto desc = context->WriteUAVDescriptor(buffers_[i]);
        if (i == 0) {
          first_descriptor = desc;
        }
        context->RequireResourceState(command_list, buffers_[i].buffer->Buffer()->Handle(),
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
      break;
  }
  switch (bind_point_) {
    case BIND_POINT_GRAPHICS:
      command_list->SetGraphicsRootDescriptorTable(slot_, first_descriptor);
      break;
    case BIND_POINT_COMPUTE:
    case BIND_POINT_RAYTRACING:
      command_list->SetComputeRootDescriptorTable(slot_, first_descriptor);
      break;
  }
}

D3D12CmdBindResourceImages::D3D12CmdBindResourceImages(int slot,
                                                       const std::vector<D3D12Image *> &images,
                                                       D3D12ProgramBase *program,
                                                       BindPoint bind_point)
    : slot_(slot), images_(images), program_(program), bind_point_(bind_point) {
}

void D3D12CmdBindResourceImages::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  auto descriptor_range = program_->DescriptorRange(slot_);
  CD3DX12_GPU_DESCRIPTOR_HANDLE first_descriptor;
  first_descriptor.ptr = 0;
  switch (descriptor_range->RangeType) {
    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
      for (size_t i = 0; i < images_.size(); ++i) {
        auto desc = context->WriteSRVDescriptor(images_[i]);
        if (i == 0) {
          first_descriptor = desc;
        }
        context->RequireResourceState(command_list, images_[i]->Image()->Handle(), D3D12_RESOURCE_STATE_GENERIC_READ);
      }
      break;
    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
      for (size_t i = 0; i < images_.size(); ++i) {
        auto desc = context->WriteUAVDescriptor(images_[i]);
        if (i == 0) {
          first_descriptor = desc;
        }
        context->RequireResourceState(command_list, images_[i]->Image()->Handle(),
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
      break;
  }
  switch (bind_point_) {
    case BIND_POINT_GRAPHICS:
      command_list->SetGraphicsRootDescriptorTable(slot_, first_descriptor);
      break;
    case BIND_POINT_COMPUTE:
    case BIND_POINT_RAYTRACING:
      command_list->SetComputeRootDescriptorTable(slot_, first_descriptor);
      break;
  }
}

D3D12CmdBindResourceSamplers::D3D12CmdBindResourceSamplers(int slot,
                                                           const std::vector<D3D12Sampler *> &samplers,
                                                           D3D12ProgramBase *program,
                                                           BindPoint bind_point)
    : slot_(slot), samplers_(samplers), program_(program), bind_point_(bind_point) {
}

void D3D12CmdBindResourceSamplers::CompileCommand(D3D12CommandContext *context,
                                                  ID3D12GraphicsCommandList *command_list) {
  CD3DX12_GPU_DESCRIPTOR_HANDLE first_descriptor;
  first_descriptor.ptr = 0;
  for (size_t i = 0; i < samplers_.size(); ++i) {
    auto desc = context->WriteSamplerDescriptor(samplers_[i]->SamplerDesc());
    if (i == 0) {
      first_descriptor = desc;
    }
  }
  switch (bind_point_) {
    case BIND_POINT_GRAPHICS:
      command_list->SetGraphicsRootDescriptorTable(slot_, first_descriptor);
      break;
    case BIND_POINT_COMPUTE:
    case BIND_POINT_RAYTRACING:
      command_list->SetComputeRootDescriptorTable(slot_, first_descriptor);
      break;
  }
}

D3D12CmdBindResourceAccelerationStructure::D3D12CmdBindResourceAccelerationStructure(
    int slot,
    D3D12AccelerationStructure *acceleration_structure,
    D3D12ProgramBase *program,
    BindPoint bind_point)
    : slot_(slot), acceleration_structure_(acceleration_structure), program_(program), bind_point_(bind_point) {
}

void D3D12CmdBindResourceAccelerationStructure::CompileCommand(D3D12CommandContext *context,
                                                               ID3D12GraphicsCommandList *command_list) {
  CD3DX12_GPU_DESCRIPTOR_HANDLE first_descriptor;
  first_descriptor = context->WriteSRVDescriptor(acceleration_structure_);
  switch (bind_point_) {
    case BIND_POINT_GRAPHICS:
      command_list->SetGraphicsRootDescriptorTable(slot_, first_descriptor);
      break;
    case BIND_POINT_COMPUTE:
    case BIND_POINT_RAYTRACING:
      command_list->SetComputeRootDescriptorTable(slot_, first_descriptor);
      break;
  }
}

D3D12CmdBeginRendering::D3D12CmdBeginRendering(const std::vector<D3D12Image *> &color_targets, D3D12Image *depth_target)
    : color_targets_(color_targets), depth_target_(depth_target) {
}

void D3D12CmdBeginRendering::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  CD3DX12_CPU_DESCRIPTOR_HANDLE rtv_handles[8]{};
  CD3DX12_CPU_DESCRIPTOR_HANDLE dsv_handle{};
  for (size_t i = 0; i < color_targets_.size(); ++i) {
    context->RequireResourceState(command_list, color_targets_[i]->Image()->Handle(),
                                  D3D12_RESOURCE_STATE_RENDER_TARGET);
    rtv_handles[i] = context->RTVHandle(color_targets_[i]->Image()->Handle());
  }
  if (depth_target_) {
    context->RequireResourceState(command_list, depth_target_->Image()->Handle(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
    dsv_handle = context->DSVHandle(depth_target_->Image()->Handle());
    command_list->OMSetRenderTargets(color_targets_.size(), rtv_handles, FALSE, &dsv_handle);
  } else {
    command_list->OMSetRenderTargets(color_targets_.size(), rtv_handles, FALSE, nullptr);
  }
}

D3D12CmdClearImage::D3D12CmdClearImage(D3D12Image *image, const ClearValue &clear_value)
    : image_(image), clear_value_(clear_value) {
}

void D3D12CmdClearImage::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  if (IsDepthFormat(image_->Format())) {
    context->RequireResourceState(command_list, image_->Image()->Handle(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.DepthStencil.Depth = clear_value_.depth.depth;
    clear_value.DepthStencil.Stencil = 0;
    command_list->ClearDepthStencilView(context->DSVHandle(image_->Image()->Handle()), D3D12_CLEAR_FLAG_DEPTH,
                                        clear_value.DepthStencil.Depth, clear_value.DepthStencil.Stencil, 0, nullptr);
  } else {
    context->RequireResourceState(command_list, image_->Image()->Handle(), D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CLEAR_VALUE clear_value;
    clear_value.Format = ImageFormatToDXGIFormat(image_->Format());
    clear_value.Color[0] = clear_value_.color.r;
    clear_value.Color[1] = clear_value_.color.g;
    clear_value.Color[2] = clear_value_.color.b;
    clear_value.Color[3] = clear_value_.color.a;
    command_list->ClearRenderTargetView(context->RTVHandle(image_->Image()->Handle()), clear_value.Color, 0, nullptr);
  }
}

D3D12CmdSetViewport::D3D12CmdSetViewport(const Viewport &viewport) : viewport_(viewport) {
}

void D3D12CmdSetViewport::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  D3D12_VIEWPORT viewport;
  viewport.TopLeftX = viewport_.x;
  viewport.TopLeftY = viewport_.y;
  viewport.Width = viewport_.width;
  viewport.Height = viewport_.height;
  viewport.MinDepth = viewport_.min_depth;
  viewport.MaxDepth = viewport_.max_depth;
  command_list->RSSetViewports(1, &viewport);
}

D3D12CmdSetScissor::D3D12CmdSetScissor(const Scissor &scissor) : scissor_(scissor) {
}

void D3D12CmdSetScissor::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  D3D12_RECT scissor_rect;
  scissor_rect.left = scissor_.offset.x;
  scissor_rect.top = scissor_.offset.y;
  scissor_rect.right = scissor_.offset.x + scissor_.extent.width;
  scissor_rect.bottom = scissor_.offset.y + scissor_.extent.height;
  command_list->RSSetScissorRects(1, &scissor_rect);
}

D3D12CmdSetPrimitiveTopology::D3D12CmdSetPrimitiveTopology(PrimitiveTopology topology) : topology_(topology) {
}

void D3D12CmdSetPrimitiveTopology::CompileCommand(D3D12CommandContext *context,
                                                  ID3D12GraphicsCommandList *command_list) {
  command_list->IASetPrimitiveTopology(PrimitiveTopologyToD3D12PrimitiveTopology(topology_));
}

D3D12CmdDraw::D3D12CmdDraw(uint32_t index_count,
                           uint32_t instance_count,
                           int32_t vertex_offset,
                           uint32_t first_instance)
    : index_count_(index_count),
      instance_count_(instance_count),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
}

void D3D12CmdDraw::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  command_list->DrawInstanced(index_count_, instance_count_, vertex_offset_, first_instance_);
}

D3D12CmdDrawIndexed::D3D12CmdDrawIndexed(uint32_t index_count,
                                         uint32_t instance_count,
                                         uint32_t first_index,
                                         int32_t vertex_offset,
                                         uint32_t first_instance)
    : index_count_(index_count),
      instance_count_(instance_count),
      first_index_(first_index),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
}

void D3D12CmdDrawIndexed::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  command_list->DrawIndexedInstanced(index_count_, instance_count_, first_index_, vertex_offset_, first_instance_);
}

D3D12CmdPresent::D3D12CmdPresent(D3D12Window *window, D3D12Image *image) : image_(image), window_(window) {
}

void D3D12CmdPresent::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  auto gpu_descriptor = context->WriteSRVDescriptor(image_);

  context->RequireResourceState(command_list, image_->Image()->Handle(), D3D12_RESOURCE_STATE_GENERIC_READ);

  CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

  command_list->ResourceBarrier(1, &barrier);

  auto root_signature = context->Core()->BlitPipeline()->root_signature->Handle();
  auto pso = context->Core()->BlitPipeline()->GetPipelineState(window_->SwapChain()->BackBufferFormat());
  Extent2D extent;
  extent.width = window_->SwapChain()->Width();
  extent.height = window_->SwapChain()->Height();
  command_list->SetPipelineState(pso->Handle());
  command_list->SetGraphicsRootSignature(root_signature);
  command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  D3D12_RECT scissor_rect = {0, 0, LONG(extent.width), LONG(extent.height)};
  command_list->RSSetScissorRects(1, &scissor_rect);
  D3D12_VIEWPORT viewport = {0.0f, 0.0f, FLOAT(extent.width), FLOAT(extent.height), 0.0f, 1.0f};
  command_list->RSSetViewports(1, &viewport);
  const auto rtv = context->RTVHandle(window_->CurrentBackBuffer());
  command_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
  command_list->SetGraphicsRootDescriptorTable(0, gpu_descriptor);
  command_list->DrawInstanced(6, 1, 0, 0);

  auto &imgui_assets = window_->ImGuiAssets();
  if (imgui_assets.context && imgui_assets.draw_command) {
    imgui_assets.draw_command = false;
    auto binding_heap = imgui_assets.srv_heap->Handle();
    command_list->SetDescriptorHeaps(1, &binding_heap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), command_list);
  }

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(window_->CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                 D3D12_RESOURCE_STATE_PRESENT);
  command_list->ResourceBarrier(1, &barrier);
}

D3D12CmdDispatchRays::D3D12CmdDispatchRays(D3D12RayTracingProgram *program,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t depth)
    : program_(program), width_(width), height_(height), depth_(depth) {
}

void D3D12CmdDispatchRays::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  d3d12::ComPtr<ID3D12GraphicsCommandList4> dxr_command_list;
  if (SUCCEEDED(command_list->QueryInterface(IID_PPV_ARGS(&dxr_command_list)))) {
    auto shader_table = program_->ShaderTable();
    UINT shader_record_size =
        d3d12::SizeAlignTo(D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);
    D3D12_DISPATCH_RAYS_DESC dispatch_desc = {};
    dispatch_desc.HitGroupTable.StartAddress = shader_table->GetHitGroupDeviceAddress();
    dispatch_desc.HitGroupTable.SizeInBytes = shader_record_size * shader_table->HitGroupShaderCount();
    dispatch_desc.HitGroupTable.StrideInBytes = shader_record_size;
    dispatch_desc.MissShaderTable.StartAddress = shader_table->GetMissDeviceAddress();
    dispatch_desc.MissShaderTable.SizeInBytes = shader_record_size * shader_table->MissShaderCount();
    dispatch_desc.MissShaderTable.StrideInBytes = shader_record_size;
    dispatch_desc.RayGenerationShaderRecord.StartAddress = shader_table->GetRayGenDeviceAddress();
    dispatch_desc.RayGenerationShaderRecord.SizeInBytes = shader_record_size;
    dispatch_desc.CallableShaderTable.StartAddress = shader_table->GetCallableDeviceAddress();
    dispatch_desc.CallableShaderTable.SizeInBytes = shader_record_size * shader_table->CallableShaderCount();
    dispatch_desc.CallableShaderTable.StrideInBytes = shader_record_size;
    dispatch_desc.Width = width_;
    dispatch_desc.Height = height_;
    dispatch_desc.Depth = depth_;
    dxr_command_list->DispatchRays(&dispatch_desc);

    D3D12_RESOURCE_BARRIER uav_barrier = {};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uav_barrier.UAV.pResource = nullptr;  // This barrier is a placeholder, it will be set by the program
    command_list->ResourceBarrier(1, &uav_barrier);
  }
}

D3D12CmdDispatch::D3D12CmdDispatch(uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z)
    : group_count_x_(group_count_x), group_count_y_(group_count_y), group_count_z_(group_count_z) {
}

void D3D12CmdDispatch::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  command_list->Dispatch(group_count_x_, group_count_y_, group_count_z_);

  D3D12_RESOURCE_BARRIER uav_barrier = {};
  uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  uav_barrier.UAV.pResource = nullptr;  // This barrier is a placeholder, it will be set by the program
  command_list->ResourceBarrier(1, &uav_barrier);
}

D3D12CmdCopyBuffer::D3D12CmdCopyBuffer(D3D12Buffer *dst_buffer,
                                       D3D12Buffer *src_buffer,
                                       uint64_t size,
                                       uint64_t dst_offset,
                                       uint64_t src_offset)
    : dst_buffer_(dst_buffer), src_buffer_(src_buffer), size_(size), dst_offset_(dst_offset), src_offset_(src_offset) {
}

void D3D12CmdCopyBuffer::CompileCommand(D3D12CommandContext *context, ID3D12GraphicsCommandList *command_list) {
  context->RequireResourceState(command_list, dst_buffer_->Buffer()->Handle(), D3D12_RESOURCE_STATE_COPY_DEST);
  context->RequireResourceState(command_list, src_buffer_->Buffer()->Handle(), D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_list->CopyBufferRegion(dst_buffer_->Buffer()->Handle(), dst_offset_, src_buffer_->Buffer()->Handle(),
                                 src_offset_, size_);
}

}  // namespace CD::graphics::backend
