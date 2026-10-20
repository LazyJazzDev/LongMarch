#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12CommandContext::D3D12CommandContext(D3D12Device *device) : CommandContext(device), device_(device) {
}

D3D12CommandContext::~D3D12CommandContext() {
  if (reusable_) {
    if (!closed_)
      reusable_ = SUCCEEDED(list_->Close());
    if (reusable_) {
      device_->RecycleCommandResources(std::move(allocator_), std::move(list_), event_);
      return;
    }
  }
  if (event_)
    CloseHandle(event_);
}

D3D12CommandContext *D3D12CommandContext::Create(D3D12Device *device) {
  try {
    auto object = std::unique_ptr<D3D12CommandContext>(new D3D12CommandContext(device));
    object->reusable_ = object->device_->AcquireCommandResources(object->allocator_, object->list_, object->event_);
    if (!object->reusable_)
      return nullptr;
    return object.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "D3D12CommandContext creation ran out of memory.");
    return nullptr;
  }
}

bool D3D12CommandContext::TransitionResource(ID3D12Resource *resource,
                                             D3D12_RESOURCE_STATES state,
                                             D3D12_RESOURCE_STATES default_state) {
  // Upload heaps stay in GENERIC_READ; compatible reads need no transition.
  if (default_state == D3D12_RESOURCE_STATE_GENERIC_READ) {
#if !defined(NDEBUG)
    if ((state & default_state) != state) {
      recording_failed_ = true;
      ReportError(-1, "Upload buffers only support GPU read usages.");
      return false;
    }
#endif
    return true;
  }
  try {
    auto &current =
        resource_states_.try_emplace(resource, ResourceState{default_state, default_state}).first->second.current;
    if (current != state) {
      D3D12_RESOURCE_BARRIER barrier{};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition.pResource = resource;
      barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      barrier.Transition.StateBefore = current;
      barrier.Transition.StateAfter = state;
      list_->ResourceBarrier(1, &barrier);
      current = state;
    }
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    ReportError(-1, "Resource state tracking ran out of memory.");
    return false;
  }
}

bool D3D12CommandContext::RenderTarget(D3D12Image *image, D3D12_CPU_DESCRIPTOR_HANDLE &handle) {
  try {
    auto existing = rtv_leases_.find(image);
    if (existing != rtv_leases_.end()) {
      handle = existing->second;
      return true;
    }
    auto lease = device_->RTVPool().Acquire(1);
    if (!lease || !RetainResource(lease.Get())) {
      recording_failed_ = true;
      return false;
    }
    handle = lease->CPU();
    device_->NativeDevice()->CreateRenderTargetView(image->NativeImage(), nullptr, handle);
    rtv_leases_.emplace(image, handle);
    return true;
  } catch (const std::bad_alloc &) {
    rtv_leases_.erase(image);
    recording_failed_ = true;
    return false;
  }
}

bool D3D12CommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  const auto native = dynamic_cast<D3D12Image *>(image);
  if (!native || !native->IsValid() || native->DeviceIdentity() != device_) {
    recording_failed_ = true;
    ReportError(-1, "CmdClearColor requires a valid image from this device.");
    return false;
  }
#else
  const auto native = static_cast<D3D12Image *>(image);
#endif
  if (!RetainResource(native))
    return false;
  if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, native->DefaultState()))
    return false;
  D3D12_CPU_DESCRIPTOR_HANDLE handle{};
  if (!RenderTarget(native, handle))
    return false;
  list_->ClearRenderTargetView(handle, color.data(), 0, nullptr);
  return true;
}

bool D3D12CommandContext::CmdPresent(rhi::IWindow *window) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  const auto target = dynamic_cast<D3D12Window *>(window);
  if (!target || target->DeviceIdentity() != device_ || !glfw::IsWindowThread()) {
    recording_failed_ = true;
    ReportError(-1, "CmdPresent requires a window from this device on the window thread.");
    return false;
  }
  for (const auto &present : presents_) {
    if (present.first == target) {
      recording_failed_ = true;
      ReportError(-1, "A context can present each window only once.");
      return false;
    }
  }
#else
  const auto target = static_cast<D3D12Window *>(window);
#endif
  auto frame = ObjectPtr<D3D12SwapChainImage>::Adopt(static_cast<D3D12SwapChainImage *>(target->GetNextImage()));
  if (!frame)
    return false;
  if (!RetainResource(frame.Get()))
    return false;
  try {
    presents_.emplace_back(target, frame.Get());
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  return TransitionResource(frame->NativeImage(), D3D12_RESOURCE_STATE_PRESENT, frame->DefaultState());
}

bool D3D12CommandContext::EndRecording() {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
#endif
  // Restore each resource to its own creation-time default, including non-presented images.
  for (auto &entry : resource_states_) {
    if (!TransitionResource(entry.first, entry.second.default_state, entry.second.default_state))
      return false;
  }
  closed_ = Check(list_->Close(), "CommandList::Close");
  reusable_ = closed_;
  return closed_;
}

CommandContext::SubmitResult D3D12CommandContext::SubmitNative() {
#if !defined(NDEBUG)
  if (!presents_.empty() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  for (const auto &present : presents_) {
    if (!present.second->IsValid())
      return SUBMIT_RESULT_REJECTED;
  }
#endif
  {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    submission_failed_ = true;
    ID3D12CommandList *lists[] = {list_.Get()};
    device_->NativeQueue()->ExecuteCommandLists(1, lists);
    MarkSubmitted();
    bool presented = true;
    for (const auto &present : presents_) {
      if (!Check(present.second->Present(), "Swapchain::Present"))
        presented = false;
    }
    value_ = device_->NextSerial();
    submission_failed_ = !Check(device_->NativeQueue()->Signal(device_->Fence(), value_), "Queue::Signal");
    if (!presented)
      submission_failed_ = true;
  }
  if (submission_failed_)
    return SUBMIT_RESULT_FAILED_IN_FLIGHT;
  for (const auto &present : presents_) {
    if (!present.first->UpdateNextImage()) {
      ReportError(-1, "Failed to prepare the next swapchain image after submission.");
      std::abort();
    }
  }
  return SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus D3D12CommandContext::WaitForProgress() {
  if (submission_failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    if (device_->WaitIdle() || FAILED(device_->NativeDevice()->GetDeviceRemovedReason()))
      return SUBMISSION_STATUS_FAILED;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return SUBMISSION_STATUS_PENDING;
  }
  if (device_->Wait(value_, event_))
    return SUBMISSION_STATUS_COMPLETED;
  // A failed host wait does not prove that GPU resource use has ended.
  submission_failed_ = true;
  return SUBMISSION_STATUS_PENDING;
}

bool D3D12CommandContext::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  auto native = dynamic_cast<D3D12Buffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<D3D12Buffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;

  if (!data || !size || offset > native->GetDesc().size || size > native->GetDesc().size - offset) {
    recording_failed_ = true;
    return false;
  }
  auto upload = device_->UploadPool().Acquire(size);
  if (!upload) {
    recording_failed_ = true;
    return false;
  }
  if (!upload->Write(data, size)) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(upload.Get()))
    return false;
  if (!TransitionResource(native->NativeBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, native->DefaultState()))
    return false;
  list_->CopyBufferRegion(native->NativeBuffer(), offset, upload->NativeBuffer(), 0, size);
  return true;
}

bool D3D12CommandContext::CmdCopyBuffer(IBuffer *source,
                                        uint64_t source_offset,
                                        IBuffer *destination,
                                        uint64_t destination_offset,
                                        uint64_t size) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  auto src = dynamic_cast<D3D12Buffer *>(source);
  auto dst = dynamic_cast<D3D12Buffer *>(destination);
  if (!src || !dst || src->Owner() != device_ || dst->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto src = static_cast<D3D12Buffer *>(source);
  auto dst = static_cast<D3D12Buffer *>(destination);
#endif
  if (src == dst || !size || source_offset > src->GetDesc().size || size > src->GetDesc().size - source_offset ||
      destination_offset > dst->GetDesc().size || size > dst->GetDesc().size - destination_offset) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(src))
    return false;
  if (!RetainResource(dst))
    return false;
  if (!TransitionResource(src->NativeBuffer(), D3D12_RESOURCE_STATE_COPY_SOURCE, src->DefaultState()))
    return false;
  if (!TransitionResource(dst->NativeBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, dst->DefaultState()))
    return false;
  list_->CopyBufferRegion(dst->NativeBuffer(), destination_offset, src->NativeBuffer(), source_offset, size);
  return true;
}

bool D3D12CommandContext::CmdBindProgram(IProgram *program) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<D3D12Program *>(program);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<D3D12Program *>(program);
#endif
  if (!RetainResource(native))
    return false;
  if (!native->IsFinalized()) {
    recording_failed_ = true;
    return false;
  }
  try {
    vertex_buffers_.assign(native->GetDesc().vertex_bindings.size(), nullptr);
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  try {
    groups_.assign(native->GetDesc().pipeline_layout.groups.size(), nullptr);
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  groups_dirty_ = true;
  program_ = native;
  list_->OMSetStencilRef(native->GetDesc().depth_stencil.stencil_reference);
  list_->SetPipelineState(native->NativePipeline());
  list_->SetGraphicsRootSignature(native->NativeRootSignature());
  list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  return true;
}

bool D3D12CommandContext::CmdBeginRendering(const RenderingInfo &info) {
  if (sealed_ || render_target_ || !PrepareRendering(info)) {
    recording_failed_ = true;
    return false;
  }
  std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 8> colors{};
  auto count = static_cast<UINT>(rendering_.color_attachments.size());
  auto rtvs = count ? device_->RTVPool().Acquire(count) : ObjectPtr<D3D12DescriptorPool::Lease>{};
  if (count && (!rtvs || !RetainResource(rtvs.Get()))) {
    recording_failed_ = true;
    return false;
  }
  for (UINT i = 0; i < count; ++i) {
    const auto &attachment = rendering_.color_attachments[i];
    auto image = static_cast<D3D12Image *>(attachment.color_attachment);
    if (!TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState()))
      return false;
    colors[i] = rtvs->CPU(i);
    device_->NativeDevice()->CreateRenderTargetView(image->NativeImage(), nullptr, colors[i]);
    if (attachment.load_op == ATTACHMENT_LOAD_OP_CLEAR)
      list_->ClearRenderTargetView(colors[i], attachment.clear_color.data(), 0, nullptr);
    else if (attachment.load_op == ATTACHMENT_LOAD_OP_DISCARD)
      list_->DiscardResource(image->NativeImage(), nullptr);
  }
  auto depth = static_cast<D3D12Image *>(rendering_.depth_stencil.image);
  D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
  if (depth) {
    auto lease = device_->DSVPool().Acquire(1);
    if (!lease || !RetainResource(lease.Get())) {
      recording_failed_ = true;
      return false;
    }
    dsv = lease->CPU();
    if (!TransitionResource(depth->NativeImage(), D3D12_RESOURCE_STATE_DEPTH_WRITE, depth->DefaultState()))
      return false;
    device_->NativeDevice()->CreateDepthStencilView(depth->NativeImage(), nullptr, dsv);
    const auto &ds = rendering_.depth_stencil;
    UINT flags = ds.depth_load_op == ATTACHMENT_LOAD_OP_CLEAR ? D3D12_CLEAR_FLAG_DEPTH : 0;
    if (HasStencil(depth->GetDesc().format) && ds.stencil_load_op == ATTACHMENT_LOAD_OP_CLEAR)
      flags |= D3D12_CLEAR_FLAG_STENCIL;
    if (flags)
      list_->ClearDepthStencilView(dsv, static_cast<D3D12_CLEAR_FLAGS>(flags), ds.clear_depth,
                                   static_cast<UINT8>(ds.clear_stencil), 0, nullptr);
  }
  list_->OMSetRenderTargets(count, colors.data(), FALSE, depth ? &dsv : nullptr);
  render_target_ = count ? static_cast<D3D12Image *>(rendering_.color_attachments[0].color_attachment) : depth;
  return true;
}

bool D3D12CommandContext::CmdEndRendering() {
  if (sealed_ || !render_target_ || recording_failed_) {
    recording_failed_ = true;
    return false;
  }
  for (const auto &color : rendering_.color_attachments)
    if (color.store_op == ATTACHMENT_STORE_OP_DISCARD)
      list_->DiscardResource(static_cast<D3D12Image *>(color.color_attachment)->NativeImage(), nullptr);
  list_->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
  render_target_ = nullptr;
  return true;
}

bool D3D12CommandContext::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<D3D12Buffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<D3D12Buffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if (!program_ || binding >= program_->GetDesc().vertex_bindings.size() || offset >= native->GetDesc().size) {
    recording_failed_ = true;
    return false;
  }
  if (native->GetDesc().size - offset > UINT_MAX) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionResource(native->NativeBuffer(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                          native->DefaultState()))
    return false;
  D3D12_VERTEX_BUFFER_VIEW view{native->NativeBuffer()->GetGPUVirtualAddress() + offset,
                                static_cast<UINT>(native->GetDesc().size - offset),
                                program_->GetDesc().vertex_bindings[binding].stride};
  list_->IASetVertexBuffers(binding, 1, &view);
  vertex_buffers_[binding] = native;
  return true;
}

bool D3D12CommandContext::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<D3D12Buffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<D3D12Buffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if (format != INDEX_FORMAT_UINT16 && format != INDEX_FORMAT_UINT32) {
    recording_failed_ = true;
    return false;
  }
  auto alignment = format == INDEX_FORMAT_UINT16 ? 2u : 4u;
  if (offset >= native->GetDesc().size || offset % alignment) {
    recording_failed_ = true;
    return false;
  }
  if (native->GetDesc().size - offset > UINT_MAX) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionResource(native->NativeBuffer(), D3D12_RESOURCE_STATE_INDEX_BUFFER, native->DefaultState()))
    return false;
  D3D12_INDEX_BUFFER_VIEW view{native->NativeBuffer()->GetGPUVirtualAddress() + offset,
                               static_cast<UINT>(native->GetDesc().size - offset),
                               format == INDEX_FORMAT_UINT16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT};
  list_->IASetIndexBuffer(&view);
  index_buffer_ = native;
  index_bound_ = true;
  return true;
}

bool D3D12CommandContext::CmdSetViewport(const Viewport &viewport) {
#if !defined(NDEBUG)
  if (sealed_ || !viewport.width || !viewport.height) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  D3D12_VIEWPORT native{viewport.x,      viewport.y,         viewport.width,
                        viewport.height, viewport.min_depth, viewport.max_depth};
  list_->RSSetViewports(1, &native);
  viewport_set_ = true;
  return true;
}

bool D3D12CommandContext::CmdSetScissor(const Scissor &scissor) {
#if !defined(NDEBUG)
  if (sealed_ || !scissor.width || !scissor.height) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  D3D12_RECT native{scissor.x, scissor.y, scissor.x + static_cast<LONG>(scissor.width),
                    scissor.y + static_cast<LONG>(scissor.height)};
  list_->RSSetScissorRects(1, &native);
  scissor_set_ = true;
  return true;
}

bool D3D12CommandContext::CmdDraw(uint32_t vertex_count,
                                  uint32_t instance_count,
                                  uint32_t first_vertex,
                                  uint32_t first_instance) {
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !render_target_ || !viewport_set_ || !scissor_set_) {
    recording_failed_ = true;
    return false;
  }
  if (!RenderingMatches(program_)) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!PrepareDrawBuffers(false))
    return false;
  if (!PrepareGroups())
    return false;
  list_->DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
  return true;
}

bool D3D12CommandContext::CmdDrawIndexed(uint32_t index_count,
                                         uint32_t instance_count,
                                         uint32_t first_index,
                                         int32_t vertex_offset,
                                         uint32_t first_instance) {
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !render_target_ || !viewport_set_ || !scissor_set_ || !index_bound_) {
    recording_failed_ = true;
    return false;
  }
  if (!RenderingMatches(program_)) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!PrepareDrawBuffers(true))
    return false;
  if (!PrepareGroups())
    return false;
  list_->DrawIndexedInstanced(index_count, instance_count, first_index, vertex_offset, first_instance);
  return true;
}

bool D3D12CommandContext::PrepareDrawBuffers(bool indexed) {
  for (auto buffer : vertex_buffers_) {
#if !defined(NDEBUG)
    if (!buffer) {
      recording_failed_ = true;
      return false;
    }
#endif
    auto state = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    if (indexed && buffer == index_buffer_)
      state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
    if (!TransitionResource(buffer->NativeBuffer(), state, buffer->DefaultState()))
      return false;
  }
  if (indexed) {
    auto state = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    if (std::find(vertex_buffers_.begin(), vertex_buffers_.end(), index_buffer_) != vertex_buffers_.end())
      state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    if (!TransitionResource(index_buffer_->NativeBuffer(), state, index_buffer_->DefaultState()))
      return false;
  }
  for (const auto &color : rendering_.color_attachments) {
    auto image = static_cast<D3D12Image *>(color.color_attachment);
    if (!TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState()))
      return false;
  }
  return true;
}

bool D3D12CommandContext::CmdBindGroup(uint32_t index, IBindingGroup *group) {
  auto native = dynamic_cast<BindingGroup *>(group);
  if (sealed_ || !program_ || !native || native->Owner() != device_ || index >= groups_.size() ||
      !SameBindingLayout(program_->GetDesc().pipeline_layout.groups[index], group->GetLayout()) ||
      !RetainResource(group)) {
    recording_failed_ = true;
    return false;
  }
  groups_[index] = group;
  groups_dirty_ = true;
  return true;
}

bool D3D12CommandContext::PrepareGroups() {
  try {
    const auto mappings = MapBindings(program_->GetDesc().pipeline_layout, BACKEND_API_D3D12);
    for (size_t i = 0; i < groups_.size(); ++i)
      if (!groups_[i] && !program_->GetDesc().pipeline_layout.groups[i].bindings.empty()) {
        recording_failed_ = true;
        return false;
      }
    uint32_t resource_count = 0, sampler_count = 0;
    for (auto m : mappings)
      (m.desc.type == BINDING_SAMPLER ? sampler_count : resource_count) += m.desc.count;
    auto resources = groups_dirty_ && resource_count ? device_->ResourceDescriptors().Acquire(resource_count)
                                                     : ObjectPtr<D3D12DescriptorPool::Lease>{};
    auto samplers = groups_dirty_ && sampler_count ? device_->SamplerDescriptors().Acquire(sampler_count)
                                                   : ObjectPtr<D3D12DescriptorPool::Lease>{};
    if (groups_dirty_ && ((resource_count && (!resources || !RetainResource(resources.Get()))) ||
                          (sampler_count && (!samplers || !RetainResource(samplers.Get()))))) {
      recording_failed_ = true;
      return false;
    }
    uint32_t r = 0, s = 0;
    std::vector<std::array<uint32_t, 2>> starts(groups_.size());
    for (uint32_t g = 0; g < groups_.size(); ++g) {
      starts[g] = {r, s};
      if (!groups_[g])
        continue;
      for (auto m : mappings)
        if (m.group == g)
          for (auto e : groups_[g]->GetEntries())
            if (e.binding == m.desc.binding) {
              auto type = m.desc.type;
              D3D12_CPU_DESCRIPTOR_HANDLE handle{};
              if (groups_dirty_)
                handle = type == BINDING_SAMPLER ? samplers->CPU(s++) : resources->CPU(r++);
              if (type <= BINDING_RW_STORAGE_BUFFER) {
                auto buffer = static_cast<D3D12Buffer *>(e.buffer);
                if (!TransitionResource(buffer->NativeBuffer(),
                                        type == BINDING_RW_STORAGE_BUFFER ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                                          : D3D12_RESOURCE_STATE_GENERIC_READ,
                                        buffer->DefaultState()))
                  return false;
                if (type == BINDING_UNIFORM_BUFFER) {
                  auto size = (e.size + 255) & ~uint64_t(255);
                  if (e.offset % 256 || size > 65536 || size > buffer->GetDesc().size - e.offset) {
                    recording_failed_ = true;
                    return false;
                  }
                  D3D12_CONSTANT_BUFFER_VIEW_DESC desc{buffer->NativeBuffer()->GetGPUVirtualAddress() + e.offset,
                                                       UINT(size)};
                  if (groups_dirty_)
                    device_->NativeDevice()->CreateConstantBufferView(&desc, handle);
                } else if (type == BINDING_STORAGE_BUFFER) {
                  D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                  desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                  desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                  desc.Buffer.FirstElement = e.offset / e.stride;
                  desc.Buffer.NumElements = UINT(e.size / e.stride);
                  desc.Buffer.StructureByteStride = e.stride;
                  if (groups_dirty_)
                    device_->NativeDevice()->CreateShaderResourceView(buffer->NativeBuffer(), &desc, handle);
                } else {
                  D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
                  desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                  desc.Buffer.FirstElement = e.offset / e.stride;
                  desc.Buffer.NumElements = UINT(e.size / e.stride);
                  desc.Buffer.StructureByteStride = e.stride;
                  if (groups_dirty_)
                    device_->NativeDevice()->CreateUnorderedAccessView(buffer->NativeBuffer(), nullptr, &desc, handle);
                  D3D12_RESOURCE_BARRIER barrier{};
                  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                  barrier.UAV.pResource = buffer->NativeBuffer();
                  list_->ResourceBarrier(1, &barrier);
                }
              } else if (type == BINDING_TEXTURE2D || type == BINDING_RW_TEXTURE2D) {
                auto image = static_cast<D3D12Image *>(e.image);
                for (auto color : rendering_.color_attachments)
                  if (color.color_attachment == e.image) {
                    recording_failed_ = true;
                    return false;
                  }
                if (!TransitionResource(image->NativeImage(),
                                        type == BINDING_TEXTURE2D ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                                  : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        image->DefaultState()))
                  return false;
                if (groups_dirty_) {
                  if (type == BINDING_TEXTURE2D)
                    device_->NativeDevice()->CreateShaderResourceView(image->NativeImage(), nullptr, handle);
                  else
                    device_->NativeDevice()->CreateUnorderedAccessView(image->NativeImage(), nullptr, nullptr, handle);
                }
              } else if (groups_dirty_) {
                D3D12_SAMPLER_DESC desc{};
                desc.Filter = e.sampler.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
                                                                        : D3D12_FILTER_MIN_MAG_MIP_POINT;
                const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                                                            D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                                            D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
                desc.AddressU = modes[e.sampler.address_u];
                desc.AddressV = modes[e.sampler.address_v];
                desc.AddressW = desc.AddressV;
                desc.MaxLOD = D3D12_FLOAT32_MAX;
                desc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
                desc.MaxAnisotropy = 1;
                device_->NativeDevice()->CreateSampler(&desc, handle);
              }
            }
    }
    if (groups_dirty_) {
      ID3D12DescriptorHeap *heaps[2]{};
      UINT count = 0;
      if (resources)
        heaps[count++] = resources->Heap();
      if (samplers)
        heaps[count++] = samplers->Heap();
      if (count)
        list_->SetDescriptorHeaps(count, heaps);
      for (uint32_t g = 0; g < groups_.size(); ++g)
        for (int kind = 0; kind < 2; ++kind) {
          auto parameter = program_->RootParameter(g, kind != 0);
          if (parameter == UINT32_MAX)
            continue;
          list_->SetGraphicsRootDescriptorTable(parameter, (kind ? samplers : resources)->GPU(starts[g][kind]));
        }
      groups_dirty_ = false;
    }
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool D3D12CommandContext::CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) {
  auto native = dynamic_cast<D3D12Image *>(image);
  if (!native || native->DeviceIdentity() != device_ || !native->IsValid() || !data || sealed_ || render_target_ ||
      IsDepthFormat(image->GetDesc().format) || row_pitch < uint64_t(image->GetDesc().width) * 4 ||
      (image->GetDesc().height - 1) > (SIZE_MAX - uint64_t(image->GetDesc().width) * 4) / row_pitch) {
    recording_failed_ = true;
    return false;
  }
  auto desc = native->NativeImage()->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows{};
  UINT64 bytes{}, total{};
  device_->NativeDevice()->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &bytes, &total);
  try {
    std::vector<uint8_t> copy(size_t(total), 0);
    for (UINT y = 0; y < rows; ++y)
      std::memcpy(copy.data() + size_t(y) * footprint.Footprint.RowPitch,
                  static_cast<const uint8_t *>(data) + size_t(y) * row_pitch, size_t(bytes));
    auto upload = device_->UploadPool().Acquire(total);
    if (!upload || !upload->Write(copy.data(), total) || !RetainResource(upload.Get()) || !RetainResource(native)) {
      recording_failed_ = true;
      return false;
    }
    if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_COPY_DEST, native->DefaultState()))
      return false;
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = upload->NativeBuffer();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;
    dst.pResource = native->NativeImage();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

}  // namespace grassland::rhi::backend::d3d12
