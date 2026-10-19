#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include <algorithm>
#include <cstdlib>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12CommandContext::D3D12CommandContext(D3D12Device *device) : CommandContext(device), device_(device) {
}

D3D12CommandContext::~D3D12CommandContext() {
  for (const auto &entry : rtv_leases_)
    device_->RTVPool().Release(entry.second.first);
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
      handle = existing->second.second;
      return true;
    }
    // Allocate tracking storage before acquiring a slot, so failure cannot leak a lease.
    auto inserted = rtv_leases_.try_emplace(image);
    if (!device_->RTVPool().Acquire(image->NativeImage(), inserted.first->second.first,
                                    inserted.first->second.second)) {
      rtv_leases_.erase(inserted.first);
      recording_failed_ = true;
      return false;
    }
    handle = inserted.first->second.second;
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
  program_ = native;
  list_->SetPipelineState(native->NativePipeline());
  list_->SetGraphicsRootSignature(native->NativeRootSignature());
  list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  return true;
}

bool D3D12CommandContext::CmdBeginRendering(const RenderingInfo &info) {
  auto image = info.color_attachment;
#if !defined(NDEBUG)
  auto native = dynamic_cast<D3D12Image *>(image);
  if (!native || !native->IsValid() || native->DeviceIdentity() != device_ || sealed_ || render_target_ ||
      info.load_op < ATTACHMENT_LOAD_OP_LOAD || info.load_op > ATTACHMENT_LOAD_OP_DISCARD ||
      info.store_op < ATTACHMENT_STORE_OP_STORE || info.store_op > ATTACHMENT_STORE_OP_DISCARD) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<D3D12Image *>(image);
#endif
  if (!RetainResource(native))
    return false;
  if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, native->DefaultState()))
    return false;
  D3D12_CPU_DESCRIPTOR_HANDLE handle{};
  if (!RenderTarget(native, handle))
    return false;
  if (info.load_op == ATTACHMENT_LOAD_OP_CLEAR)
    list_->ClearRenderTargetView(handle, info.clear_color.data(), 0, nullptr);
  else if (info.load_op == ATTACHMENT_LOAD_OP_DISCARD)
    list_->DiscardResource(native->NativeImage(), nullptr);
  list_->OMSetRenderTargets(1, &handle, FALSE, nullptr);
  store_op_ = info.store_op;
  render_target_ = native;
  return true;
}

bool D3D12CommandContext::CmdEndRendering() {
#if !defined(NDEBUG)
  if (sealed_ || !render_target_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (store_op_ == ATTACHMENT_STORE_OP_DISCARD)
    list_->DiscardResource(render_target_->NativeImage(), nullptr);
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
  if (program_->GetDesc().color_format != render_target_->GetDesc().format) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!PrepareDrawBuffers(false))
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
  if (program_->GetDesc().color_format != render_target_->GetDesc().format) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!PrepareDrawBuffers(true))
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
  if (!TransitionResource(render_target_->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                          render_target_->DefaultState()))
    return false;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
