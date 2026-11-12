#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <system_error>
#include <unordered_map>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_image.h"
#include "grassland/rhi/rhi_program.h"
#include "grassland/rhi/rhi_queue.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::d3d12 {

D3D12CommandContextImpl::D3D12CommandContextImpl(ID3D12Queue *queue) : device_(queue->Owner()) {
  compute_queue_ = rhi::QueryInterfacePtr<rhi::IQueue>(queue)->GetType() == QUEUE_TYPE_COMPUTE;
  InitializeCommandContext(queue);
}

D3D12CommandContextImpl::~D3D12CommandContextImpl() {
  if (reusable_) {
    if (!closed_)
      reusable_ = SUCCEEDED(list_->Close());
    if (reusable_) {
      device_->RecycleCommandResources(std::move(allocator_), std::move(list_));
      return;
    }
  }
}

ICommandContext *D3D12CommandContextImpl::Create(ID3D12Queue *queue) {
  auto object = std::unique_ptr<D3D12CommandContextImpl>(new D3D12CommandContextImpl(queue));
  object->reusable_ =
      object->device_->AcquireCommandResources(object->allocator_, object->list_, queue->NativeQueue()->GetDesc().Type);
  if (!object->reusable_)
    return nullptr;
  if (object->device_->GetPhysicalDevice().ray_tracing_pipeline_feature ||
      object->device_->GetPhysicalDevice().ray_query_feature) {
    if (FAILED(object->list_.As(&object->ray_list_)))
      return nullptr;
  }
  return object.release();
}

bool D3D12CommandContextImpl::TransitionResource(ID3D12Resource *resource,
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
}

bool D3D12CommandContextImpl::RequireGraphicsQueue() {
  if (!compute_queue_)
    return true;
  recording_failed_ = true;
  ReportError(-1, "Graphics commands and depth clears are unavailable on a Compute queue.");
  return false;
}

bool D3D12CommandContextImpl::CmdClearImage(rhi::IImage *image, const ClearValue &value) {
  const auto native = rhi::QueryInterfacePtr<ID3D12Image>(image).Get();
#if !defined(NDEBUG)
  if (rendering_active_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  if (!native || !image->IsValid() || (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue()))) {
    recording_failed_ = true;
    ReportError(-1, "CmdClearImage requires a valid image from this device.");
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;
  const auto format = image->GetDesc().format;
  if (IsDepthFormat(format) && !RequireGraphicsQueue())
    return false;
  if (IsDepthFormat(format)) {
    if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_DEPTH_WRITE, native->DefaultState()))
      return false;
    auto flags = D3D12_CLEAR_FLAG_DEPTH;
    if (HasStencil(format))
      flags = static_cast<D3D12_CLEAR_FLAGS>(flags | D3D12_CLEAR_FLAG_STENCIL);
    list_->ClearDepthStencilView(native->AttachmentView(), flags, value.depth_stencil.depth,
                                 static_cast<UINT8>(value.depth_stencil.stencil), 0, nullptr);
    return true;
  }
  return ClearColor(native, image->GetDesc().format, value.color);
}

bool D3D12CommandContextImpl::ClearColor(ID3D12Image *image, ImageFormat format, const ClearColorValue &value) {
  const bool signed_integer = IsSignedIntegerFormat(format);
  const bool floating = !signed_integer && !IsUnsignedIntegerFormat(format);
  if (floating && !compute_queue_) {
    if (!TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState()))
      return false;
    list_->ClearRenderTargetView(image->AttachmentView(), value.float32.data(), 0, nullptr);
    return true;
  }
  if (!image->SupportsStorage()) {
    recording_failed_ = true;
    ReportError(-1, "D3D12 image clear requires UAV support on this queue.");
    return false;
  }
  const auto pixel_size = ImageFormatSize(format);
  const uint32_t component_size =
      pixel_size > 4 || format == IMAGE_FORMAT_R32_UINT || format == IMAGE_FORMAT_R32_INT ? 4 : 1;
  std::array<UINT, 4> integers{};
  for (size_t i = 0; !floating && i < integers.size(); ++i) {
    integers[i] = signed_integer ? static_cast<UINT>(value.int32[i]) : value.uint32[i];
    // Supply the native-width bit pattern, not a sign-extended uint32, to the
    // unsigned clear operation for 8-bit signed views.
    if (component_size == 1)
      integers[i] &= 0xff;
  }
  auto storage = device_->ResourceDescriptorStorage().Acquire(1);
  auto visible = device_->ResourceDescriptors().Acquire(1);
  if (!storage || !visible) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(storage.Get()) || !RetainResource(visible.Get()) ||
      !TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, image->DefaultState()))
    return false;
  D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
  view.Format = NativeImageFormat(format);
  view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device_->NativeDevice()->CreateUnorderedAccessView(image->NativeImage(), nullptr, &view, storage->CPU());
  device_->NativeDevice()->CopyDescriptorsSimple(1, visible->CPU(), storage->CPU(),
                                                 D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto heap = visible->Heap();
  list_->SetDescriptorHeaps(1, &heap);
  // UAV clears are compute-like and need ordering even if the state was already UAV.
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = image->NativeImage();
  list_->ResourceBarrier(1, &barrier);
  if (floating)
    list_->ClearUnorderedAccessViewFloat(visible->GPU(), storage->CPU(), image->NativeImage(), value.float32.data(), 0,
                                         nullptr);
  else
    list_->ClearUnorderedAccessViewUint(visible->GPU(), storage->CPU(), image->NativeImage(), integers.data(), 0,
                                        nullptr);
  // Restore existing heaps; their root tables must be rebound on the next draw/dispatch.
  ID3D12DescriptorHeap *heaps[2]{};
  UINT count = 0;
  for (auto page : descriptor_pages_)
    if (page)
      heaps[count++] = page->Heap();
  if (count)
    list_->SetDescriptorHeaps(count, heaps);
  tables_dirty_ = true;
  return compute_queue_ ||
         TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState());
}

bool D3D12CommandContextImpl::CmdPresent(rhi::IWindow *window) {
  if (!RequireGraphicsQueue())
    return false;
  const auto target = rhi::QueryInterfacePtr<ID3D12Window>(window).Get();
#if !defined(NDEBUG)
  if (rendering_active_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  if (!target || target->DeviceIdentity() != device_ || target->OwnerQueue() != OwnerQueue() ||
      !glfw::IsWindowThread()) {
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
#endif
  ObjPtr<IImage> frame_image = window->GetNextImage();
  auto frame = rhi::QueryInterfacePtr<ID3D12SwapChainImage>(frame_image.Get());
  if (!frame)
    return false;
  if (!RetainResource(frame.Get()))
    return false;
  if (!target->RecordImGui(this, frame_image.Get())) {
    recording_failed_ = true;
    return false;
  }
  presents_.emplace_back(target, frame.Get());

  return TransitionResource(frame->NativeImage(), D3D12_RESOURCE_STATE_PRESENT, frame->DefaultState());
}

bool D3D12CommandContextImpl::EndRecording() {
#if !defined(NDEBUG)
  if (rendering_active_) {
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

ID3D12CommandContext::SubmitResult D3D12CommandContextImpl::SubmitNative(const SubmitInfo &info, Event signal) {
  auto queue = queue_;
#if !defined(NDEBUG)
  if (!presents_.empty() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  for (const auto &present : presents_) {
    if (!rhi::QueryInterfacePtr<IImage>(present.second)->IsValid())
      return SUBMIT_RESULT_REJECTED;
  }
#endif
  {
    for (const auto &wait : info.waits) {
      auto timeline = rhi::QueryInterfacePtr<ID3D12Timeline>(wait.timeline).Get();
      if (!Check(queue->NativeQueue()->Wait(timeline->NativeFence(), wait.value), "Queue::Wait"))
        return SUBMIT_RESULT_REJECTED;
    }
    submission_failed_ = true;
    ID3D12CommandList *lists[] = {list_.Get()};
    queue->NativeQueue()->ExecuteCommandLists(1, lists);
    MarkSubmitted();
    for (auto &entry : as_dependencies_)
      entry.first->CommitDependencies(std::move(entry.second));
    as_dependencies_.clear();
    bool presented = true;
    for (const auto &present : presents_) {
      if (!Check(present.second->Present(), "Swapchain::Present"))
        presented = false;
    }
    submission_failed_ =
        !Check(queue->NativeQueue()->Signal(queue->GetTimeline()->NativeFence(), signal.value), "Queue::Signal");
    if (!presented)
      submission_failed_ = true;
  }
  if (submission_failed_)
    return SUBMIT_RESULT_FAILED_IN_FLIGHT;
  return SUBMIT_RESULT_SUBMITTED;
}

void D3D12CommandContextImpl::UpdatePresentedImages() {
  for (const auto &present : presents_) {
    if (!present.first->UpdateNextImage()) {
      ReportError(-1, "Failed to prepare the next swapchain image after submission.");
    }
  }
}

SubmissionStatus D3D12CommandContextImpl::PollCompletion(Event signal, uint64_t completed, bool timeline_valid) {
  auto queue = queue_;
  if (!timeline_valid)
    return SUBMISSION_STATUS_FAILED;
  if (completed >= signal.value)
    return submission_failed_ ? SUBMISSION_STATUS_FAILED : SUBMISSION_STATUS_COMPLETED;
  if (submission_failed_) {
    if (FAILED(device_->NativeDevice()->GetDeviceRemovedReason()))
      return SUBMISSION_STATUS_FAILED;
    // A failed signal does not prove the preceding ExecuteCommandLists finished.
    // Retry with a new monotonically ordered value, without blocking other queues.
    std::lock_guard<std::mutex> lock(queue->QueueMutex());
    Check(queue->NativeQueue()->Signal(queue->GetTimeline()->NativeFence(), queue->NextSerial()),
          "Signal failed submission retirement");
  }
  return SUBMISSION_STATUS_PENDING;
}

bool D3D12CommandContextImpl::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
  auto native = rhi::QueryInterfacePtr<ID3D12Buffer>(buffer).Get();
#if !defined(NDEBUG)
  if (rendering_active_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  assert(!native || native->Owner() == device_);
  if (!native || sealed_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;

  if (!data || !size || offset > buffer->GetDesc().size || size > buffer->GetDesc().size - offset) {
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

bool D3D12CommandContextImpl::CmdCopyBuffer(IBuffer *source,
                                            uint64_t source_offset,
                                            IBuffer *destination,
                                            uint64_t destination_offset,
                                            uint64_t size) {
  auto src = rhi::QueryInterfacePtr<ID3D12Buffer>(source).Get();
  auto dst = rhi::QueryInterfacePtr<ID3D12Buffer>(destination).Get();
#if !defined(NDEBUG)
  if (rendering_active_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  assert(!src || src->Owner() == device_);
  assert(!dst || dst->Owner() == device_);
  if (!src || !dst || sealed_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (src == dst || !size || source_offset > source->GetDesc().size || size > source->GetDesc().size - source_offset ||
      destination_offset > destination->GetDesc().size || size > destination->GetDesc().size - destination_offset) {
    recording_failed_ = true;
    return false;
  }
  return CopyBuffer(src, source_offset, dst, destination_offset, size);
}

bool D3D12CommandContextImpl::CopyBuffer(ID3D12Buffer *src,
                                         uint64_t source_offset,
                                         ID3D12Buffer *dst,
                                         uint64_t destination_offset,
                                         uint64_t size) {
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

bool D3D12CommandContextImpl::CmdBindComputeProgram(IComputeProgram *program) {
  auto native = rhi::QueryInterfacePtr<ID3D12ComputeProgram>(program).Get();
#if !defined(NDEBUG)
  if (sealed_ || rendering_active_ || !native || native->Owner() != device_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;
  native_program_ = native;
  compute_active_ = true;
  ray_program_ = nullptr;
  program_ = nullptr;
  binding_layout_ = &program->GetDesc().pipeline_layout;
  groups_.assign(binding_layout_->groups.size(), {});
  tables_dirty_ = true;
  list_->SetPipelineState(native->NativePipeline());
  list_->SetComputeRootSignature(native->NativeRootSignature());
  return true;
}

bool D3D12CommandContextImpl::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
#if !defined(NDEBUG)
  if (sealed_ || rendering_active_ || !compute_active_ || ray_program_) {
    recording_failed_ = true;
    return false;
  }
  if (x > 65535 || y > 65535 || z > 65535) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!PrepareGroups())
    return false;
  list_->Dispatch(x, y, z);
  return true;
}

bool D3D12CommandContextImpl::CmdBindProgram(IProgram *program) {
  if (!RequireGraphicsQueue())
    return false;
  auto native = rhi::QueryInterfacePtr<ID3D12Program>(program).Get();
#if !defined(NDEBUG)
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;
  vertex_buffers_.assign(program->GetDesc().vertex_bindings.size(), nullptr);

  groups_.assign(program->GetDesc().pipeline_layout.groups.size(), {});

  tables_dirty_ = true;
  program_ = program;
  native_program_ = native;
  compute_active_ = false;
  ray_program_ = nullptr;
  binding_layout_ = &program->GetDesc().pipeline_layout;
  list_->OMSetStencilRef(program->GetDesc().depth_stencil.stencil_reference);
  list_->SetPipelineState(native->NativePipeline());
  list_->SetGraphicsRootSignature(native->NativeRootSignature());
  list_->IASetPrimitiveTopology(NativePrimitiveTopology(program->GetDesc().topology));
  return true;
}

bool D3D12CommandContextImpl::CmdBeginRendering(const RenderingInfo &info) {
  if (!RequireGraphicsQueue())
    return false;
  if (sealed_ || rendering_active_ || !PrepareRendering(info)) {
    recording_failed_ = true;
    return false;
  }
  std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 8> colors{};
  auto count = static_cast<UINT>(rendering_.color_attachments.size());
  for (UINT i = 0; i < count; ++i) {
    const auto &attachment = rendering_.color_attachments[i];
    auto image = rhi::QueryInterfacePtr<ID3D12Image>(attachment.color_attachment).Get();
    if (attachment.load_op == ATTACHMENT_LOAD_OP_CLEAR) {
      if (!ClearColor(image, attachment.color_attachment->GetDesc().format, attachment.clear_value))
        return false;
    } else {
      if (!TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState()))
        return false;
      if (attachment.load_op == ATTACHMENT_LOAD_OP_DISCARD)
        list_->DiscardResource(image->NativeImage(), nullptr);
    }
    colors[i] = image->AttachmentView();
  }
  auto depth = rhi::QueryInterfacePtr<ID3D12Image>(rendering_.depth_stencil.image).Get();
  D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
  if (depth) {
    dsv = depth->AttachmentView();
    if (!TransitionResource(depth->NativeImage(), D3D12_RESOURCE_STATE_DEPTH_WRITE, depth->DefaultState()))
      return false;
    const auto &ds = rendering_.depth_stencil;
    UINT flags = ds.depth_load_op == ATTACHMENT_LOAD_OP_CLEAR ? D3D12_CLEAR_FLAG_DEPTH : 0;
    if (HasStencil(rendering_.depth_stencil.image->GetDesc().format) && ds.stencil_load_op == ATTACHMENT_LOAD_OP_CLEAR)
      flags |= D3D12_CLEAR_FLAG_STENCIL;
    if (flags)
      list_->ClearDepthStencilView(dsv, static_cast<D3D12_CLEAR_FLAGS>(flags), ds.clear_value.depth,
                                   static_cast<UINT8>(ds.clear_value.stencil), 0, nullptr);
  }
  list_->OMSetRenderTargets(count, colors.data(), FALSE, depth ? &dsv : nullptr);
  rendering_active_ = true;
  return true;
}

bool D3D12CommandContextImpl::CmdEndRendering() {
  if (!RequireGraphicsQueue())
    return false;
  if (sealed_ || !rendering_active_ || recording_failed_) {
    recording_failed_ = true;
    return false;
  }
  for (const auto &color : rendering_.color_attachments)
    if (color.store_op == ATTACHMENT_STORE_OP_DISCARD)
      list_->DiscardResource(rhi::QueryInterfacePtr<ID3D12Image>(color.color_attachment).Get()->NativeImage(), nullptr);
  list_->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
  rendering_active_ = false;
  rendering_ = {};
  return true;
}

bool D3D12CommandContextImpl::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
  if (!RequireGraphicsQueue())
    return false;
  auto native = rhi::QueryInterfacePtr<ID3D12Buffer>(buffer).Get();
#if !defined(NDEBUG)
  assert(!native || native->Owner() == device_);
  if (!native || sealed_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;
  if (!program_ || binding >= program_->GetDesc().vertex_bindings.size() || offset >= buffer->GetDesc().size) {
    recording_failed_ = true;
    return false;
  }
  if (buffer->GetDesc().size - offset > UINT_MAX) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionResource(native->NativeBuffer(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                          native->DefaultState()))
    return false;
  D3D12_VERTEX_BUFFER_VIEW view{native->NativeBuffer()->GetGPUVirtualAddress() + offset,
                                static_cast<UINT>(buffer->GetDesc().size - offset),
                                program_->GetDesc().vertex_bindings[binding].stride};
  list_->IASetVertexBuffers(binding, 1, &view);
  vertex_buffers_[binding] = native;
  return true;
}

bool D3D12CommandContextImpl::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
  if (!RequireGraphicsQueue())
    return false;
  auto native = rhi::QueryInterfacePtr<ID3D12Buffer>(buffer).Get();
#if !defined(NDEBUG)
  assert(!native || native->Owner() == device_);
  if (!native || sealed_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (!RetainResource(native))
    return false;
  if (format != INDEX_FORMAT_UINT16 && format != INDEX_FORMAT_UINT32) {
    recording_failed_ = true;
    return false;
  }
  auto alignment = format == INDEX_FORMAT_UINT16 ? 2u : 4u;
  if (offset >= buffer->GetDesc().size || offset % alignment) {
    recording_failed_ = true;
    return false;
  }
  if (buffer->GetDesc().size - offset > UINT_MAX) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionResource(native->NativeBuffer(), D3D12_RESOURCE_STATE_INDEX_BUFFER, native->DefaultState()))
    return false;
  D3D12_INDEX_BUFFER_VIEW view{native->NativeBuffer()->GetGPUVirtualAddress() + offset,
                               static_cast<UINT>(buffer->GetDesc().size - offset),
                               format == INDEX_FORMAT_UINT16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT};
  list_->IASetIndexBuffer(&view);
  index_buffer_ = native;
  index_bound_ = true;
  return true;
}

bool D3D12CommandContextImpl::CmdSetViewport(const Viewport &viewport) {
  if (!RequireGraphicsQueue())
    return false;
#if !defined(NDEBUG)
  if (sealed_ || viewport.width <= 0 || viewport.height <= 0) {
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

bool D3D12CommandContextImpl::CmdSetScissor(const Scissor &scissor) {
  if (!RequireGraphicsQueue())
    return false;
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

bool D3D12CommandContextImpl::CmdDraw(uint32_t vertex_count,
                                      uint32_t instance_count,
                                      uint32_t first_vertex,
                                      uint32_t first_instance) {
  if (!RequireGraphicsQueue())
    return false;
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !rendering_active_ || !viewport_set_ || !scissor_set_) {
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

bool D3D12CommandContextImpl::CmdDrawIndexed(uint32_t index_count,
                                             uint32_t instance_count,
                                             uint32_t first_index,
                                             int32_t vertex_offset,
                                             uint32_t first_instance) {
  if (!RequireGraphicsQueue())
    return false;
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !rendering_active_ || !viewport_set_ || !scissor_set_ || !index_bound_) {
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

bool D3D12CommandContextImpl::PrepareDrawBuffers(bool indexed) {
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
    auto image = rhi::QueryInterfacePtr<ID3D12Image>(color.color_attachment).Get();
    if (!TransitionResource(image->NativeImage(), D3D12_RESOURCE_STATE_RENDER_TARGET, image->DefaultState()))
      return false;
  }
  return true;
}

bool D3D12CommandContextImpl::CmdBindGroup(uint32_t index, IBindingGroup *group) {
  auto native = rhi::QueryInterfacePtr<ID3D12BindingGroup>(group).Get();
  if (sealed_ || !binding_layout_ || !native || native->Owner() != device_ || index >= groups_.size() ||
      !SameBindingLayout(binding_layout_->groups[index], group->GetLayout()) ||
      !native->CurrentVersion()->IsComplete()) {
    recording_failed_ = true;
    return false;
  }
  auto version = native->CurrentVersion();
  auto resources = version->ResourceDescriptors();
  if (version->ResourceCount() && !resources) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(version))
    return false;

  auto &bound = groups_[index];
  std::array<D3D12DescriptorPool::Lease *, 2> sources{resources, version->SamplerDescriptors()};
  for (size_t kind = 0; kind < sources.size(); ++kind) {
    if (bound.sources[kind] != sources[kind]) {
      bound.sources[kind] = sources[kind];
      bound.dirty[kind] = true;
      tables_dirty_ = true;
    }
  }
  bound.version = version;
  return true;
}

bool D3D12CommandContextImpl::PrepareGroups() {
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto version = groups_[i].version;
    if (!version) {
      if (!binding_layout_->groups[i].bindings.empty()) {
        recording_failed_ = true;
        return false;
      }
      continue;
    }
    for (size_t binding_index = 0; binding_index < version->GetLayout().bindings.size(); ++binding_index) {
      const auto &binding = version->GetLayout().bindings[binding_index];
      for (uint32_t j = 0; j < binding.count; ++j) {
        auto type = binding.type;
        ID3D12Resource *resource = nullptr;
        if (type == BINDING_ACCELERATION_STRUCTURE) {
          auto top = rhi::QueryInterfacePtr<ID3D12TopLevelAccelerationStructure>(
                         version->GetReferences()[binding_index][j].Get())
                         .Get();
          for (const auto &blas : top->BottomLevels())
            if (!RetainResource(blas.Get()))
              return false;
        }
        if (type <= BINDING_RW_BYTE_ADDRESS_BUFFER) {
          auto buffer = rhi::QueryInterfacePtr<ID3D12Buffer>(version->GetReferences()[binding_index][j].Get()).Get();
          assert(!buffer || buffer->Owner() == device_);
          resource = buffer->NativeBuffer();
          if (!TransitionResource(
                  resource,
                  (type == BINDING_RW_STRUCTURED_BUFFER || type == BINDING_RW_BYTE_ADDRESS_BUFFER)
                      ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                  : compute_queue_ ? (type == BINDING_UNIFORM_BUFFER ? D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                                                                     : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                                   : D3D12_RESOURCE_STATE_GENERIC_READ,
                  buffer->DefaultState()))
            return false;
        } else if (type == BINDING_TEXTURE2D || type == BINDING_RW_TEXTURE2D) {
          auto image = rhi::QueryInterfacePtr<ID3D12Image>(version->GetReferences()[binding_index][j].Get()).Get();
          resource = image->NativeImage();
          for (auto color : rendering_.color_attachments)
            if (color.color_attachment == rhi::QueryInterfacePtr<IImage>(image).Get()) {
              recording_failed_ = true;
              return false;
            }
          if (!TransitionResource(resource,
                                  type == BINDING_TEXTURE2D
                                      ? (compute_queue_ ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                        : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                                      : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  image->DefaultState()))
            return false;
        }
        if (type == BINDING_RW_STRUCTURED_BUFFER || type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
            type == BINDING_RW_TEXTURE2D) {
          D3D12_RESOURCE_BARRIER barrier{};
          barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
          barrier.UAV.pResource = resource;
          list_->ResourceBarrier(1, &barrier);
        }
      }
    }
  }
  if (!tables_dirty_)
    return true;
  bool heaps_changed = false;
  for (size_t kind = 0; kind < 2; ++kind) {
    uint64_t total = 0, needed = 0;
    for (const auto &bound : groups_) {
      if (!bound.sources[kind])
        continue;
      total += bound.sources[kind]->Count();
      if (bound.dirty[kind])
        needed += bound.sources[kind]->Count();
    }
    if (!needed)
      continue;
    const auto available = descriptor_pages_[kind] ? descriptor_pages_[kind]->Count() - descriptor_used_[kind] : 0;
    if (needed > available) {
      const uint32_t limit = kind ? 2048 : 1000000;
      if (total > limit) {
        recording_failed_ = true;
        return false;
      }
      auto &pool = kind ? device_->SamplerDescriptors() : device_->ResourceDescriptors();
      // The context chooses growth; the pool only rounds and reuses size classes.
      // At the native limit, further rollovers necessarily use equal-size heaps.
      const uint32_t previous = descriptor_pages_[kind] ? descriptor_pages_[kind]->Count() : 0;
      const uint32_t minimum = std::max(uint32_t(total), std::min(previous * 2, limit));
      auto page = pool.Acquire(minimum);
      if (!page) {
        recording_failed_ = true;
        return false;
      }
      if (!RetainResource(page.Get()))
        return false;
      descriptor_pages_[kind] = page.Get();
      descriptor_used_[kind] = 0;
      heaps_changed = true;
      // Every still-bound table of this heap type must move together. Earlier
      // draws retain the old page through the context's resource references.
      for (auto &bound : groups_)
        bound.dirty[kind] = bound.sources[kind] != nullptr;
    }
    auto page = descriptor_pages_[kind];
    const auto type = kind ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    for (auto &bound : groups_) {
      if (!bound.dirty[kind] || !bound.sources[kind])
        continue;
      auto source = bound.sources[kind];
      auto &cursor = descriptor_used_[kind];
      // Copy only from CPU storage: shader-visible heaps cannot be copy sources.
      device_->NativeDevice()->CopyDescriptorsSimple(source->Count(), page->CPU(cursor), source->CPU(), type);
      bound.tables[kind] = page->GPU(cursor);
      cursor += source->Count();
      bound.dirty[kind] = false;
    }
  }
  if (heaps_changed) {
    ID3D12DescriptorHeap *heaps[2]{};
    UINT count = 0;
    for (auto page : descriptor_pages_)
      if (page)
        heaps[count++] = page->Heap();
    list_->SetDescriptorHeaps(count, heaps);
  }
  // Changing either heap invalidates root tables. Rebind both heap types,
  // including unchanged tables which did not need to be copied again.
  for (uint32_t g = 0; g < groups_.size(); ++g)
    for (size_t kind = 0; kind < 2; ++kind) {
      auto parameter = native_program_->RootParameter(g, kind != 0);
      if (parameter != UINT32_MAX) {
        if (compute_active_)
          list_->SetComputeRootDescriptorTable(parameter, groups_[g].tables[kind]);
        else
          list_->SetGraphicsRootDescriptorTable(parameter, groups_[g].tables[kind]);
      }
    }
  tables_dirty_ = false;
  return true;
}

bool D3D12CommandContextImpl::CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) {
  auto native = rhi::QueryInterfacePtr<ID3D12Image>(image).Get();
  if (!native || (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue())) || !image->IsValid() ||
      !data || sealed_ || rendering_active_ || IsDepthFormat(image->GetDesc().format) ||
      row_pitch < uint64_t(image->GetDesc().width) * ImageFormatSize(image->GetDesc().format) ||
      (image->GetDesc().height - 1) >
          (SIZE_MAX - uint64_t(image->GetDesc().width) * ImageFormatSize(image->GetDesc().format)) / row_pitch) {
    recording_failed_ = true;
    return false;
  }
  auto desc = native->NativeImage()->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows{};
  UINT64 bytes{}, total{};
  device_->NativeDevice()->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &bytes, &total);
  auto upload = device_->UploadPool().Acquire(total);
  if (!upload || !RetainResource(upload.Get()) || !RetainResource(native)) {
    recording_failed_ = true;
    return false;
  }
  for (UINT y = 0; y < rows; ++y) {
    if (!upload->Write(static_cast<const uint8_t *>(data) + size_t(y) * row_pitch, bytes,
                       footprint.Offset + uint64_t(y) * footprint.Footprint.RowPitch)) {
      recording_failed_ = true;
      return false;
    }
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
}

bool D3D12CommandContextImpl::CmdBindRayTracingProgram(IRayTracingProgram *program) {
  auto native = rhi::QueryInterfacePtr<ID3D12RayTracingProgram>(program).Get();
  if (sealed_ || rendering_active_ || !native || native->Owner() != device_) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(native))
    return false;
  native_program_ = native;
  ray_program_ = native;
  compute_active_ = true;
  program_ = nullptr;
  binding_layout_ = &program->GetDesc().pipeline_layout;
  groups_.assign(binding_layout_->groups.size(), {});
  tables_dirty_ = true;
  ray_list_->SetPipelineState1(native->NativePipeline());
  list_->SetComputeRootSignature(native->NativeRootSignature());
  return true;
}

bool D3D12CommandContextImpl::CmdDispatchRays(uint32_t width,
                                              uint32_t height,
                                              uint32_t depth,
                                              IShaderBindingTable *sbt) {
  if (sealed_ || rendering_active_ || !ray_program_ || !width || !height || !depth) {
    recording_failed_ = true;
    return false;
  }
  if (recording_failed_)
    return false;
  if (uint64_t(width) * height > (1ull << 30) / depth) {
    recording_failed_ = true;
    return false;
  }
  const IBackendShaderBindingTable *table = ray_program_->DefaultShaderBindingTable();
  if (sbt) {
    auto custom = rhi::QueryInterfacePtr<IBackendShaderBindingTable>(sbt).Get();
    if (!custom || sbt->GetProgram() != rhi::QueryInterfacePtr<IRayTracingProgram>(ray_program_).Get()) {
      recording_failed_ = true;
      return false;
    }
    if (!RetainResource(custom))
      return false;
    table = custom;
  }
  const auto &regions = table->Regions();
  if (!PrepareGroups())
    return false;
  D3D12_DISPATCH_RAYS_DESC dispatch{};
  dispatch.RayGenerationShaderRecord = {regions[0].address, regions[0].size};
  dispatch.MissShaderTable = {regions[1].address, regions[1].size, regions[1].stride};
  dispatch.HitGroupTable = {regions[2].address, regions[2].size, regions[2].stride};
  dispatch.CallableShaderTable = {regions[3].address, regions[3].size, regions[3].stride};
  dispatch.Width = width;
  dispatch.Height = height;
  dispatch.Depth = depth;
  ray_list_->DispatchRays(&dispatch);
  return true;
}

IDownloadToken *D3D12CommandContextImpl::CmdDownloadBuffer(IBuffer *buffer, uint64_t offset, uint64_t size) {
  auto native = rhi::QueryInterfacePtr<ID3D12Buffer>(buffer).Get();
#if !defined(NDEBUG)
  assert(!native || native->Owner() == device_);
  if (sealed_ || rendering_active_ || !native) {
    recording_failed_ = true;
    return nullptr;
  }
#endif
  if (recording_failed_ || !size || offset > buffer->GetDesc().size || size > buffer->GetDesc().size - offset) {
    recording_failed_ = true;
    return nullptr;
  }
  auto lease = device_->DownloadPool().Acquire(size);
  if (!lease || !CopyBuffer(native, offset, lease.Get(), 0, size)) {
    recording_failed_ = true;
    return nullptr;
  }
  return RegisterDownload(ObjectPtr<ID3D12DownloadToken>::Adopt(CreateD3D12DownloadToken(std::move(lease), size)));
}

IDownloadToken *D3D12CommandContextImpl::CmdDownloadImage(IImage *image, const ImageDownloadRegion &region) {
  auto native = rhi::QueryInterfacePtr<ID3D12Image>(image).Get();
#if !defined(NDEBUG)
  if (sealed_ || rendering_active_ || !native ||
      (native->DeviceIdentity() != device_ || !native->SupportsQueue(OwnerQueue()))) {
    recording_failed_ = true;
    return nullptr;
  }
  if (image->GetDesc().format == IMAGE_FORMAT_D24_UNORM_S8_UINT) {
    recording_failed_ = true;
    ReportError(-1, "CmdDownloadImage does not support D24S8 readback; use a depth-only or color image.");
    return nullptr;
  }
#endif
  if (recording_failed_ || !image->IsValid()) {
    recording_failed_ = true;
    return nullptr;
  }
  const auto desc = image->GetDesc();
  if (region.x >= desc.width || region.y >= desc.height) {
    recording_failed_ = true;
    return nullptr;
  }
  const uint32_t width = region.width ? region.width : desc.width - region.x;
  const uint32_t height = region.height ? region.height : desc.height - region.y;
  const uint64_t packed = uint64_t(width) * ImageFormatSize(desc.format);
  if (width > desc.width - region.x || height > desc.height - region.y || !packed || height > SIZE_MAX / packed) {
    recording_failed_ = true;
    return nullptr;
  }
  if (IsDepthFormat(desc.format)) {
    auto native_desc = native->NativeImage()->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
    UINT rows[2]{};
    UINT64 row_bytes[2]{}, total{};
    const UINT planes = HasStencil(desc.format) ? 2 : 1;
    device_->NativeDevice()->GetCopyableFootprints(&native_desc, 0, planes, 0, footprints, rows, row_bytes, &total);
    auto lease = device_->DownloadPool().Acquire(total);
    if (!lease || !RetainResource(native) || !RetainResource(lease.Get())) {
      recording_failed_ = true;
      return nullptr;
    }
    if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_COPY_SOURCE, native->DefaultState()))
      return nullptr;
    for (UINT plane = 0; plane < planes; ++plane) {
      D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
      src.pResource = native->NativeImage();
      src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      src.SubresourceIndex = plane;
      dst.pResource = lease->NativeBuffer();
      dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      dst.PlacedFootprint = footprints[plane];
      list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    const auto depth = footprints[0], stencil = footprints[1];
    DownloadLayout layout;
    layout.offset = depth.Offset + uint64_t(region.y) * depth.Footprint.RowPitch +
                    uint64_t(region.x) * ImageFormatSize(desc.format);
    layout.row_pitch = depth.Footprint.RowPitch;
    layout.row_bytes = packed;
    layout.rows = height;
    if (planes == 2) {
      layout.stencil_offset = stencil.Offset + uint64_t(region.y) * stencil.Footprint.RowPitch + region.x;
      layout.stencil_pitch = stencil.Footprint.RowPitch;
    }
    return RegisterDownload(ObjectPtr<ID3D12DownloadToken>::Adopt(CreateD3D12DownloadToken(std::move(lease), layout)));
  }
  const uint64_t pitch =
      (packed + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(uint64_t(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) - 1);
  if (pitch > UINT32_MAX || height > SIZE_MAX / pitch) {
    recording_failed_ = true;
    return nullptr;
  }
  auto lease = device_->DownloadPool().Acquire(pitch * height);
  if (!lease || !RetainResource(native) || !RetainResource(lease.Get())) {
    recording_failed_ = true;
    return nullptr;
  }
  if (!TransitionResource(native->NativeImage(), D3D12_RESOURCE_STATE_COPY_SOURCE, native->DefaultState()))
    return nullptr;
  D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
  src.pResource = native->NativeImage();
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst.pResource = lease->NativeBuffer();
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint.Footprint = {native->NativeImage()->GetDesc().Format, width, height, 1, static_cast<UINT>(pitch)};
  D3D12_BOX box{region.x, region.y, 0, region.x + width, region.y + height, 1};
  list_->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
  return RegisterDownload(ObjectPtr<ID3D12DownloadToken>::Adopt(
      CreateD3D12DownloadToken(std::move(lease), DownloadLayout{0, pitch, packed, height})));
}

ICommandContext *CreateD3D12CommandContext(ID3D12Queue *queue) {
  return D3D12CommandContextImpl::Create(queue);
}

bool D3D12CommandContextImpl::CmdDrawImGui(IWindow *window, IImage *target) {
  if (!RequireGraphicsQueue())
    return false;
  auto native = rhi::QueryInterfacePtr<ID3D12Window>(window).Get();
  auto image = rhi::QueryInterfacePtr<ID3D12Image>(target).Get();
#if !defined(NDEBUG)
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdDrawImGui requires a window and target from this device outside rendering on the window thread.");
    return false;
  }
#endif
  const auto format = target->GetDesc().format;
  if (!IsColorFormat(format) || IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format)) {
    recording_failed_ = true;
    ReportError(-1, "CmdDrawImGui requires a non-integer color target.");
    return false;
  }
  if (!RetainResource(target))
    return false;
  if (!native->RecordImGui(this, target, COLOR_SPACE_BT709_LINEAR, 1.0f)) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

bool D3D12CommandContextImpl::CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) {
  if (!RequireGraphicsQueue())
    return false;
  auto native = rhi::QueryInterfacePtr<ID3D12Window>(window).Get();
  auto image = rhi::QueryInterfacePtr<ID3D12Image>(frame_img).Get();
#if !defined(NDEBUG)
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdPresentImage requires a window and image from this device outside rendering on the window thread.");
    return false;
  }
#endif
  if (!RetainResource(frame_img))
    return false;
  if (!native->RecordPresentImage(this, frame_img, frame_color_space)) {
    recording_failed_ = true;
    return false;
  }
  return CmdPresent(window);
}

bool D3D12CommandContextImpl::PrepareRendering(const RenderingInfo &info) {
  auto next = info;
  if (next.color_attachments.size() > 8 || (next.color_attachments.empty() && !next.depth_stencil.image)) {
    recording_failed_ = true;
    return false;
  }
  std::vector<IImage *> images;
  auto validate = [&](IImage *image, bool depth, AttachmentLoadOp load, AttachmentStoreOp store) {
    auto native = rhi::QueryInterfacePtr<ID3D12Image>(image).Get();
    if (!native || !image->IsValid() ||
        (native->DeviceIdentity() != DeviceIdentity() || !native->SupportsQueue(OwnerQueue())) ||
        IsDepthFormat(image->GetDesc().format) != depth || load < ATTACHMENT_LOAD_OP_LOAD ||
        load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE || store > ATTACHMENT_STORE_OP_DISCARD)
      return false;
    for (auto previous : images)
      if (previous == image || previous->GetDesc().width != image->GetDesc().width ||
          previous->GetDesc().height != image->GetDesc().height)
        return false;
    images.push_back(image);
    return RetainResource(image);
  };
  for (auto color : next.color_attachments)
    if (!validate(color.color_attachment, false, color.load_op, color.store_op)) {
      recording_failed_ = true;
      return false;
    }
  const auto &ds = next.depth_stencil;
  if (ds.image &&
      (!validate(ds.image, true, ds.depth_load_op, ds.depth_store_op) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
       ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
       ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD ||
       !(ds.clear_value.depth >= 0 && ds.clear_value.depth <= 1))) {
    recording_failed_ = true;
    return false;
  }
  rendering_ = std::move(next);
  return true;
}

bool D3D12CommandContextImpl::RenderingMatches(const IProgram *program) const {
  const auto &desc = program->GetDesc();
  if (desc.color_formats.size() != rendering_.color_attachments.size())
    return false;
  for (size_t i = 0; i < desc.color_formats.size(); ++i)
    if (desc.color_formats[i] != rendering_.color_attachments[i].color_attachment->GetDesc().format)
      return false;
  return desc.depth_stencil.format ==
         (rendering_.depth_stencil.image ? rendering_.depth_stencil.image->GetDesc().format : IMAGE_FORMAT_UNKNOWN);
}

void D3D12CommandContextImpl::InitializeCommandContext(ID3D12Queue *queue) {
  queue_ = queue;
}

rhi::ISubmissionToken *D3D12CommandContextImpl::Submit(const rhi::SubmitInfo &info) {
  try {
    auto submission = queue_->Submit(ObjectPtr<ID3D12CommandContext>::Retain(this), info);
    return rhi::QueryInterfacePtr<ISubmissionToken>(submission.Get()).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}

void D3D12CommandContextImpl::BindDownloads(rhi::ISubmissionToken *submission) {
  for (const auto &token : downloads_)
    token->BindSubmission(submission);
}

void D3D12CommandContextImpl::ReleaseDownloads() {
  downloads_.clear();
}

IDownloadToken *D3D12CommandContextImpl::RegisterDownload(ObjectPtr<ID3D12DownloadToken> token) {
  downloads_.push_back(token);
  return rhi::QueryInterfacePtr<IDownloadToken>(token.Get()).Detach();
}

bool D3D12CommandContextImpl::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (sealed_)
    return false;
#endif
  if (recording_failed_)
    return false;
  auto identity = rhi::QueryInterfacePtr<rhi::IObject>(resource);
  if (!identity)
    return false;
  resource = identity.Get();
  if (resources_.find(resource) != resources_.end())
    return true;
  resources_.emplace(resource, ObjectPtr<rhi::IObject>::Adopt(identity.Detach()));

  return true;
}

bool D3D12CommandContextImpl::Seal() {
  if (recording_failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_) {
    ReportError(-1, "Cannot submit a command context twice.");
    return false;
  }
#endif
  sealed_ = true;
  if (!EndRecording()) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

ID3D12Queue *D3D12CommandContextImpl::OwnerQueue() const {
  return queue_;
}

const void *D3D12CommandContextImpl::DeviceIdentity() const {
  return queue_->Owner();
}

void D3D12CommandContextImpl::MarkSubmitted() {
  submitted_ = true;
}

bool D3D12CommandContextImpl::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend::d3d12
