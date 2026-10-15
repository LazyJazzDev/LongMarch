#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include <cstdlib>

#include "grassland/rhi/backend/backend_glfw_helper.h"
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

bool D3D12CommandContext::Initialize() {
  reusable_ = device_->AcquireCommandResources(allocator_, list_, event_);
  return reusable_;
}

bool D3D12CommandContext::TransitionResource(ID3D12Resource *resource,
                                             D3D12_RESOURCE_STATES state,
                                             D3D12_RESOURCE_STATES default_state) {
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

bool D3D12CommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
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
  list_->ClearRenderTargetView(native->RenderTarget(), color.data(), 0, nullptr);
  return true;
}

bool D3D12CommandContext::CmdPresent(rhi::IWindow *window) {
#if !defined(NDEBUG)
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

}  // namespace grassland::rhi::backend::d3d12
