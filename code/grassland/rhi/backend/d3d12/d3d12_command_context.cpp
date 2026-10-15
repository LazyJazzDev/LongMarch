#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

namespace grassland::rhi::backend::d3d12 {

D3D12CommandContext::D3D12CommandContext(D3D12Device *device, QueueType type)
    : CommandContext(device, type),
      device_(device) {
}

D3D12CommandContext::~D3D12CommandContext() {
  if (event_)
    CloseHandle(event_);
}

CommandContext::SubmitResult D3D12CommandContext::SubmitNative() {
  if (Frame() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SubmitResult::Rejected;
  }
  event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event_ ||
      !Check(device_->NativeDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
             "CreateCommandAllocator") ||
      !Check(device_->NativeDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                                        IID_PPV_ARGS(&list_)),
             "CreateCommandList"))
    return SubmitResult::Rejected;
  auto frame = dynamic_cast<D3D12SwapChainImage *>(Frame().Get());
  if (Frame() && !frame)
    return SubmitResult::Rejected;
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  if (frame) {
    if (!frame->IsValid())
      return SubmitResult::Rejected;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = frame->NativeImage();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    list_->ResourceBarrier(1, &barrier);
    list_->ClearRenderTargetView(frame->RenderTarget(), Color().data(), 0, nullptr);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list_->ResourceBarrier(1, &barrier);
  }
  if (!Check(list_->Close(), "CommandList::Close"))
    return SubmitResult::Rejected;
  failed_ = true;
  ID3D12CommandList *lists[] = {list_.Get()};
  device_->NativeQueue()->ExecuteCommandLists(1, lists);
  MarkSubmitted();
  const HRESULT present = frame ? frame->Present() : S_OK;
  value_ = device_->NextSerial();
  failed_ = !Check(device_->NativeQueue()->Signal(device_->Fence(), value_), "Queue::Signal");
  failed_ = !Check(present, "Swapchain::Present") || failed_;
  return failed_ ? SubmitResult::FailedInFlight : SubmitResult::Submitted;
}

SubmissionStatus D3D12CommandContext::WaitForProgress() {
  if (failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    if (device_->WaitIdle() || FAILED(device_->NativeDevice()->GetDeviceRemovedReason()))
      return SubmissionStatus::Failed;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return SubmissionStatus::Pending;
  }
  if (device_->Wait(value_, event_))
    return SubmissionStatus::Completed;
  // A failed host wait does not prove that GPU resource use has ended.
  failed_ = true;
  return SubmissionStatus::Pending;
}

}  // namespace grassland::rhi::backend::d3d12
