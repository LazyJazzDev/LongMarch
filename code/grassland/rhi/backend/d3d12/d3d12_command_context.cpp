#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

namespace grassland::rhi::backend::d3d12 {

D3D12CommandContext::D3D12CommandContext(D3D12Device *device) : CommandContext(device), device_(device) {
}

D3D12CommandContext::~D3D12CommandContext() {
  if (event_)
    CloseHandle(event_);
}

bool D3D12CommandContext::Initialize() {
  event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event_)
    return false;
  if (!Check(device_->NativeDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
             "CreateCommandAllocator"))
    return false;
  return Check(device_->NativeDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                                          IID_PPV_ARGS(&list_)),
               "CreateCommandList");
}

bool D3D12CommandContext::RecordClear(Image *image, const std::array<float, 4> &color, bool first_use) {
#if !defined(NDEBUG)
  const auto frame = dynamic_cast<D3D12SwapChainImage *>(image);
  if (!frame || !list_)
    return false;
#else
  const auto frame = static_cast<D3D12SwapChainImage *>(image);
#endif
  if (first_use) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = frame->NativeImage();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    list_->ResourceBarrier(1, &barrier);
  }
  list_->ClearRenderTargetView(frame->RenderTarget(), color.data(), 0, nullptr);
  return true;
}

bool D3D12CommandContext::RecordPresent(Image *image) {
#if !defined(NDEBUG)
  const auto frame = dynamic_cast<D3D12SwapChainImage *>(image);
  if (!frame || !list_)
    return false;
#else
  const auto frame = static_cast<D3D12SwapChainImage *>(image);
#endif
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = frame->NativeImage();
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
  list_->ResourceBarrier(1, &barrier);
  return true;
}

bool D3D12CommandContext::EndRecording() {
  return Check(list_->Close(), "CommandList::Close");
}

CommandContext::SubmitResult D3D12CommandContext::SubmitNative() {
#if !defined(NDEBUG)
  if (Frame() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
#endif
  auto frame = static_cast<D3D12SwapChainImage *>(Frame());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  failed_ = true;
  ID3D12CommandList *lists[] = {list_.Get()};
  device_->NativeQueue()->ExecuteCommandLists(1, lists);
  MarkSubmitted();
  const HRESULT present = frame ? frame->Present() : S_OK;
  value_ = device_->NextSerial();
  failed_ = !Check(device_->NativeQueue()->Signal(device_->Fence(), value_), "Queue::Signal");
  failed_ = !Check(present, "Swapchain::Present") || failed_;
  return failed_ ? SUBMIT_RESULT_FAILED_IN_FLIGHT : SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus D3D12CommandContext::WaitForProgress() {
  if (failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    if (device_->WaitIdle() || FAILED(device_->NativeDevice()->GetDeviceRemovedReason()))
      return SUBMISSION_STATUS_FAILED;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return SUBMISSION_STATUS_PENDING;
  }
  if (device_->Wait(value_, event_))
    return SUBMISSION_STATUS_COMPLETED;
  // A failed host wait does not prove that GPU resource use has ended.
  failed_ = true;
  return SUBMISSION_STATUS_PENDING;
}

}  // namespace grassland::rhi::backend::d3d12
