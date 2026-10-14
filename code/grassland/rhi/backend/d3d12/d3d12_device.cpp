#include "grassland/rhi/backend/d3d12/d3d12_device.h"

#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Device::BackendLifetime::BackendLifetime(ComPtr<ID3D12Device> device) : device_(std::move(device)) {
}

D3D12Device::BackendLifetime::~BackendLifetime() {
  if (idle_event_)
    CloseHandle(idle_event_);
}

bool D3D12Device::BackendLifetime::Initialize() {
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_)), "CreateDXGIFactory2") ||
      !Check(device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue_)), "CreateCommandQueue") ||
      !Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence"))
    return false;
  idle_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!idle_event_) {
    ReportError(GetLastError(), "Create fence event failed.");
    return false;
  }
  ComPtr<IDXGIFactory5> factory5;
  BOOL tearing = FALSE;
  if (SUCCEEDED(factory_.As(&factory5)) &&
      SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
    tearing_ = tearing;
  return true;
}

ID3D12Device *D3D12Device::BackendLifetime::Device() const {
  return device_.Get();
}

IDXGIFactory4 *D3D12Device::BackendLifetime::Factory() const {
  return factory_.Get();
}

ID3D12CommandQueue *D3D12Device::BackendLifetime::Queue() const {
  return queue_.Get();
}

ID3D12Fence *D3D12Device::BackendLifetime::Fence() const {
  return fence_.Get();
}

bool D3D12Device::BackendLifetime::SupportsTearing() const {
  return tearing_;
}

std::mutex &D3D12Device::BackendLifetime::QueueMutex() {
  return mutex_;
}

uint64_t D3D12Device::BackendLifetime::NextSerial() {
  return ++serial_;
}

bool D3D12Device::BackendLifetime::Wait(uint64_t value, HANDLE event) const {
  if (fence_->GetCompletedValue() == UINT64_MAX)
    return false;
  if (fence_->GetCompletedValue() >= value)
    return true;
  if (!Check(fence_->SetEventOnCompletion(value, event), "SetEventOnCompletion"))
    return false;
  DWORD result;
  do {
    result = WaitForSingleObject(event, 50);
    if (FAILED(device_->GetDeviceRemovedReason()))
      return false;
  } while (result == WAIT_TIMEOUT);
  return result == WAIT_OBJECT_0 && fence_->GetCompletedValue() != UINT64_MAX;
}

bool D3D12Device::BackendLifetime::WaitIdle() {
  auto value = NextSerial();
  return Check(queue_->Signal(fence_.Get(), value), "Signal idle fence") && Wait(value, idle_event_);
}

D3D12Device::D3D12Device(PhysicalDevice info, const ObjectPtr<ICore> &core, ComPtr<ID3D12Device> device)
    : IDevice(std::move(info), core),
      lifetime_(std::make_shared<BackendLifetime>(std::move(device))) {
  if (!lifetime_->Initialize())
    lifetime_.reset();
}

D3D12Device::~D3D12Device() {
  ShutdownQueues();
}

bool D3D12Device::IsInitialized() const {
  return static_cast<bool>(lifetime_);
}

const std::shared_ptr<D3D12Device::BackendLifetime> &D3D12Device::NativeLifetime() const {
  return lifetime_;
}

const void *D3D12Device::NativeIdentity() const {
  return lifetime_.get();
}

std::unique_ptr<IWindow> D3D12Device::CreateWindow(const WindowDesc &desc) {
  auto window = std::make_unique<D3D12Window>(lifetime_);
  return window->Initialize(desc) ? std::unique_ptr<IWindow>(std::move(window)) : nullptr;
}

std::unique_ptr<IQueue> D3D12Device::CreateQueue(QueueType type) {
  return std::make_unique<D3D12Queue>(ObjectPtr<D3D12Device>::Retain(this), type);
}

}  // namespace grassland::rhi::backend::d3d12
