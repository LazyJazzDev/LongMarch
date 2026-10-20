#include "grassland/rhi/backend/d3d12/d3d12_device.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

// Match the Windows SDK declarations used by the backend.
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include <D3D12MemAlloc.h>
#undef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED

namespace grassland::rhi::backend::d3d12 {

bool D3D12Device::AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator,
                                          ComPtr<ID3D12GraphicsCommandList> &list,
                                          HANDLE &event) {
  {
    std::lock_guard<std::mutex> lock(command_pool_mutex_);
    if (!command_pool_.empty()) {
      std::tie(allocator, list, event) = std::move(command_pool_.back());
      command_pool_.pop_back();
    }
  }
  if (list) {
    if (!Check(allocator->Reset(), "Reset command allocator"))
      return false;
    if (!ResetEvent(event))
      return false;
    return Check(list->Reset(allocator.Get(), nullptr), "Reset command list");
  }
  event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event)
    return false;
  if (!Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
             "CreateCommandAllocator"))
    return false;
  return Check(
      device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
      "CreateCommandList");
}

void D3D12Device::RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator,
                                          ComPtr<ID3D12GraphicsCommandList> list,
                                          HANDLE event) {
  try {
    std::lock_guard<std::mutex> lock(command_pool_mutex_);
    command_pool_.emplace_back(std::move(allocator), std::move(list), event);
    return;
  } catch (const std::bad_alloc &) {
    // Recycling is optional: release the idle native objects if the pool cannot grow.
  }
  CloseHandle(event);
}

D3D12Device *D3D12Device::Create(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device) {
  try {
    auto object = std::unique_ptr<D3D12Device>(new D3D12Device(std::move(info), core, std::move(device)));
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&object->factory_)), "CreateDXGIFactory2"))
      return nullptr;
    if (!Check(object->device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&object->queue_)), "CreateCommandQueue"))
      return nullptr;
    if (!Check(object->device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&object->fence_)), "CreateFence"))
      return nullptr;
    ComPtr<IDXGIAdapter> adapter;
    if (!Check(object->factory_->EnumAdapterByLuid(object->device_->GetAdapterLuid(), IID_PPV_ARGS(&adapter)),
               "Find allocation adapter"))
      return nullptr;
    D3D12MA::ALLOCATOR_DESC memory{};
    memory.pDevice = object->device_.Get();
    memory.pAdapter = adapter.Get();
    if (!Check(D3D12MA::CreateAllocator(&memory, &object->allocator_), "Create D3D12MA allocator"))
      return nullptr;
    object->idle_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!object->idle_event_) {
      ReportError(GetLastError(), "Create fence event failed.");
      return nullptr;
    }
    ComPtr<IDXGIFactory5> factory5;
    BOOL tearing = FALSE;
    if (SUCCEEDED(object->factory_.As(&factory5)) &&
        SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
      object->tearing_ = tearing;
    return object.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "D3D12Device creation ran out of memory.");
    return nullptr;
  }
}

D3D12TransientDescriptorPool &D3D12Device::ResourceDescriptors() {
  return resource_descriptors_;
}

D3D12TransientDescriptorPool &D3D12Device::SamplerDescriptors() {
  return sampler_descriptors_;
}

D3D12DescriptorPool &D3D12Device::DSVPool() {
  return dsv_pool_;
}

D3D12DescriptorPool &D3D12Device::RTVPool() {
  return rtv_pool_;
}

D3D12MA::Allocator *D3D12Device::Allocator() const {
  return allocator_.Get();
}

ID3D12Device *D3D12Device::NativeDevice() const {
  return device_.Get();
}

D3D12DescriptorPool &D3D12Device::ResourceDescriptorStorage() {
  return resource_descriptor_storage_;
}

D3D12DescriptorPool &D3D12Device::SamplerDescriptorStorage() {
  return sampler_descriptor_storage_;
}

IDXGIFactory4 *D3D12Device::Factory() const {
  return factory_.Get();
}

ID3D12CommandQueue *D3D12Device::NativeQueue() const {
  return queue_.Get();
}

ID3D12Fence *D3D12Device::Fence() const {
  return fence_.Get();
}

bool D3D12Device::SupportsTearing() const {
  return tearing_;
}

std::mutex &D3D12Device::QueueMutex() {
  return mutex_;
}

uint64_t D3D12Device::NextSerial() {
  return ++serial_;
}

bool D3D12Device::Wait(uint64_t value, HANDLE event) const {
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

bool D3D12Device::WaitIdle() {
  auto value = NextSerial();
  return Check(queue_->Signal(fence_.Get(), value), "Signal idle fence") && Wait(value, idle_event_);
}

D3D12Device::D3D12Device(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device)
    : Device(std::move(info), core),
      upload_pool_(this),
      device_(std::move(device)),
      rtv_pool_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
      dsv_pool_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV),
      resource_descriptors_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
      sampler_descriptors_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER),
      resource_descriptor_storage_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096),
      sampler_descriptor_storage_(device_.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 256) {
}

D3D12Device::~D3D12Device() {
  ShutdownQueues();
  upload_pool_.Clear();
  for (auto &entry : command_pool_)
    CloseHandle(std::get<2>(entry));
  command_pool_.clear();
  if (idle_event_)
    CloseHandle(idle_event_);
}

rhi::IProgram *D3D12Device::CreateProgram(const ProgramDesc &desc) {
  try {
    return new D3D12Program(this, desc);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Program allocation failed.");
    return nullptr;
  }
}

D3D12UploadPool &D3D12Device::UploadPool() {
  return upload_pool_;
}

rhi::IImage *D3D12Device::CreateImage(const ImageDesc &desc) {
  return D3D12AllocatedImage::Create(this, desc);
}

rhi::IBuffer *D3D12Device::CreateBuffer(const BufferDesc &desc) {
  return D3D12AllocatedBuffer::Create(this, desc);
}

rhi::IWindow *D3D12Device::CreateGLFWWindow(const WindowDesc &desc) {
  return D3D12Window::Create(this, desc);
}

rhi::ICommandContext *D3D12Device::CreateCommandContext() {
  return D3D12CommandContext::Create(this);
}

rhi::IQueue *D3D12Device::GetQueue(QueueType type) {
  try {
    return new D3D12Queue(this, type);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Queue creation ran out of memory.");
    return nullptr;
  }
}

IBindingGroup *D3D12Device::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return D3D12BindingGroup::Create(this, layout);
}

}  // namespace grassland::rhi::backend::d3d12
