#include "grassland/rhi/backend/d3d12/d3d12_device.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"
#include "grassland/util/log.h"

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
  auto object = std::unique_ptr<D3D12Device>(new D3D12Device(std::move(info), core, std::move(device)));
  if (SUCCEEDED(object->device_.As(&object->debug_messages_))) {
    auto callback = [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR description,
                       void *) {
      if (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || severity == D3D12_MESSAGE_SEVERITY_ERROR)
        LogError("D3D12 [{}]: {}", static_cast<int>(id), description);
      else if (severity == D3D12_MESSAGE_SEVERITY_WARNING)
        LogWarning("D3D12 [{}]: {}", static_cast<int>(id), description);
    };
    if (!Check(object->debug_messages_->RegisterMessageCallback(callback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr,
                                                                &object->debug_callback_cookie_),
               "Register D3D12 debug message callback"))
      return nullptr;
  } else {
    ComPtr<ID3D12InfoQueue> messages;
    if (SUCCEEDED(object->device_.As(&messages)))
      LogWarning(
          "D3D12 debug message callbacks are unavailable; validation messages remain in the native debug output.");
  }
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&object->factory_)), "CreateDXGIFactory2"))
    return nullptr;
  if (!Check(object->device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&object->queue_)), "CreateCommandQueue"))
    return nullptr;
  if (!Check(object->device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&object->fence_)), "CreateFence"))
    return nullptr;
  if (!Check(object->device_.As(&object->device1_), "Query ID3D12Device1") ||
      !Check(object->device_->CreateCommandQueue(&desc, IID_PPV_ARGS(&object->transfer_queue_)),
             "Create transfer queue") ||
      !Check(object->device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&object->transfer_fence_)),
             "Create transfer timeline"))
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
  ComPtr<IDXGIFactory5> factory5;
  BOOL tearing = FALSE;
  if (SUCCEEDED(object->factory_.As(&factory5)) &&
      SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
    object->tearing_ = tearing;
  return object.release();
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

ID3D12CommandQueue *D3D12Device::NativeQueue(QueueType type) const {
  return type == QUEUE_TYPE_TRANSFER ? transfer_queue_.Get() : queue_.Get();
}

ID3D12Fence *D3D12Device::Fence(QueueType type) const {
  return type == QUEUE_TYPE_TRANSFER ? transfer_fence_.Get() : fence_.Get();
}

bool D3D12Device::SupportsTearing() const {
  return tearing_;
}

std::mutex &D3D12Device::QueueMutex(QueueType type) {
  return type == QUEUE_TYPE_TRANSFER ? transfer_mutex_ : mutex_;
}

uint64_t D3D12Device::NextSerial(QueueType type) {
  return type == QUEUE_TYPE_TRANSFER ? ++transfer_serial_ : ++serial_;
}

bool D3D12Device::Wait(uint64_t value, HANDLE event, QueueType type) const {
  auto fence = Fence(type);
  if (fence->GetCompletedValue() == UINT64_MAX)
    return false;
  if (fence->GetCompletedValue() >= value)
    return true;
  if (!Check(fence->SetEventOnCompletion(value, event), "SetEventOnCompletion"))
    return false;
  DWORD result;
  do {
    result = WaitForSingleObject(event, 50);
    if (FAILED(device_->GetDeviceRemovedReason()))
      return false;
  } while (result == WAIT_TIMEOUT);
  return result == WAIT_OBJECT_0 && fence->GetCompletedValue() != UINT64_MAX;
}

bool D3D12Device::WaitIdle(QueueType type) {
  auto value = NextSerial(type);
  return Check(NativeQueue(type)->Signal(Fence(type), value), "Signal idle fence") &&
         WaitTimelines({{type, value}}, true);
}

D3D12Device::D3D12Device(PhysicalDevice info, Core *core, ComPtr<ID3D12Device> device)
    : Device(std::move(info), core, BACKEND_API_D3D12),
      upload_pool_(this, TEMPORAL_BUFFER_UPLOAD),
      download_pool_(this, TEMPORAL_BUFFER_DOWNLOAD),
      scratch_pool_(this),
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
  download_pool_.Clear();
  scratch_pool_.Clear();
  for (auto &entry : command_pool_)
    CloseHandle(std::get<2>(entry));
  command_pool_.clear();
  if (debug_callback_cookie_)
    debug_messages_->UnregisterMessageCallback(debug_callback_cookie_);
}

rhi::IComputeProgram *D3D12Device::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return D3D12ComputeProgram::Create(this, desc);
}

rhi::IProgram *D3D12Device::CreateProgram(const ProgramDesc &desc) {
  return D3D12Program::Create(this, desc);
}

D3D12TemporalBufferPool &D3D12Device::UploadPool() {
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
  if (type != QUEUE_TYPE_GRAPHICS && type != QUEUE_TYPE_TRANSFER)
    return nullptr;
  return new D3D12Queue(this, type);
}

ISampler *D3D12Device::CreateSampler(const SamplerDesc &desc) {
  return D3D12Sampler::Create(this, desc);
}

IBindingGroup *D3D12Device::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return D3D12BindingGroup::Create(this, layout);
}

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
D3D12ScratchPool &D3D12Device::ScratchPool() {
  return scratch_pool_;
}
}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
IBottomLevelAccelerationStructure *D3D12Device::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return D3D12BottomLevelAccelerationStructure::Create(this, desc);
}

ITopLevelAccelerationStructure *D3D12Device::CreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &desc) {
  return D3D12TopLevelAccelerationStructure::Create(this, desc);
}

IRayTracingProgram *D3D12Device::CreateRayTracingProgram(const RayTracingProgramDesc &desc) {
  if (!SupportRayTracingPipeline())
    return nullptr;
  return D3D12RayTracingProgram::Create(this, desc);
}

D3D12TemporalBufferPool &D3D12Device::DownloadPool() {
  return download_pool_;
}
}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
bool D3D12Device::SharesNativeQueue(QueueType a, QueueType b) const {
  return a == b;
}

bool D3D12Device::QueryTimeline(QueueType type, uint64_t &value) const {
  value = Fence(type)->GetCompletedValue();
  return value != UINT64_MAX;
}

bool D3D12Device::WaitTimelines(const std::vector<TimelinePoint> &points, bool wait_all) const {
  std::array<uint64_t, 2> targets{};
  std::array<bool, 2> present{};
  for (auto p : points) {
    if (!present[p.queue])
      targets[p.queue] = p.value;
    else
      targets[p.queue] = wait_all ? std::max(targets[p.queue], p.value) : std::min(targets[p.queue], p.value);
    present[p.queue] = true;
  }
  std::array<ID3D12Fence *, 2> fences{};
  std::array<uint64_t, 2> values{};
  UINT count = 0;
  for (int i = 0; i < 2; ++i)
    if (present[i]) {
      fences[count] = Fence(static_cast<QueueType>(i));
      values[count++] = targets[i];
    }
  return SUCCEEDED(device1_->SetEventOnMultipleFenceCompletion(
             fences.data(), values.data(), count,
             wait_all ? D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL : D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, nullptr)) &&
         SUCCEEDED(device_->GetDeviceRemovedReason());
}

void D3D12Device::WakeCollector() {
  std::vector<TimelinePoint> wake;
  for (auto type : {QUEUE_TYPE_GRAPHICS, QUEUE_TYPE_TRANSFER}) {
    std::lock_guard<std::mutex> lock(QueueMutex(type));
    if (NativeQueue(type) && Fence(type)) {
      auto value = NextSerial(type);
      if (Check(NativeQueue(type)->Signal(Fence(type), value), "Signal collector shutdown"))
        wake.push_back({type, value});
    }
  }
  if (!wake.empty())
    WaitTimelines(wake, true);
}
}  // namespace grassland::rhi::backend::d3d12
