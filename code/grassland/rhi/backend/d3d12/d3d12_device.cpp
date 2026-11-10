#include "grassland/rhi/backend/d3d12/d3d12_device.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_cuda_buffer.h"
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
bool D3D12Device::WaitNativeEvents(const std::vector<Event> &events, bool wait_all) const {
  std::vector<ID3D12Fence *> fences;
  std::vector<uint64_t> values;
  for (const auto &event : events) {
    fences.push_back(static_cast<D3D12Timeline *>(event.timeline)->NativeFence());
    values.push_back(event.value);
  }
  ComPtr<ID3D12Device1> device;
  if (!Check(device_.As(&device), "Query multi-fence wait interface"))
    return false;
  if (!Check(device->SetEventOnMultipleFenceCompletion(
                 fences.data(), values.data(), static_cast<UINT>(fences.size()),
                 wait_all ? D3D12_MULTIPLE_FENCE_WAIT_FLAG_ALL : D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY, nullptr),
             "Wait for timeline events"))
    return false;
  return SUCCEEDED(device_->GetDeviceRemovedReason());
}

void D3D12Device::InterruptCollector() {
  SetEvent(collector_shutdown_);
}

void D3D12Device::WaitForProgress(const std::vector<Event> &next) {
  std::array<HANDLE, 3> handles{collector_shutdown_};
  DWORD count = 1;
  for (const auto &event : next) {
    for (size_t index = 0; index < queues_.size(); ++index) {
      if (!queues_[index] || queues_[index]->GetTimeline() != event.timeline)
        continue;
      auto &handle = progress_events_[index];
      if (progress_values_[index] != event.value) {
        ResetEvent(handle);
        if (!Check(
                static_cast<D3D12Timeline *>(event.timeline)->NativeFence()->SetEventOnCompletion(event.value, handle),
                "Listen for queue progress")) {
          return;
        }
        progress_values_[index] = event.value;
      }
      handles[count++] = handle;
      break;
    }
  }
  WaitForMultipleObjects(count, handles.data(), FALSE, INFINITE);
}

bool D3D12Device::AcquireCommandResources(ComPtr<ID3D12CommandAllocator> &allocator,
                                          ComPtr<ID3D12GraphicsCommandList> &list) {
  {
    std::lock_guard<std::mutex> lock(command_pool_mutex_);
    if (!command_pool_.empty()) {
      std::tie(allocator, list) = std::move(command_pool_.back());
      command_pool_.pop_back();
    }
  }
  if (list) {
    if (!Check(allocator->Reset(), "Reset command allocator"))
      return false;
    return Check(list->Reset(allocator.Get(), nullptr), "Reset command list");
  }
  if (!Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
             "CreateCommandAllocator"))
    return false;
  return Check(
      device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
      "CreateCommandList");
}

void D3D12Device::RecycleCommandResources(ComPtr<ID3D12CommandAllocator> allocator,
                                          ComPtr<ID3D12GraphicsCommandList> list) {
  try {
    std::lock_guard<std::mutex> lock(command_pool_mutex_);
    command_pool_.emplace_back(std::move(allocator), std::move(list));
    return;
  } catch (const std::bad_alloc &) {
    // Recycling is optional: release the idle native objects if the pool cannot grow.
  }
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
  if (!Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&object->factory_)), "CreateDXGIFactory2"))
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
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    object->queues_[type] = ObjectPtr<Queue>::Adopt(D3D12Queue::Create(object.get(), type));
    if (!object->queues_[type])
      return nullptr;
  }
  object->collector_shutdown_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!object->collector_shutdown_) {
    ReportError(GetLastError(), "Create collector shutdown event");
    return nullptr;
  }
  for (auto &event : object->progress_events_) {
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) {
      ReportError(GetLastError(), "Create queue progress event");
      return nullptr;
    }
  }
  object->StartCollector();
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

bool D3D12Device::SupportsTearing() const {
  return tearing_;
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
  command_pool_.clear();
  for (auto &queue : queues_)
    queue.Reset();
  for (auto event : progress_events_)
    if (event)
      CloseHandle(event);
  if (collector_shutdown_)
    CloseHandle(collector_shutdown_);
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

#if defined(LONGMARCH_CUDA_RUNTIME)
ICUDABuffer *D3D12Device::CreateCUDABuffer(const BufferDesc &desc) {
  return D3D12CUDABuffer::Create(this, desc);
}
#endif

rhi::IWindow *D3D12Device::CreateGLFWWindow(const WindowDesc &desc) {
  return D3D12Window::Create(this, desc);
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
