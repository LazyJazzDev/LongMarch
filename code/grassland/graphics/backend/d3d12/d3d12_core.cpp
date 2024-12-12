#include "grassland/graphics/backend/d3d12/d3d12_core.h"

#include "grassland/graphics/backend/d3d12/d3d12_buffer.h"
#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {

D3D12Core::D3D12Core(const Settings &settings) : Core(settings) {
  d3d12::DXGIFactoryCreateHint hint{DebugEnabled()};
  d3d12::CreateDXGIFactory(hint, &dxgi_factory_);
}

D3D12Core::~D3D12Core() {
}

int D3D12Core::CreateBuffer(size_t size,
                            BufferType type,
                            double_ptr<Buffer> pp_buffer) {
  pp_buffer.construct<D3D12StaticBuffer>(size, this);
  return 0;
}

int D3D12Core::CreateImage(int width,
                           int height,
                           ImageFormat format,
                           double_ptr<Image> pp_image) {
  return 0;
}

int D3D12Core::CreateWindowObject(int width,
                                  int height,
                                  const std::string &title,
                                  double_ptr<Window> pp_window) {
  pp_window.construct<D3D12Window>(width, height, title, this);
  return 0;
}

int D3D12Core::GetPhysicalDeviceProperties(
    PhysicalDeviceProperties *p_physical_device_properties) {
  auto adapters = dxgi_factory_->EnumerateAdapters();
  if (adapters.empty()) {
    return 0;
  }

  if (p_physical_device_properties) {
    for (int i = 0; i < adapters.size(); ++i) {
      auto adapter = adapters[i];
      PhysicalDeviceProperties properties{};
      properties.name = adapter.Name();
      properties.score = adapter.Evaluate();
      properties.ray_tracing_support = adapter.SupportRayTracing();
      p_physical_device_properties[i] = properties;
    }
  }

  return adapters.size();
}

int D3D12Core::InitializeLogicalDevice(int device_index) {
  auto adapters = dxgi_factory_->EnumerateAdapters();

  if (device_index < 0 || device_index >= adapters.size()) {
    return -1;
  }

  dxgi_factory_->CreateDevice(
      d3d12::DeviceFeatureRequirement{
          adapters[device_index].SupportRayTracing()},
      device_index, &device_);

  device_name_ = adapters[device_index].Name();
  ray_tracing_support_ = adapters[device_index].SupportRayTracing();

  device_->CreateCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT, &command_queue_);
  command_allocators_.resize(FramesInFlight());
  command_lists_.resize(FramesInFlight());
  fences_.resize(FramesInFlight());

  for (int i = 0; i < FramesInFlight(); i++) {
    device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                    &command_allocators_[i]);
    device_->CreateFence(&fences_[i]);

    command_allocators_[i]->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              &command_lists_[i]);
  }

  device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  &single_time_allocator_);
  device_->CreateFence(&single_time_fence_);
  single_time_allocator_->CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            &single_time_command_list_);

  return 0;
}

void D3D12Core::WaitGPU() {
  single_time_fence_->Signal(command_queue_.get());
  single_time_fence_->Wait();
}

void D3D12Core::SingleTimeCommand(
    std::function<void(ID3D12GraphicsCommandList *)> command) {
  single_time_allocator_->ResetCommandRecord(single_time_command_list_.get());
  command(single_time_command_list_->Handle());
  single_time_command_list_->Handle()->Close();

  ID3D12CommandList *command_lists[] = {single_time_command_list_->Handle()};
  command_queue_->Handle()->ExecuteCommandLists(1, command_lists);
  single_time_fence_->Signal(command_queue_.get());
  single_time_fence_->Wait();
}

}  // namespace grassland::graphics::backend
