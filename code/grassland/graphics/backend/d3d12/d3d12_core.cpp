#include "grassland/graphics/backend/d3d12/d3d12_core.h"

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
  return 0;
}

int D3D12Core::CreateImage(int width,
                           int height,
                           ImageFormat format,
                           double_ptr<Image> pp_image) {
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

int D3D12Core::InitialLogicalDevice(int device_index) {
  return 0;
}

}  // namespace grassland::graphics::backend
