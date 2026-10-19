#include "grassland/rhi/backend/d3d12/d3d12_core.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/util/string_convert.h"

namespace grassland::rhi::backend::d3d12 {
namespace {

bool DescribeDevice(IDXGIAdapter1 *adapter, ID3D12Device *device, PhysicalDevice &info) {
  DXGI_ADAPTER_DESC1 desc{};
  const HRESULT result = adapter->GetDesc1(&desc);
  if (FAILED(result)) {
    ReportError(result, "IDXGIAdapter1::GetDesc1 failed.");
    return false;
  }
  info = {};
  info.name = WStringToString(desc.Description);
  info.dedicated_memory = desc.DedicatedVideoMemory;
  info.unified_memory = desc.SharedSystemMemory;
  D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
  const bool unified =
      SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture))) &&
      architecture.UMA;
  info.type = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ? PHYSICAL_DEVICE_TYPE_SOFTWARE
              : unified                                 ? PHYSICAL_DEVICE_TYPE_INTEGRATED
                                                        : PHYSICAL_DEVICE_TYPE_DEDICATED;
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
  if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options)))) {
    info.ray_tracing_pipeline_feature = options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
    info.ray_query_feature = options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
  }
  // Geometry shaders are part of the feature-level 11_0 baseline used below.
  info.geometry_shader_feature = true;
  return true;
}

}  // namespace

std::unique_ptr<D3D12Core> D3D12Core::CreateD3D12Core() {
  auto core = std::make_unique<D3D12Core>();
  HRESULT result = CreateDXGIFactory2(0, IID_PPV_ARGS(&core->factory_));
  if (FAILED(result)) {
    ReportError(result, "CreateDXGIFactory2 failed.");
    return {};
  }
  for (UINT i = 0;; ++i) {
    ComPtr<IDXGIAdapter1> adapter;
    result = core->factory_->EnumAdapters1(i, &adapter);
    if (result == DXGI_ERROR_NOT_FOUND)
      break;
    if (FAILED(result)) {
      ReportError(result, "DXGI adapter enumeration failed.");
      return {};
    }
    ComPtr<ID3D12Device> probe;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&probe))))
      continue;
    PhysicalDevice info{};
    if (!DescribeDevice(adapter.Get(), probe.Get(), info))
      return {};
    core->adapters_.push_back({std::move(adapter), std::move(info)});
  }
  return core;
}

std::vector<PhysicalDevice> D3D12Core::GetPhysicalDevices() const {
  std::vector<PhysicalDevice> devices;
  for (const auto &adapter : adapters_)
    devices.push_back(adapter.info);
  return devices;
}

rhi::IDevice *D3D12Core::CreateDevice(int32_t device_idx) {
  try {
#if !defined(NDEBUG)
    if (device_idx < 0 || static_cast<size_t>(device_idx) >= adapters_.size()) {
      ReportError(-1, "Physical device ID is out of range.");
      return {};
    }
#endif
    const auto &adapter = adapters_[device_idx];
    ComPtr<ID3D12Device> device;
    const HRESULT result = D3D12CreateDevice(adapter.native.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(result)) {
      ReportError(result, "D3D12CreateDevice failed.");
      return {};
    }
    // D3D12 exposes supported capabilities automatically; there is no Vulkan-style feature mask.
    PhysicalDevice info{};
    if (!DescribeDevice(adapter.native.Get(), device.Get(), info))
      return {};
    auto result_device = std::make_unique<D3D12Device>(std::move(info), this, std::move(device));
    if (!result_device->IsInitialized())
      return {};
    return result_device.release();
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

}  // namespace grassland::rhi::backend::d3d12
