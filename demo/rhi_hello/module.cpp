#include "module.h"

#include <iostream>

int ModuleBase::Run(grassland::rhi::BackendAPI api, int32_t device_index, const ModuleOptions &options) {
  device_ = {};
  core_ = {};
  api_ = api;
  core_ = grassland::rhi::CreateCore(api);
  if (!core_)
    return 1;
  const auto devices = core_->GetPhysicalDevices();
  if (device_index < 0 || static_cast<size_t>(device_index) >= devices.size()) {
    std::cerr << "Selected device is no longer available.\n";
    return 1;
  }
  device_ = core_->CreateDevice(device_index);
  if (!device_)
    return 1;
  return RunImpl(options);
}

grassland::rhi::ICore *ModuleBase::GetCore() const {
  return core_.Get();
}

grassland::rhi::IDevice *ModuleBase::GetDevice() const {
  return device_.Get();
}

grassland::rhi::BackendAPI ModuleBase::GetBackendAPI() const {
  return api_;
}

std::unique_ptr<ModuleBase> CreateModule(std::string_view name) {
  if (name == "window")
    return std::make_unique<WindowModule>();
  if (name == "triangle")
    return std::make_unique<TriangleModule>();
  if (name == "cube")
    return std::make_unique<CubeModule>();
  if (name == "texture")
    return std::make_unique<TextureModule>();
  if (name == "blend")
    return std::make_unique<BlendModule>();
  if (name == "nbody")
    return std::make_unique<NBodyModule>();
  if (name == "raytracing")
    return std::make_unique<RayTracingModule>();
  if (name == "ray_query")
    return std::make_unique<RayQueryModule>();
  if (name == "hdr")
    return std::make_unique<HDRModule>();
  if (name == "imgui")
    return std::make_unique<ImGuiModule>();
  if (name == "imgui_multi_window")
    return std::make_unique<ImGuiMultiWindowModule>();
  return {};
}
