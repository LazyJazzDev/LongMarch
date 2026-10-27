#include "module.h"

#include <utility>

ModuleBase::ModuleBase(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                       grassland::rhi::ObjPtr<grassland::rhi::IDevice> device)
    : core_(std::move(core)),
      device_(std::move(device)) {
}

int ModuleBase::Run(const ModuleOptions &options) {
  if (!core_ || !device_)
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
  return core_->GetBackendAPI();
}

std::unique_ptr<ModuleBase> CreateModule(std::string_view name,
                                         const grassland::rhi::ObjPtr<grassland::rhi::ICore> &core,
                                         const grassland::rhi::ObjPtr<grassland::rhi::IDevice> &device) {
  if (name == "window")
    return std::make_unique<WindowModule>(core, device);
  if (name == "triangle")
    return std::make_unique<TriangleModule>(core, device);
  if (name == "cube")
    return std::make_unique<CubeModule>(core, device);
  if (name == "texture")
    return std::make_unique<TextureModule>(core, device);
  if (name == "blend")
    return std::make_unique<BlendModule>(core, device);
  if (name == "nbody")
    return std::make_unique<NBodyModule>(core, device);
  if (name == "raytracing")
    return std::make_unique<RayTracingModule>(core, device);
  if (name == "ray_query")
    return std::make_unique<RayQueryModule>(core, device);
  if (name == "hdr")
    return std::make_unique<HDRModule>(core, device);
  if (name == "imgui")
    return std::make_unique<ImGuiModule>(core, device);
  if (name == "imgui_multi_window")
    return std::make_unique<ImGuiMultiWindowModule>(core, device);
  return {};
}
