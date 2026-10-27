#include "module.h"

#include <utility>

ModuleBase::ModuleBase(grassland::rhi::BackendAPI api,
                       grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                       grassland::rhi::ObjPtr<grassland::rhi::IDevice> device)
    : api_(api),
      core_(std::move(core)),
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
  return api_;
}

std::unique_ptr<ModuleBase> CreateModule(std::string_view name,
                                         grassland::rhi::BackendAPI api,
                                         const grassland::rhi::ObjPtr<grassland::rhi::ICore> &core,
                                         const grassland::rhi::ObjPtr<grassland::rhi::IDevice> &device) {
  if (name == "window")
    return std::make_unique<WindowModule>(api, core, device);
  if (name == "triangle")
    return std::make_unique<TriangleModule>(api, core, device);
  if (name == "cube")
    return std::make_unique<CubeModule>(api, core, device);
  if (name == "texture")
    return std::make_unique<TextureModule>(api, core, device);
  if (name == "blend")
    return std::make_unique<BlendModule>(api, core, device);
  if (name == "nbody")
    return std::make_unique<NBodyModule>(api, core, device);
  if (name == "raytracing")
    return std::make_unique<RayTracingModule>(api, core, device);
  if (name == "ray_query")
    return std::make_unique<RayQueryModule>(api, core, device);
  if (name == "hdr")
    return std::make_unique<HDRModule>(api, core, device);
  if (name == "imgui")
    return std::make_unique<ImGuiModule>(api, core, device);
  if (name == "imgui_multi_window")
    return std::make_unique<ImGuiMultiWindowModule>(api, core, device);
  return {};
}
