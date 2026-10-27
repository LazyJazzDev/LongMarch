#include "module.h"
#include "modules/blend/module.h"
#include "modules/cube/module.h"
#include "modules/hdr/module.h"
#include "modules/imgui/module.h"
#include "modules/imgui_multi_window/module.h"
#include "modules/nbody/module.h"
#include "modules/ray_query/module.h"
#include "modules/raytracing/module.h"
#include "modules/texture/module.h"
#include "modules/triangle/module.h"
#include "modules/window/module.h"

std::unique_ptr<ModuleBase> CreateModule(std::string_view name,
                                         const grassland::rhix::Core &core,
                                         const grassland::rhix::Device &device) {
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
