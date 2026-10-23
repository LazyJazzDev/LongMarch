#include <charconv>
#include <iostream>
#include <string>

#include "module.h"
#include "snowberg/tui/tui.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  std::string module;
  bool backend_specified = false;
  bool tui = false;
  bool headless = false;
  bool all = false;
  int frames = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--headless") {
      headless = true;
      continue;
    }
    if (arg == "--all") {
      all = true;
      continue;
    }
    if (arg == "--tui") {
      tui = true;
      continue;
    }
    if (arg == "--list") {
      std::cout << "window\ntriangle\ncube\ntexture\nblend\nnbody\nraytracing\nray_query\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout
          << "[--module window|triangle|cube|texture|blend|nbody|raytracing|ray_query | --tui | --all] [--backend auto|d3d12|vulkan|metal] [--headless] [--frames N]\nHeadless defaults to 3000 frames per scene; --all defaults to 300 frames with windows.\nMissing backend or module: select each missing option in the terminal menu; a sole enabled backend is automatic. --backend auto explicitly uses the default API.\n";
      return 0;
    }
    if (arg == "--module" && i + 1 < argc) {
      module = argv[++i];
      continue;
    }
    if (arg == "--frames" && i + 1 < argc) {
      std::string value = argv[++i];
      auto result = std::from_chars(value.data(), value.data() + value.size(), frames);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || frames < 0)
        return 1;
      continue;
    }
    if (arg == "--backend" && i + 1 < argc)
      arg = argv[++i];
    if (arg == "auto")
      api = BACKEND_API_DEFAULT;
    else if (arg == "d3d12")
      api = BACKEND_API_D3D12;
    else if (arg == "vulkan")
      api = BACKEND_API_VULKAN;
    else if (arg == "metal")
      api = BACKEND_API_METAL;
    else {
      std::cerr << "Unknown option: " << arg << '\n';
      return 1;
    }
    backend_specified = true;
  }
  if (all && (tui || !module.empty())) {
    std::cerr << "Use --all without --module or --tui\n";
    return 1;
  }
  if (tui && !module.empty()) {
    std::cerr << "Use either --module or --tui, not both\n";
    return 1;
  }
  if (!backend_specified) {
    const std::vector<BackendAPI> backends{
#if defined(LONGMARCH_D3D12_ENABLED)
        BACKEND_API_D3D12,
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
        BACKEND_API_VULKAN,
#endif
#if defined(LONGMARCH_METAL_ENABLED)
        BACKEND_API_METAL,
#endif
    };
    std::vector<snowberg::tui::MenuOption> options;
    for (const auto backend : backends) {
      switch (backend) {
        case BACKEND_API_D3D12:
          options.push_back({BackendAPIName(backend), "Microsoft | Windows"});
          break;
        case BACKEND_API_VULKAN:
          options.push_back({BackendAPIName(backend), "Khronos Group | Windows, Linux, macOS"});
          break;
        case BACKEND_API_METAL:
          options.push_back({BackendAPIName(backend), "Apple | macOS"});
          break;
      }
    }
    if (backends.empty()) {
      std::cerr << "No graphics backend is enabled in this build.\n";
      return 1;
    }
    if (backends.size() == 1) {
      api = backends.front();
    } else {
      snowberg::tui::Menu menu;
      const auto selected = menu.Select("RHI Hello - API Selection", options);
      if (!selected)
        return 0;
      api = backends[*selected];
    }
  }
  if (!frames) {
    if (headless)
      frames = 3000;
    else if (all)
      frames = 300;
  }
  if (all) {
    int result = 0;
    if (RunWindow(api, frames, headless) != 0) {
      std::cerr << "window: FAILED\n";
      result = 1;
    }
    if (RunTriangle(api, frames, headless) != 0) {
      std::cerr << "triangle: FAILED\n";
      result = 1;
    }
    if (RunCube(api, frames, headless) != 0) {
      std::cerr << "cube: FAILED\n";
      result = 1;
    }
    if (RunTexture(api, frames, headless) != 0) {
      std::cerr << "texture: FAILED\n";
      result = 1;
    }
    if (RunBlend(api, frames, headless) != 0) {
      std::cerr << "blend: FAILED\n";
      result = 1;
    }
    if (RunNBody(api, frames, headless) != 0) {
      std::cerr << "nbody: FAILED\n";
      result = 1;
    }
    if (RunRayTracing(api, frames, headless) != 0) {
      std::cerr << "raytracing: FAILED\n";
      result = 1;
    }
    if (RunRayQuery(api, frames, headless) != 0) {
      std::cerr << "ray_query: FAILED\n";
      result = 1;
    }
    return result;
  }
  if (module.empty()) {
    const std::vector<snowberg::tui::MenuOption> options{
        {"window", "Clear and present a window"},
        {"triangle", "Colored indexed triangle"},
        {"cube", "Rotating cube with depth and uniform binding"},
        {"texture", "Sampled XOR texture triangle"},
        {"blend", "Source-alpha blended triangles"},
        {"nbody", "Compute simulation and particle rendering"},
        {"raytracing", "Ray pipeline, triangles and procedural sphere (D3D12/Vulkan)"},
        {"ray_query", "Inline ray queries"},
    };
    snowberg::tui::Menu menu;
    const auto selected = menu.Select("RHI Hello - Module Selection", options);
    if (!selected)
      return 0;
    module = options[*selected].name;
  }
  if (module == "window")
    return RunWindow(api, frames, headless);
  if (module == "triangle")
    return RunTriangle(api, frames, headless);
  if (module == "cube")
    return RunCube(api, frames, headless);
  if (module == "texture")
    return RunTexture(api, frames, headless);
  if (module == "blend")
    return RunBlend(api, frames, headless);
  if (module == "nbody")
    return RunNBody(api, frames, headless);
  if (module == "raytracing")
    return RunRayTracing(api, frames, headless);
  if (module == "ray_query")
    return RunRayQuery(api, frames, headless);
  std::cerr << "Unknown module: " << module << '\n';
  return 1;
}
