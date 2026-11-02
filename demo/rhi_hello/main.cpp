#include <charconv>
#include <iostream>
#include <string>
#include <vector>

#include "grassland/rhi/rhi.h"
#include "module.h"
#include "snowberg/tui/tui.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;

  struct Demo {
    snowberg::tui::MenuOption option;
    int (*run)(IDevice *);
  };

  const std::vector<Demo> demos{
      {{"window", "Clear and present a window"}, RunWindow},
      {{"triangle", "Colored indexed triangle"}, RunTriangle},
      {{"cube", "Rotating cube with depth and uniform binding"}, RunCube},
      {{"texture", "Sampled XOR texture triangle"}, RunTexture},
      {{"blend", "Source-alpha blended triangles"}, RunBlend},
      {{"nbody", "Compute simulation and particle rendering"}, RunNBody},
      {{"raytracing", "Ray tracing pipeline (requires device support)"}, RunRayTracing},
      {{"ray_query", "Inline ray queries (requires device support)"}, RunRayQuery},
      {{"hdr", "HDR color spaces and reference white"}, RunHDR},
      {{"imgui", "Interactive ImGui overlay"}, RunImGui},
      {{"imgui_multi_window", "Independent ImGui windows"}, RunImGuiMultiWindow},
  };
  std::string backend, module, device_option;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "[--backend auto|d3d12|vulkan|metal] [--module NAME] [--device auto|N] [--list]\n"
                   "Select missing choices from the terminal menu. Device numbers start at 1.\n"
                   "Each demo runs until its windows are closed.\n";
      return 0;
    }
    if (arg == "--list") {
      for (const auto &demo : demos)
        std::cout << demo.option.name << '\n';
      return 0;
    }
    if ((arg == "--backend" || arg == "--module" || arg == "--device") && i + 1 < argc && argv[i + 1][0]) {
      auto &value = arg == "--backend" ? backend : arg == "--module" ? module : device_option;
      value = argv[++i];
    } else {
      std::cerr << "Unknown option or missing value: " << arg << '\n';
      return 1;
    }
  }
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
  if (backends.empty()) {
    std::cerr << "No graphics backend is enabled in this build.\n";
    return 1;
  }
  BackendAPI api = BACKEND_API_DEFAULT;
  snowberg::tui::Menu menu;
  if (backend.empty()) {
    if (backends.size() == 1) {
      api = backends.front();
    } else {
      std::vector<snowberg::tui::MenuOption> options;
      size_t initial = 0;
      for (size_t i = 0; i < backends.size(); ++i) {
        options.push_back({BackendAPIName(backends[i]), "Graphics API"});
        if (backends[i] == BACKEND_API_DEFAULT)
          initial = i;
      }
      const auto selected = menu.Select("RHI Hello - API Selection", options, initial);
      if (!selected)
        return 0;
      api = backends[*selected];
    }
  } else if (backend == "auto") {
    api = BACKEND_API_DEFAULT;
  } else if (backend == "d3d12") {
    api = BACKEND_API_D3D12;
  } else if (backend == "vulkan") {
    api = BACKEND_API_VULKAN;
  } else if (backend == "metal") {
    api = BACKEND_API_METAL;
  } else {
    std::cerr << "Unknown backend: " << backend << '\n';
    return 1;
  }
  size_t demo_index = demos.size();
  if (module.empty()) {
    std::vector<snowberg::tui::MenuOption> options;
    for (const auto &demo : demos)
      options.push_back(demo.option);
    const auto selected = menu.Select("RHI Hello - Module Selection", options);
    if (!selected)
      return 0;
    demo_index = *selected;
  } else {
    for (size_t i = 0; i < demos.size(); ++i)
      if (demos[i].option.name == module)
        demo_index = i;
    if (demo_index == demos.size()) {
      std::cerr << "Unknown module: " << module << '\n';
      return 1;
    }
  }
  ObjPtr<ICore> core = CreateCore(api);
  if (!core) {
    std::cerr << "Could not create backend: " << BackendAPIName(api) << '\n';
    return 1;
  }
  ObjPtr<IDevice> device;
  if (device_option == "auto") {
    device = core->CreateDeviceByAutoSelection();
  } else {
    const auto devices = core->GetPhysicalDevices();
    if (devices.empty()) {
      std::cerr << "No physical devices available.\n";
      return 1;
    }
    size_t selected = 0;
    if (!device_option.empty()) {
      size_t number = 0;
      const auto result = std::from_chars(device_option.data(), device_option.data() + device_option.size(), number);
      if (result.ec != std::errc{} || result.ptr != device_option.data() + device_option.size() || number == 0 ||
          number > devices.size()) {
        std::cerr << "Device must be auto or a number from 1 to " << devices.size() << ".\n";
        return 1;
      }
      selected = number - 1;
    } else if (devices.size() > 1) {
      std::vector<snowberg::tui::MenuOption> options;
      for (const auto &info : devices)
        options.push_back({info.name, PhysicalDeviceTypeName(info.type)});
      const auto choice = menu.Select("RHI Hello - Device Selection", options);
      if (!choice)
        return 0;
      selected = *choice;
    }
    device = core->CreateDevice(static_cast<int32_t>(selected));
  }
  if (!device) {
    std::cerr << "Could not create the selected device.\n";
    return 1;
  }
  std::cout << "Backend: " << BackendAPIName(api) << "\nDevice: " << device->Name()
            << "\nDemo: " << demos[demo_index].option.name << '\n';
  // The selected device outlives all local resources owned by the demo function.
  return demos[demo_index].run(device.Get());
}
