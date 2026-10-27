#include <charconv>
#include <cmath>
#include <iostream>
#include <string>

#include "grassland/rhi/rhix/rhix.h"
#include "module.h"
#include "snowberg/tui/tui.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  std::string module;
  std::string capture_directory;
  bool backend_specified = false;
  bool tui = false;
  bool headless = false;
  bool all = false;
  int frames = 0;
  int device_number = 0;
  bool auto_device = false;
  const char *font_file = nullptr;
  float font_size = 20.0f;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device") {
      if (i + 1 >= argc) {
        std::cerr << "--device requires auto or a positive device number (starting at 1)\n";
        return 1;
      }
      const std::string value = argv[++i];
      if (value == "auto") {
        auto_device = true;
        continue;
      }
      auto_device = false;
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), device_number);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || device_number <= 0) {
        std::cerr << "--device requires auto or a positive device number (starting at 1)\n";
        return 1;
      }
      continue;
    }
    if (arg == "--save-last-frame") {
      if (i + 1 >= argc || !argv[i + 1][0]) {
        std::cerr << "--save-last-frame requires an output directory\n";
        return 1;
      }
      capture_directory = argv[++i];
      continue;
    }
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
      std::cout
          << "window\ntriangle\ncube\ntexture\nblend\nnbody\nraytracing\nray_query\nhdr\nimgui\nimgui_multi_window\n";
      return 0;
    }
    if (arg == "--help") {
      std::cout
          << "[--module window|triangle|cube|texture|blend|nbody|raytracing|ray_query|hdr|imgui|imgui_multi_window | --tui | --all] [--backend auto|d3d12|vulkan|metal] [--device auto|N] [--headless] [--frames N] [--save-last-frame DIR] [--font PATH] [--font-size PIXELS]\n--save-last-frame enables per-frame GPU image readback and saves the final frame at exit as PNG, native .bin pixels and .txt metadata.\nFont options apply to ImGui modules; defaults: built-in font, 20 pixels.\nHDR requires a window; --headless --all skips HDR.\nHeadless defaults to 3000 frames per scene; --all defaults to 300 frames with windows.\nMissing backend, device or module: select each missing option in the terminal menu; a sole backend or device is automatic. Device numbers start at 1, match the device menu, and are specific to the selected backend. --device auto skips the menu and uses RHI automatic device selection. --device auto|N applies to all modules, including --all. --backend auto explicitly uses the default API.\n";
      return 0;
    }
    if (arg == "--module" && i + 1 < argc) {
      module = argv[++i];
      continue;
    }
    if (arg == "--font") {
      if (i + 1 >= argc || !argv[i + 1][0]) {
        std::cerr << "--font requires a font file path\n";
        return 1;
      }
      font_file = argv[++i];
      continue;
    }
    if (arg == "--font-size") {
      if (i + 1 >= argc) {
        std::cerr << "--font-size requires a positive pixel size\n";
        return 1;
      }
      std::string value = argv[++i];
      auto result = std::from_chars(value.data(), value.data() + value.size(), font_size);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(font_size) ||
          font_size <= 0) {
        std::cerr << "--font-size requires a finite positive pixel size\n";
        return 1;
      }
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
      size_t initial_selection = 0;
      for (size_t i = 0; i < backends.size(); ++i) {
        if (backends[i] == BACKEND_API_DEFAULT) {
          initial_selection = i;
          break;
        }
      }
      snowberg::tui::Menu menu;
      const auto selected = menu.Select("RHIX Hello - API Selection", options, initial_selection);
      if (!selected)
        return 0;
      api = backends[*selected];
    }
  }
  grassland::rhix::Core core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  api = core.GetBackendAPI();
  int32_t device_index = 0;
  grassland::rhix::Device device;
  {
    const auto devices = core.GetPhysicalDevices();
    if (devices.empty()) {
      std::cerr << "No graphics device is available for " << BackendAPIName(api) << ".\n";
      return 1;
    }
    std::vector<snowberg::tui::MenuOption> options;
    for (const auto &device : devices) {
      options.push_back({"[" + std::to_string(options.size() + 1) + "] " + device.name,
                         PhysicalDeviceTypeName(device.type) +
                             " | dedicated: " + std::to_string(device.dedicated_memory / (1024 * 1024)) +
                             " MiB | shared: " + std::to_string(device.unified_memory / (1024 * 1024)) + " MiB"});
    }
    if (auto_device) {
      device = core.CreateDeviceByAutoSelection();
      if (!device)
        return 1;
    } else if (device_number > 0) {
      if (static_cast<size_t>(device_number) > devices.size()) {
        std::cerr << "Device number " << device_number << " is out of range; expected 1-" << devices.size() << ".\n";
        return 1;
      }
      device_index = device_number - 1;
    } else if (devices.size() > 1) {
      snowberg::tui::Menu menu;
      const auto selected = menu.Select("RHIX Hello - Device Selection", options);
      if (!selected)
        return 0;
      device_index = static_cast<int32_t>(*selected);
    }
    if (auto_device)
      std::cout << "Selected device (auto): " << device.Name() << " (" << BackendAPIName(api) << ")\n";
    else
      std::cout << "Selected device " << device_index + 1 << ": " << devices[device_index].name << " ("
                << BackendAPIName(api) << ")\n";
  }
  if (!device)
    device = core.CreateDevice(device_index);
  if (!device)
    return 1;
  if (!frames) {
    if (headless)
      frames = 3000;
    else if (all)
      frames = 300;
  }
  const ModuleOptions options{frames, headless, capture_directory, font_file, font_size};
  const std::vector<snowberg::tui::MenuOption> module_options{
      {"window", "Clear and present a window"},
      {"triangle", "Colored indexed triangle"},
      {"cube", "Rotating cube with depth and uniform binding"},
      {"texture", "Sampled XOR texture triangle"},
      {"blend", "Source-alpha blended triangles"},
      {"nbody", "Compute simulation and particle rendering"},
      {"raytracing", "Ray pipeline, triangles and procedural sphere (D3D12/Vulkan)"},
      {"ray_query", "Inline ray queries"},
      {"hdr", "HDR presentation (window only)"},
      {"imgui", "Interactive ImGui overlay"},
      {"imgui_multi_window", "Independent ImGui windows"},
  };
  auto run_module = [&](const std::string &name) {
    auto instance = CreateModule(name, core, device);
    if (!instance) {
      std::cerr << "Unknown module: " << name << '\n';
      return 1;
    }
    if (name == "hdr" && headless) {
      std::cerr << "hdr requires a window; --headless is unsupported.\n";
      return 1;
    }
    return instance->Run(options);
  };
  if (all) {
    int result = 0;
    for (const auto &entry : module_options) {
      if (entry.name == "hdr" && headless) {
        std::cout << "hdr: SKIPPED (requires a window)\n";
        continue;
      }
      if (run_module(entry.name) != 0) {
        std::cerr << entry.name << ": FAILED\n";
        result = 1;
      }
    }
    return result;
  }
  if (module.empty()) {
    snowberg::tui::Menu menu;
    const auto selected = menu.Select("RHIX Hello - Module Selection", module_options);
    if (!selected)
      return 0;
    module = module_options[*selected].name;
  }
  return run_module(module);
}
