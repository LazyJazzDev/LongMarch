#include <array>
#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>
#include <type_traits>

#include "module.h"
#include "modules/blend/module.h"
#include "modules/cube/module.h"
#include "modules/external_shader/module.h"
#include "modules/hdr/module.h"
#include "modules/ray_query/module.h"
#include "modules/raytracing/module.h"
#include "modules/resize/module.h"
#include "modules/rt_multi_shader_group/module.h"
#include "modules/sdr_sample/module.h"
#include "modules/texture/module.h"
#include "modules/triangle/module.h"
#include "modules/window/module.h"
#include "snowberg/tui/tui.h"

namespace graphics_hello {
namespace {
using grassland::graphics::BackendAPI;

struct ModuleInfo {
  const char *name;
  const char *description;
  std::unique_ptr<Module> (*create)(BackendAPI, bool);
  bool supports_headless{};
};

template <typename T>
std::unique_ptr<Module> CreateModule(BackendAPI api, bool headless) {
  if constexpr (std::is_constructible_v<T, BackendAPI, bool>)
    return std::make_unique<T>(api, headless);
  else
    return std::make_unique<T>(api);
}

const std::array<ModuleInfo, 12> modules{{
    {"triangle", "Colored triangle", CreateModule<triangle::ModuleTriangle>, true},
    {"blend", "Alpha blending", CreateModule<blend::ModuleBlend>, true},
    {"cube", "Rotating cube", CreateModule<cube::ModuleCube>, true},
    {"texture", "Textured triangle", CreateModule<texture::ModuleTexture>, true},
    {"resize", "Resizable window", CreateModule<resize::ModuleResize>},
    {"hdr", "HDR gradient and SDR reference (H toggles HDR/SDR)", CreateModule<hdr::ModuleHDR>},
    {"sdr_sample", "SDR sampling", CreateModule<sdr_sample::ModuleSDRSample>},
    {"raytracing", "Ray tracing pipeline (unavailable on Metal)", CreateModule<raytracing::ModuleRayTracing>},
    {"rt_multi_shader_group", "Triangle + procedural sphere (requires RT pipelines)",
     CreateModule<rt_multi_shader_group::ModuleRTMultiShaderGroup>},
    {"external_shader", "RT scene with shaders loaded from assets (requires RT pipelines)",
     CreateModule<external_shader::ModuleExternalShader>},
    {"ray_query", "Compute ray queries (requires device/backend support)", CreateModule<ray_query::ModuleRayQuery>},
    {"window", "Clear sky-blue image", CreateModule<window::ModuleWindow>, true},
}};

const ModuleInfo *FindModule(std::string_view name) {
  for (const auto &module : modules)
    if (name == module.name)
      return &module;
  return nullptr;
}

int PositiveInteger(std::string_view value) {
  int result = 0;
  auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result <= 0)
    throw std::invalid_argument("Expected a positive integer: " + std::string(value));
  return result;
}

BackendAPI ParseBackend(std::string_view name) {
  using namespace grassland::graphics;
  if (name == "auto")
    return BACKEND_API_DEFAULT;
  if (name == "metal")
    return BACKEND_API_METAL;
  if (name == "vulkan")
    return BACKEND_API_VULKAN;
  if (name == "d3d12")
    return BACKEND_API_D3D12;
  throw std::invalid_argument("Unknown backend: " + std::string(name));
}

void ListModules() {
  for (size_t i = 0; i < modules.size(); ++i)
    std::cout << fmt::format("  {}. {:<21} {}\n", i + 1, modules[i].name, modules[i].description);
}

const ModuleInfo *SelectModule(BackendAPI api) {
  std::vector<snowberg::tui::MenuOption> options;
  for (const auto &module : modules)
    options.push_back({module.name, module.description});
  snowberg::tui::Menu menu;
  const auto selected = menu.Select(
      "Graphics Hello - Module Selection (" + std::string(grassland::graphics::BackendAPIString(api)) + ")", options);
  if (!selected)
    return nullptr;
  return &modules[*selected];
}

void RunModule(const ModuleInfo &info, BackendAPI api, int frames, bool headless) {
  grassland::LogInfo("Module: {}", info.name);
  auto module = info.create(api, headless);
  module->OnInit();
  if (headless) {
    for (int i = 0; i < 60; ++i) {
      module->OnUpdate();
      module->OnRender();
    }
    module->GetCore()->WaitGPU();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
      module->OnUpdate();
      module->OnRender();
    }
    module->GetCore()->WaitGPU();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << fmt::format("{}: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             info.name, frames, seconds, frames / seconds);
    module->OnClose();
    return;
  }
  auto *window = module->GetWindow();
  const std::string title = window->GetTitle();
  window->SetTitle(title + " | FPS: --");
  auto fps_start = std::chrono::steady_clock::now();
  int fps_frames = 0;
  int rendered = 0;
  while (module->IsAlive() && (!frames || rendered < frames)) {
    grassland::graphics::Window::PollEvents();
    module->OnUpdate();
    if (module->IsAlive()) {
      module->OnRender();
      if (frames)
        ++rendered;
      ++fps_frames;
      const auto now = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(now - fps_start).count();
      if (seconds >= 0.5) {
        window->SetTitle(fmt::format("{} | FPS: {:.1f}", title, fps_frames / seconds));
        fps_start = now;
        fps_frames = 0;
      }
    }
  }
  module->OnClose();
}

void PrintHelp(const char *executable) {
  std::cout
      << "Usage: " << executable
      << " [--module NAME | --tui | --all] [--backend auto|metal|vulkan|d3d12] [--frames N] [--headless]\n"
         "       --headless Benchmark window/triangle/cube/texture/blend without a window (default 600 frames)\n"
         "       --all      Run all five headless benchmarks (requires --headless)\n"
         "       --list     List all modules without opening a window\n"
         "       --help     Show this help\n"
         "Missing backend or module: select each missing option in the terminal menu; a sole enabled backend is automatic.\n--backend auto explicitly uses the default API.\n\n";
  ListModules();
}
}  // namespace

int Main(int argc, char **argv) {
  using namespace grassland::graphics;
  try {
    BackendAPI api = BACKEND_API_DEFAULT;
    const ModuleInfo *module = nullptr;
    int frames = 0;
    bool backend_specified = false;
    bool tui = false;
    bool list = false;
    bool headless = false;
    bool all = false;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        PrintHelp(argv[0]);
        return 0;
      }
      if (option == "--headless")
        headless = true;
      else if (option == "--all")
        all = true;
      else if (option == "--tui")
        tui = true;
      else if (option == "--list")
        list = true;
      else if ((option == "--module" || option == "--backend" || option == "--frames") && i + 1 < argc) {
        const std::string value = argv[++i];
        if (option == "--module") {
          module = FindModule(value);
          if (!module)
            throw std::invalid_argument("Unknown module: " + value + "; use --list to see available modules");
        } else if (option == "--backend") {
          api = ParseBackend(value);
          backend_specified = true;
        } else
          frames = PositiveInteger(value);
      } else
        throw std::invalid_argument("Unknown or incomplete option: " + option);
    }
    if (all && (!headless || module || tui)) {
      std::cerr << "--all requires --headless and cannot be combined with --module or --tui\n";
      return 1;
    }
    if (headless && module && !module->supports_headless) {
      std::cerr << "Headless benchmarks support window, triangle, cube and texture\n";
      return 1;
    }
    if (headless && !frames)
      frames = 600;
    if (tui && module)
      throw std::invalid_argument("Use either --module or --tui, not both");
    if (list) {
      ListModules();
      return 0;
    }
    if (!backend_specified) {
      std::vector<BackendAPI> backends;
      std::vector<snowberg::tui::MenuOption> options;
      for (const auto backend : {BACKEND_API_D3D12, BACKEND_API_VULKAN, BACKEND_API_METAL}) {
        if (SupportBackendAPI(backend)) {
          backends.push_back(backend);
          switch (backend) {
            case BACKEND_API_D3D12:
              options.push_back({BackendAPIString(backend), "Microsoft | Windows"});
              break;
            case BACKEND_API_VULKAN:
              options.push_back({BackendAPIString(backend), "Khronos Group | Windows, Linux, Android"});
              break;
            case BACKEND_API_METAL:
              options.push_back({BackendAPIString(backend), "Apple | macOS, iOS, iPadOS"});
              break;
          }
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
        const auto selected = menu.Select("Graphics Hello - API Selection", options);
        if (!selected)
          return 0;
        api = backends[*selected];
      }
    }
    if (!SupportBackendAPI(api))
      throw std::runtime_error("Requested graphics backend is unavailable");
    if (all) {
      for (const auto &entry : modules) {
        if (entry.supports_headless)
          RunModule(entry, api, frames, true);
      }
      return 0;
    }
    if (!module)
      module = SelectModule(api);
    if (module) {
      if (headless && !module->supports_headless) {
        std::cerr << "Headless benchmarks support window, triangle, cube and texture\n";
        return 1;
      }
      RunModule(*module, api, frames, headless);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace graphics_hello

int main(int argc, char **argv) {
  return graphics_hello::Main(argc, argv);
}
