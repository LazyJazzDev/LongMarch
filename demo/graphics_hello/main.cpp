#include <array>
#include <charconv>
#include <chrono>
#include <iostream>
#include <string_view>

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

namespace graphics_hello {
namespace {
using grassland::graphics::BackendAPI;

struct ModuleInfo {
  const char *name;
  const char *description;
  std::unique_ptr<Module> (*create)(BackendAPI);
};

template <typename T>
std::unique_ptr<Module> CreateModule(BackendAPI api) {
  return std::make_unique<T>(api);
}

const std::array<ModuleInfo, 11> modules{{
    {"triangle", "Colored triangle", CreateModule<triangle::ModuleTriangle>},
    {"blend", "Alpha blending", CreateModule<blend::ModuleBlend>},
    {"cube", "Rotating cube", CreateModule<cube::ModuleCube>},
    {"texture", "Textured triangle", CreateModule<texture::ModuleTexture>},
    {"resize", "Resizable window", CreateModule<resize::ModuleResize>},
    {"hdr", "HDR gradient and SDR reference (H toggles HDR/SDR)", CreateModule<hdr::ModuleHDR>},
    {"sdr_sample", "SDR sampling", CreateModule<sdr_sample::ModuleSDRSample>},
    {"raytracing", "Ray tracing pipeline (unavailable on Metal)", CreateModule<raytracing::ModuleRayTracing>},
    {"rt_multi_shader_group", "Triangle + procedural sphere (requires RT pipelines)",
     CreateModule<rt_multi_shader_group::ModuleRTMultiShaderGroup>},
    {"external_shader", "RT scene with shaders loaded from assets (requires RT pipelines)",
     CreateModule<external_shader::ModuleExternalShader>},
    {"ray_query", "Compute ray queries (requires device/backend support)", CreateModule<ray_query::ModuleRayQuery>},
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

// A line-based TUI works in native terminals, IDE consoles and redirected input,
// without changing terminal modes or requiring a platform-specific dependency.
const ModuleInfo *SelectModule(BackendAPI api) {
  std::cout << "\nGraphics Hello - Module Selection\n"
            << "Backend API: " << grassland::graphics::BackendAPIString(api) << "\n\n";
  ListModules();
  for (;;) {
    std::cout << "\nSelect module [1-" << modules.size() << " or name], q to quit: " << std::flush;
    std::string input;
    if (!std::getline(std::cin, input))
      return nullptr;
    const auto first = input.find_first_not_of(" \t\r");
    if (first == std::string::npos)
      continue;
    input = input.substr(first, input.find_last_not_of(" \t\r") - first + 1);
    if (input == "q" || input == "quit")
      return nullptr;
    if (auto module = FindModule(input))
      return module;
    try {
      const int index = PositiveInteger(input);
      if (index <= static_cast<int>(modules.size()))
        return &modules[index - 1];
    } catch (const std::invalid_argument &) {
    }
    std::cout << "Invalid selection. Enter a listed number or module name.\n";
  }
}

void RunModule(const ModuleInfo &info, BackendAPI api, int frames) {
  grassland::LogInfo("Module: {}", info.name);
  auto module = info.create(api);
  module->OnInit();
  auto *window = module->GetWindow();
  const std::string title = window->GetTitle();
  window->SetTitle(title + " | FPS: --");
  auto fps_start = std::chrono::steady_clock::now();
  int fps_frames = 0;
  int rendered = 0;
  while (module->IsAlive() && (!frames || rendered < frames)) {
    glfwPollEvents();
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
  std::cout << "Usage: " << executable
            << " [--module NAME | --tui] [--backend auto|metal|vulkan|d3d12] [--frames N]\n"
               "       --list     List all modules without opening a window\n"
               "       --help     Show this help\n"
               "With no module specified, the terminal selection menu opens.\n\n";
  ListModules();
}
}  // namespace

int Main(int argc, char **argv) {
  using namespace grassland::graphics;
  try {
    BackendAPI api = BACKEND_API_DEFAULT;
    const ModuleInfo *module = nullptr;
    int frames = 0;
    bool tui = false;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        PrintHelp(argv[0]);
        return 0;
      }
      if (option == "--tui")
        tui = true;
      else if (option == "--list")
        list = true;
      else if ((option == "--module" || option == "--backend" || option == "--frames") && i + 1 < argc) {
        const std::string value = argv[++i];
        if (option == "--module") {
          module = FindModule(value);
          if (!module)
            throw std::invalid_argument("Unknown module: " + value + "; use --list to see available modules");
        } else if (option == "--backend")
          api = ParseBackend(value);
        else
          frames = PositiveInteger(value);
      } else
        throw std::invalid_argument("Unknown or incomplete option: " + option);
    }
    if (tui && module)
      throw std::invalid_argument("Use either --module or --tui, not both");
    if (list) {
      ListModules();
      return 0;
    }
    if (!SupportBackendAPI(api))
      throw std::runtime_error("Requested graphics backend is unavailable");
    if (!module)
      module = SelectModule(api);
    if (module)
      RunModule(*module, api, frames);
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
