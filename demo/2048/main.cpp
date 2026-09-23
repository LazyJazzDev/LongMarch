#include <iostream>
#include <stdexcept>

#include "2048.h"

namespace {

graphics::BackendAPI ParseBackend(const std::string &name) {
  if (name == "auto") {
    return graphics::BACKEND_API_DEFAULT;
  }
  if (name == "vulkan") {
    return graphics::BACKEND_API_VULKAN;
  }
  if (name == "d3d12") {
    return graphics::BACKEND_API_D3D12;
  }
  if (name == "metal") {
    return graphics::BACKEND_API_METAL;
  }
  throw std::invalid_argument("Unknown backend: " + name);
}

void PrintHelp(const char *executable) {
  std::cout << "Usage: " << executable << " [--backend auto|vulkan|d3d12|metal] [--frames N] [--screenshot FILE]\n"
            << "  --frames N         Exit after N rendered frames\n"
            << "  --screenshot FILE  Save the last frame as PNG on exit\n"
            << "Use the arrow keys to move the blocks.\n";
}

}  // namespace

int main(int argc, char *argv[]) {
  try {
    int frames = 0;
    auto api = graphics::BACKEND_API_DEFAULT;
    std::string screenshot;

    for (int i = 1; i < argc; i++) {
      const std::string option = argv[i];
      if (option == "--help") {
        PrintHelp(argv[0]);
        return 0;
      } else if (option == "--backend" && i + 1 < argc) {
        api = ParseBackend(argv[++i]);
      } else if (option == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
      } else if (option == "--screenshot" && i + 1 < argc) {
        screenshot = argv[++i];
      } else {
        throw std::invalid_argument("Unknown or incomplete option: " + option);
      }
    }

    TwentyFourEight application("2048", 720, 960, api);
    application.SetScreenshotPath(screenshot);
    application.Run(frames);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
