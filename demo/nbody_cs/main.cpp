#include <iostream>
#include <stdexcept>

#include "nbody_cs.h"

int main(int argc, char **argv) {
  try {
    NBodyOptions options;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      auto value = [&]() -> std::string {
        if (++i >= argc)
          throw std::invalid_argument("missing value for " + arg);
        return argv[i];
      };
      if (arg == "--backend") {
        auto name = value();
        if (name == "metal")
          options.backend = graphics::BACKEND_API_METAL;
        else if (name == "vulkan")
          options.backend = graphics::BACKEND_API_VULKAN;
        else if (name != "auto")
          throw std::invalid_argument("unknown backend: " + name);
      } else if (arg == "--particles")
        options.particles = std::stoi(value());
      else if (arg == "--frames")
        options.frames = std::stoi(value());
      else if (arg == "--warmup")
        options.warmup = std::stoi(value());
      else if (arg == "--seed")
        options.seed = std::stoul(value());
      else if (arg == "--width")
        options.width = std::stoi(value());
      else if (arg == "--height")
        options.height = std::stoi(value());
      else if (arg == "--csv")
        options.csv = value();
      else if (arg == "--state-output")
        options.state_output = value();
      else if (arg == "--debug")
        options.debug = true;
      else if (arg == "--no-gpu-timing")
        options.gpu_timing = false;
      else if (arg == "--mode")
        options.mode = value();
      else if (arg == "--help") {
        std::cout << "nbody_cs [--backend auto|metal|vulkan] [--mode interactive|compute|offscreen|window]\n"
                     "  [--particles N] [--frames N] [--warmup N] [--seed N] [--width W] [--height H]\n"
                     "  [--csv path] [--state-output path] [--debug] [--no-gpu-timing]\n"
                     "Benchmark modes wait for GPU completion each frame; initialization/warmup are excluded.\n";
        return 0;
      } else
        throw std::invalid_argument("unknown argument: " + arg);
    }
    if (!graphics::SupportBackendAPI(options.backend))
      throw std::runtime_error("backend was not built");
    if (options.particles <= 0 || options.particles % 128)
      throw std::invalid_argument("particles must be a positive multiple of 128");
    if (options.frames <= 0 || options.warmup < 0 || options.width <= 0 || options.height <= 0)
      throw std::invalid_argument("invalid frame count or image dimensions");
    if (options.mode != "interactive" && options.mode != "compute" && options.mode != "offscreen" &&
        options.mode != "window")
      throw std::invalid_argument("unknown mode: " + options.mode);
    NBodyCS app(options);
    app.Run();
  } catch (const std::exception &e) {
    std::cerr << "nbody_cs: " << e.what() << '\n';
    return 1;
  }
}
