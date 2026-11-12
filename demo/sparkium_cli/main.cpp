#include <long_march.h>

#include "../sparkium_backend.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <filesystem>
#include <iostream>

#include "stb_image_write.h"

using namespace long_march;

namespace {
void Usage(const char *program) {
  std::cerr << "Usage: " << program << " <scene.json> [-o image.png] [--hdr-output image.hdr] [--frames N] "
            << "[--backend auto|metal|vulkan|d3d12] [--pipeline auto|rasterization|ray_tracing|rt_fallback|ray_query] "
               "[--require-hardware-rt] [--debug]\n"
            << "       " << program << " --list [scene-directory]\n";
}

sparkium::RenderPipeline ParsePipeline(const std::string &name) {
  if (name == "ray_query")
    return sparkium::RENDER_PIPELINE_RAY_QUERY;
  if (name == "rt_fallback")
    return sparkium::RENDER_PIPELINE_RT_FALLBACK;
  if (name == "auto")
    return sparkium::RENDER_PIPELINE_AUTO;
  if (name == "rasterization")
    return sparkium::RENDER_PIPELINE_RASTERIZATION;
  if (name == "ray_tracing")
    return sparkium::RENDER_PIPELINE_RAY_TRACING;
  throw std::runtime_error("unknown pipeline: " + name);
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      Usage(argv[0]);
      return 2;
    }
    if (std::string(argv[1]) == "--list") {
      auto directory = argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::path(FindAssetPath("scenes"));
      for (const auto &path : sparkium::FindJsonScenes(directory))
        std::cout << path.string() << '\n';
      return 0;
    }

    std::filesystem::path scene_path = argv[1];
    std::filesystem::path output = "output.png";
    std::filesystem::path hdr_output;
    int frames = 1;
    auto backend = graphics::BACKEND_API_DEFAULT;
    bool override_pipeline = false;
    bool require_hardware_rt = false;
    bool debug = false;
    sparkium::RenderPipeline pipeline = sparkium::RENDER_PIPELINE_AUTO;
    for (int i = 2; i < argc; ++i) {
      std::string argument = argv[i];
      if (argument == "--require-hardware-rt")
        require_hardware_rt = true;
      else if (argument == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (argument == "--debug")
        debug = true;
      else if ((argument == "-o" || argument == "--output") && i + 1 < argc)
        output = argv[++i];
      else if (argument == "--hdr-output" && i + 1 < argc)
        hdr_output = argv[++i];
      else if (argument == "--frames" && i + 1 < argc)
        frames = std::stoi(argv[++i]);
      else if (argument == "--pipeline" && i + 1 < argc) {
        pipeline = ParsePipeline(argv[++i]);
        override_pipeline = true;
      } else {
        throw std::runtime_error("unknown or incomplete argument: " + argument);
      }
    }
    if (frames <= 0)
      throw std::runtime_error("--frames must be positive");

    if (!hdr_output.empty() && hdr_output.extension() != ".hdr")
      throw std::runtime_error("--hdr-output requires a .hdr (linear sRGB Radiance RGBE) path");
    if (!hdr_output.empty() && std::filesystem::absolute(hdr_output).lexically_normal() ==
                                   std::filesystem::absolute(output).lexically_normal())
      throw std::runtime_error("SDR and HDR outputs must use different paths");

    std::unique_ptr<graphics::Core> graphics_core;
    if (graphics::CreateCore(backend, graphics::Core::Settings{2, debug}, &graphics_core) != 0)
      throw std::runtime_error("failed to create graphics core");
    if (graphics_core->InitializeLogicalDeviceAutoSelect(false) != 0)
      throw std::runtime_error("failed to initialize graphics device");
    std::cout << "Backend: " << graphics::BackendAPIString(graphics_core->API())
              << ", device: " << graphics_core->DeviceName() << '\n';
    if (require_hardware_rt && !graphics_core->DeviceRayTracingSupport())
      throw std::runtime_error("hardware ray tracing is unavailable on the selected device");
    sparkium::Core core(graphics_core.get());
    std::string error;
    auto loaded = sparkium::JsonScene::Load(&core, scene_path, &error);
    if (!loaded)
      throw std::runtime_error(error);
    if (!override_pipeline)
      pipeline = loaded->GetRenderPipeline();
    if (core.ResolveRenderPipeline(pipeline) == sparkium::RENDER_PIPELINE_RAY_QUERY)
      std::cout << "Tracing: native ray query (compute, native AS)\n";

    auto *film = loaded->GetFilm();
    std::unique_ptr<graphics::Image> image;
    graphics_core->CreateImage(film->GetWidth(), film->GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
    for (int frame = 0; frame < frames; ++frame)
      core.Render(loaded->GetScene(), loaded->GetCamera(), film, pipeline);
    film->Develop(image.get());
    std::vector<uint8_t> pixels(static_cast<size_t>(film->GetWidth()) * film->GetHeight() * 4);
    image->DownloadData(pixels.data());
    std::filesystem::create_directories(output.has_parent_path() ? output.parent_path() : ".");
    if (!stbi_write_png(output.string().c_str(), film->GetWidth(), film->GetHeight(), 4, pixels.data(),
                        film->GetWidth() * 4))
      throw std::runtime_error("failed to write image: " + output.string());
    if (!hdr_output.empty()) {
      std::unique_ptr<graphics::Image> hdr_image;
      graphics_core->CreateImage(film->GetWidth(), film->GetHeight(), graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT,
                                 &hdr_image);
      film->Develop(hdr_image.get(), true);
      std::vector<float> hdr_pixels(static_cast<size_t>(film->GetWidth()) * film->GetHeight() * 4);
      hdr_image->DownloadData(hdr_pixels.data());
      std::filesystem::create_directories(hdr_output.has_parent_path() ? hdr_output.parent_path() : ".");
      if (!stbi_write_hdr(hdr_output.string().c_str(), film->GetWidth(), film->GetHeight(), 4, hdr_pixels.data()))
        throw std::runtime_error("failed to write HDR image: " + hdr_output.string());
      std::cout << "Saved linear sRGB HDR (with scene exposure, without SDR tone mapping) to " << hdr_output.string()
                << '\n';
    }
    std::cout << "Rendered '" << loaded->GetName() << "' (" << film->GetWidth() << 'x' << film->GetHeight() << ", "
              << frames << " frame(s)) to " << output.string() << '\n';
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_cli: " << exception.what() << '\n';
    return 1;
  }
}
