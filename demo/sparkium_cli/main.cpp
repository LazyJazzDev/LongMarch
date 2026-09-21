#include <long_march.h>

#include "../sparkium_backend.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "stb_image_write.h"

using namespace long_march;

namespace {
void Usage(const char *program) {
  std::cerr << "Usage: " << program << " <scene.json> [-o image.png] [--frames N] "
            << "[--backend auto|metal|vulkan|d3d12|cpu|cuda] [--pipeline "
               "auto|rasterization|ray_tracing|rt_fallback|ray_query] "
               "[--require-hardware-rt] [--debug] [--profile "
               "timings.csv] [--profile-cpu-only|--profile-alternate-gpu] [--linear-output image.pfm]\n"
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
    std::filesystem::path linear_output;
    int frames = 1;
    sparkium::BackendSelection backend;
    std::filesystem::path profile_path;
    bool profile_cpu_only = false;
    bool profile_alternate_gpu = false;
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
      else if (argument == "--linear-output" && i + 1 < argc)
        linear_output = argv[++i];
      else if (argument == "--profile-alternate-gpu")
        profile_alternate_gpu = true;
      else if (argument == "--profile-cpu-only")
        profile_cpu_only = true;
      else if (argument == "--profile" && i + 1 < argc)
        profile_path = argv[++i];
      else if (argument == "--frames" && i + 1 < argc)
        frames = std::stoi(argv[++i]);
      else if (argument == "--pipeline" && i + 1 < argc) {
        pipeline = ParsePipeline(argv[++i]);
        override_pipeline = true;
      } else {
        throw std::runtime_error("unknown or incomplete argument: " + argument);
      }
    }
    if ((profile_cpu_only || profile_alternate_gpu) && profile_path.empty())
      throw std::runtime_error("profiling mode requires --profile");
    if (profile_cpu_only && profile_alternate_gpu)
      throw std::runtime_error("choose one profiling mode");
    if (frames <= 0)
      throw std::runtime_error("--frames must be positive");

    auto document = sparkium::LoadSceneDocument(scene_path);
    auto scene = document.scene;
    auto renderer = sparkium::CreateRenderer(SparkiumRendererSettings(backend, debug));
    renderer->SetScene(scene);
    auto preferred = renderer->SupportsPipeline(document.preferred_pipeline) ? document.preferred_pipeline
                                                                             : sparkium::RENDER_PIPELINE_AUTO;
    renderer->Configure({override_pipeline ? pipeline : preferred, std::nullopt});
    pipeline = renderer->Pipeline();
    const auto info = renderer->Info();
    const auto resolved_pipeline = renderer->ResolvePipeline(pipeline);
    std::cout << "Backend: " << sparkium::BackendName(backend) << ", device: " << info.device << '\n';
    if (require_hardware_rt && (!info.ray_tracing || resolved_pipeline != sparkium::RENDER_PIPELINE_RAY_TRACING))
      throw std::runtime_error("--require-hardware-rt requires available hardware and the ray_tracing pipeline");
    const char *pipeline_names[]{"rasterization", "ray_tracing", "auto", "rt_fallback", "ray_query"};
    std::cout << "Tracing pipeline: " << pipeline_names[resolved_pipeline] << '\n';
    sparkium::RenderImage image;
    std::ofstream profile_output;
    if (!profile_path.empty()) {
      std::filesystem::create_directories(profile_path.has_parent_path() ? profile_path.parent_path() : ".");
      profile_output.open(profile_path);
      if (!profile_output)
        throw std::runtime_error("cannot open profile output");
      profile_output << "frame,domain,stage,value\n" << std::fixed << std::setprecision(6);
      std::cout << "Profiling device: " << info.device << '\n';
    }
    for (int frame = 0; frame < frames; ++frame) {
      if (!profile_path.empty())
        renderer->BeginProfile(!profile_cpu_only && (!profile_alternate_gpu || frame % 4 == 0 || frame % 4 == 3));
      renderer->Render();
      if (!profile_path.empty()) {
        image = renderer->ReadImage();
        auto profile = renderer->EndProfile();
        for (const auto &[name, value] : profile.cpu_ms)
          profile_output << frame << ",cpu_ms," << name << ',' << value << '\n';
        for (const auto &[name, value] : profile.gpu_ms)
          profile_output << frame << ",gpu_ms," << name << ',' << value << '\n';
        for (const auto &[name, value] : profile.counters)
          profile_output << frame << ",count," << name << ',' << value << '\n';
        profile_output.flush();
      }
    }
    if (profile_path.empty())
      image = renderer->ReadImage();
    if (!linear_output.empty()) {
      // PFM preserves pre-display linear RGB for numerical regression checks.
      // Reject NaNs instead of hiding them in the UNORM display conversion.
      auto linear = renderer->ReadLinearImage();
      std::filesystem::create_directories(linear_output.has_parent_path() ? linear_output.parent_path() : ".");
      std::ofstream stream(linear_output, std::ios::binary);
      const uint32_t endian = 1;
      stream << "PF\n"
             << image.width << ' ' << image.height << "\n"
             << (*reinterpret_cast<const uint8_t *>(&endian) ? "-1.0\n" : "1.0\n");
      for (int y = image.height - 1; y >= 0; --y)
        for (int x = 0; x < image.width; ++x) {
          const auto &pixel = linear[static_cast<size_t>(y) * image.width + x];
          for (int c = 0; c < 3; ++c)
            if (!std::isfinite(pixel[c]))
              throw std::runtime_error("non-finite linear radiance at pixel " + std::to_string(x) + "," +
                                       std::to_string(y));
          stream.write(reinterpret_cast<const char *>(&pixel), sizeof(float) * 3);
        }
      if (!stream)
        throw std::runtime_error("failed to write linear image: " + linear_output.string());
    }

    std::filesystem::create_directories(output.has_parent_path() ? output.parent_path() : ".");
    if (!stbi_write_png(output.string().c_str(), image.width, image.height, 4, image.rgba.data(), image.width * 4))
      throw std::runtime_error("failed to write image: " + output.string());
    std::cout << "Rendered '" << scene->name << "' (" << image.width << 'x' << image.height << ", " << frames
              << " frame(s)) to " << output.string() << '\n';
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_cli: " << exception.what() << '\n';
    return 1;
  }
}
