#include "sparkium/pipelines/portable/portable.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "sparkium/core/core.h"
#include "sparkium/pipelines/portable/frame_resources.h"
#include "sparkium/pipelines/portable/kernel_gen.h"
#include "sparkium/pipelines/portable/kernel_runtime.h"
#include "sparkium/pipelines/portable/scene_bake.h"

namespace sparkium::portable {
namespace {

std::string CacheDirectory() {
  const char *env = std::getenv("SPARKIUM_PORTABLE_CACHE");
  if (env && *env)
    return env;
  return (std::filesystem::temp_directory_path() / "sparkium_portable").string();
}

}  // namespace

bool CudaAvailable() {
#ifdef SPARKIUM_PORTABLE_CUDA
  return CudaDeviceReady();
#else
  return false;
#endif
}

uint32_t HostThreadCount() {
  if (const char *env = std::getenv("SPARKIUM_CPU_THREADS"))
    if (int value = std::atoi(env); value > 0)
      return uint32_t(value);
  const unsigned int hw = std::thread::hardware_concurrency();
  return hw ? hw : 1;
}

void RenderFrame(ComputeBackendKind kind,
                 sparkium::Scene *scene,
                 sparkium::Camera *camera,
                 sparkium::Film *film,
                 const std::map<graphics::Image *, const HostImageData *> &host_images,
                 const std::vector<uint32_t> &sobol_table,
                 uint32_t *seed,
                 std::vector<float> &accumulation_color,
                 std::vector<float> &accumulation_samples) {
  const uint32_t width = uint32_t(film->GetWidth());
  const uint32_t height = uint32_t(film->GetHeight());
  const size_t pixel_count = size_t(width) * height;
  if (accumulation_color.size() != pixel_count * 4)
    accumulation_color.assign(pixel_count * 4, 0.0f);
  if (accumulation_samples.size() != pixel_count)
    accumulation_samples.assign(pixel_count, 0.0f);

  BakeResult bake = BakeScene(scene, camera, film, host_images, seed);
  const uint32_t sample_base = uint32_t(film->info.accumulated_samples);

  // Generate and compile the scene-specific kernel (cached across frames).
  auto *core = scene->GetCore();
  const std::string kernel_source =
      std::string("#include \"sparkium/pipelines/portable/hlsl_compat.h\"\n"
                  "#include <cstdio>\n"
                  "using namespace sparkium_portable;\n") +
      GenerateKernelSource(core, bake.materials);

  if (kind == BACKEND_KIND_HOST) {
    auto library = CompileKernelHost(kernel_source, CacheDirectory(), "cpu");
    RenderPixelsHostPrepared(bake, sobol_table, library->Entry(), library->ContextSlot(),
                             reinterpret_cast<sparkium_portable::float4 *>(accumulation_color.data()), accumulation_samples.data(),
                             width, height, sample_base, HostThreadCount());
  } else if (kind == BACKEND_KIND_CUDA) {
#ifdef SPARKIUM_PORTABLE_CUDA
    RenderPixelsCuda(bake, sobol_table, kernel_source,
                     reinterpret_cast<sparkium_portable::float4 *>(accumulation_color.data()), accumulation_samples.data(), width,
                     height, sample_base);
#else
    throw std::runtime_error("this build has no CUDA support; rebuild without LONGMARCH_DISABLE_CUDA");
#endif
  } else {
    throw std::runtime_error("unknown compute backend");
  }

  film->info.accumulated_samples += scene->settings.raytracing.samples_per_dispatch;
}

void Develop(const sparkium::Film *film,
             const std::vector<float> &accumulation_color,
             const std::vector<float> &accumulation_samples,
             std::vector<uint8_t> &rgba) {
  const uint32_t width = uint32_t(film->GetWidth());
  const uint32_t height = uint32_t(film->GetHeight());
  const size_t pixel_count = size_t(width) * height;
  std::vector<sparkium_portable::float4> averaged(pixel_count);
  for (size_t i = 0; i < pixel_count; ++i) {
    const float samples = accumulation_samples[i];
    if (samples > 0.0f) {
      averaged[i] = sparkium_portable::float4(accumulation_color[i * 4] / samples, accumulation_color[i * 4 + 1] / samples,
                           accumulation_color[i * 4 + 2] / samples, accumulation_color[i * 4 + 3] / samples);
    } else {
      averaged[i] = sparkium_portable::float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
  }
  rgba.resize(pixel_count * 4);
  ToneMapPixels(averaged.data(), film->info.view_transform, film->info.exposure, film->info.gamma,
                film->info.contrast, width, height, rgba.data());
}

}  // namespace sparkium::portable
