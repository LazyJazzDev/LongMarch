#include "sparkium/pipelines/native/native.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "grassland/graphics/frame_profile.h"
#include "sparkium/core/camera.h"
#include "sparkium/core/core.h"
#include "sparkium/core/scene.h"
#include "sparkium/pipelines/native/core/film.h"
#include "sparkium/pipelines/native/core/scene_data.h"

#ifdef LONGMARCH_SPARKIUM_NATIVE_CUDA
#include "sparkium/pipelines/native/cuda/cuda_renderer.h"
#endif

namespace sparkium::native {

namespace {

// Per-scene flattener, attached to the `sparkium::Scene` object chain so the
// blob cache and the BVH survive across frames.
class SceneState : public Object {
 public:
  explicit SceneState(sparkium::Scene &) {
  }

  SceneData &Data(sparkium::Core *core) {
    if (!data_)
      data_ = std::make_unique<SceneData>(core);
    return *data_;
  }

#ifdef LONGMARCH_SPARKIUM_NATIVE_CUDA
  CudaRenderer &Cuda() {
    if (!cuda_)
      cuda_ = std::make_unique<CudaRenderer>();
    return *cuda_;
  }
#endif

 private:
  std::unique_ptr<SceneData> data_;
#ifdef LONGMARCH_SPARKIUM_NATIVE_CUDA
  std::unique_ptr<CudaRenderer> cuda_;
#endif
};

SceneState *DedicatedCast(sparkium::Scene *scene) {
  COMPONENT_CAST(scene, SceneState);
}

// Runs `RenderPixel` over the frame on a thread pool. Each pixel owns its
// accumulation slot and its sample seed, so the split across threads does not
// change a single sample.
void RenderCpu(const SceneView &view,
               uint2 extent,
               std::vector<float4> &accumulated_color,
               std::vector<float> &accumulated_samples) {
  const uint32_t rows = extent.y;
  unsigned int workers = std::thread::hardware_concurrency();
  if (workers == 0)
    workers = 1;
  workers = std::min<unsigned int>(workers, std::max<uint32_t>(rows, 1u));

  std::atomic<uint32_t> next_row{0};
  auto worker = [&]() {
    for (uint32_t y = next_row++; y < rows; y = next_row++)
      for (uint32_t x = 0; x < extent.x; ++x)
        RenderPixel(view, uint2{x, y}, extent, accumulated_color.data(), accumulated_samples.data());
  };

  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (unsigned int i = 1; i < workers; ++i)
    threads.emplace_back(worker);
  worker();
  for (auto &thread : threads)
    thread.join();
}

}  // namespace

bool CudaAvailable() {
#ifdef LONGMARCH_SPARKIUM_NATIVE_CUDA
  return CudaRenderer::Available();
#else
  return false;
#endif
}

void Render(sparkium::Core *core,
            sparkium::Scene *scene,
            sparkium::Camera *camera,
            sparkium::Film *film,
            Backend backend) {
  auto *state = DedicatedCast(scene);
  auto *native_film = DedicatedCast(film);

  graphics::CpuProfileScope update_profile("scene_update");
  auto &scene_data = state->Data(core);
  scene_data.Update(scene, camera, film->info);
  update_profile.End();

  film->info.accumulated_samples += scene->settings.raytracing.samples_per_dispatch;

  const uint2 extent{static_cast<uint32_t>(film->GetWidth()), static_cast<uint32_t>(film->GetHeight())};
  graphics::CpuProfileScope trace_profile("path_trace");
  if (backend == BACKEND_CUDA) {
#ifdef LONGMARCH_SPARKIUM_NATIVE_CUDA
    state->Cuda().Render(scene_data, extent, native_film->AccumulatedColor(), native_film->AccumulatedSamples());
#else
    throw std::runtime_error("this build has no CUDA backend; reconfigure without LONGMARCH_DISABLE_CUDA");
#endif
  } else {
    RenderCpu(scene_data.View(), extent, native_film->AccumulatedColor(), native_film->AccumulatedSamples());
  }
  trace_profile.End();

  native_film->PublishRawImage();

  if (graphics::FrameProfile::active)
    graphics::FrameProfile::active->counters["native_backend_cuda"] = backend == BACKEND_CUDA ? 1 : 0;
}

void DevelopToHost(sparkium::Film *film, std::vector<uint8_t> &pixels) {
  auto *native_film = DedicatedCast(film);
  const auto &color = native_film->AccumulatedColor();
  const auto &samples = native_film->AccumulatedSamples();
  const size_t count = color.size();
  pixels.resize(count * 4);

  // `film2img.hlsl` followed by `tone_mapping.hlsl`, the same two passes
  // `Film::Develop` dispatches.
  RenderSettings settings{};
  settings.view_transform = film->info.view_transform;
  settings.exposure = film->info.exposure;
  settings.gamma = film->info.gamma;
  settings.contrast = film->info.contrast;
  for (size_t i = 0; i < count; ++i) {
    const float4 mapped = ToneMap(settings, FilmToImage(color[i], samples[i]));
    for (int channel = 0; channel < 4; ++channel)
      pixels[i * 4 + channel] =
          static_cast<uint8_t>(saturatef(mapped[channel]) * 255.0f + 0.5f);
  }
}

}  // namespace sparkium::native
