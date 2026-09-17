#include "sparkium/backends/offline_backend.h"

#include <chrono>
#include <stdexcept>

#include "sparkium/sparkium.h"

namespace sparkium::backends {

bool OfflineBackendSupported(const std::string &name) {
  if (name == "cpu")
    return true;
#if defined(LONGMARCH_CUDA_ENABLED)
  if (name == "cuda")
    return true;
#endif
  return false;
}

bool IsOfflineBackendName(const std::string &name) {
  return name == "cpu" || name == "cuda";
}

std::unique_ptr<OfflineBackend> CreateOfflineBackend(const std::string &name) {
  if (name == "cpu")
    return CreateCpuBackend();
  if (name == "cuda") {
#if defined(LONGMARCH_CUDA_ENABLED)
    return CreateCudaBackend();
#else
    throw std::runtime_error("the CUDA backend was not built (configure with CUDA enabled)");
#endif
  }
  throw std::invalid_argument("unknown offline backend: " + name);
}

OfflineRenderResult RenderOfflineScene(OfflineBackend *backend,
                                       sparkium::Scene *scene,
                                       sparkium::Camera *camera,
                                       sparkium::Film *film,
                                       uint32_t frames) {
  if (!backend)
    throw std::runtime_error("no offline backend was created");
  if (frames == 0)
    throw std::runtime_error("offline rendering needs at least one frame");
  auto flattened = OfflineScene::Build(scene, camera);

  OfflineFilm offline_film;
  offline_film.Reset(static_cast<uint32_t>(film->GetWidth()), static_cast<uint32_t>(film->GetHeight()));

  OfflineRenderResult result;
  result.width = offline_film.Width();
  result.height = offline_film.Height();
  result.stats.backend = backend->Name();
  result.stats.device = backend->DeviceDescription();
  result.stats.scene = flattened->Description();
  result.stats.width = offline_film.Width();
  result.stats.height = offline_film.Height();
  result.stats.frames = frames;
  result.stats.samples_per_dispatch = scene->settings.raytracing.samples_per_dispatch;
  result.stats.max_bounces = scene->settings.raytracing.max_bounces;
  result.stats.accumulated_samples = film->info.accumulated_samples;

  backend->Sync(*flattened);
  const auto started = std::chrono::steady_clock::now();
  for (uint32_t frame = 0; frame < frames; ++frame) {
    RenderSettings settings{};
    settings.samples_per_dispatch = scene->settings.raytracing.samples_per_dispatch;
    settings.max_bounces = scene->settings.raytracing.max_bounces;
    settings.alpha_shadow = scene->settings.raytracing.alpha_shadow;
    settings.accumulated_samples = film->info.accumulated_samples;
    const glm::vec3 &background = scene->settings.raytracing.background_color;
    settings.background_color = device::float3(background.x, background.y, background.z);
    settings.persistence = film->info.persistence;
    settings.clamping = film->info.clamping;
    settings.max_exposure = film->info.max_exposure;
    settings.view_transform = film->info.view_transform;
    settings.exposure = film->info.exposure;
    settings.gamma = film->info.gamma;
    settings.contrast = film->info.contrast;

    // The Sobol table has one row per sample index actually consumed.
    if (flattened->EnsureSobolRows(static_cast<uint32_t>(settings.accumulated_samples) +
                                   static_cast<uint32_t>(settings.samples_per_dispatch)))
      backend->Sync(*flattened);

    const auto frame_started = std::chrono::steady_clock::now();
    backend->Dispatch(*flattened, settings, offline_film);
    result.stats.frame_seconds.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_started).count());
    // Mirrors raytracing::Scene::Render, which advances the film's sample
    // counter after uploading the settings.
    film->info.accumulated_samples += settings.samples_per_dispatch;
  }
  result.stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  result.stats.accumulated_samples = film->info.accumulated_samples;

  ToneMappingSettings tone_mapping{};
  tone_mapping.view_transform = film->info.view_transform;
  tone_mapping.exposure = film->info.exposure;
  tone_mapping.gamma = film->info.gamma;
  tone_mapping.contrast = film->info.contrast;
  result.rgba8 = backend->Develop(offline_film, tone_mapping);
  return result;
}

}  // namespace sparkium::backends
