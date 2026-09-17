#pragma once
// Backend selection for the offline (CPU and CUDA) Sparkium renderers.
//
// Both backends execute the same portable shading core
// (code/sparkium/backends/core, a transcription of code/sparkium/shaders) and
// only differ in where the per-pixel path tracing runs. The graphics device is
// used solely as the scene's resource store: no compute or ray tracing
// dispatch is issued by either backend.

#include <memory>
#include <string>
#include <vector>

#include "sparkium/backends/offline_film.h"
#include "sparkium/backends/offline_scene.h"

namespace sparkium {
class Camera;
class Film;
class Scene;
}  // namespace sparkium

namespace sparkium::backends {

class OfflineBackend {
 public:
  virtual ~OfflineBackend() = default;

  virtual const char *Name() const = 0;

  // Uploads or caches backend state. Called before the first Dispatch and
  // again whenever the flattened scene changed.
  virtual void Sync(const OfflineScene &scene) {
    (void)scene;
  }
  // Resolves the flattened scene into device memory and returns the
  // DeviceScene the kernels read. Dies after Sync.
  virtual void Dispatch(const OfflineScene &scene, const RenderSettings &settings, OfflineFilm &film) = 0;

  // film2img + tone mapping for the accumulated film. The default runs on the
  // host; the CUDA backend resolves inside a kernel.
  virtual std::vector<uint8_t> Develop(const OfflineFilm &film, const ToneMappingSettings &settings) const {
    return film.Develop(settings);
  }

  virtual std::string DeviceDescription() const {
    return "";
  }
};

// True when `name` is a backend compiled into this build.
bool OfflineBackendSupported(const std::string &name);
// True when `name` names an offline renderer ("cpu" or "cuda") independently of
// whether it was built into this binary, so that a caller can route the name to
// CreateOfflineBackend and report the reason a backend is unavailable.
bool IsOfflineBackendName(const std::string &name);
// "cpu" or "cuda"; throws for unknown or unbuilt backends.
std::unique_ptr<OfflineBackend> CreateOfflineBackend(const std::string &name);

struct OfflineRenderStats {
  std::string backend;
  std::string device;
  std::string scene;
  uint32_t width{0};
  uint32_t height{0};
  uint32_t frames{0};
  int32_t samples_per_dispatch{0};
  int32_t max_bounces{0};
  int32_t accumulated_samples{0};
  double seconds{0.0};
  std::vector<double> frame_seconds;
};

struct OfflineRenderResult {
  uint32_t width{0};
  uint32_t height{0};
  std::vector<uint8_t> rgba8;
  OfflineRenderStats stats;
};

// Accumulates `frames` dispatches into a fresh film and resolves it, exactly
// like `frames` iterations of sparkium::Core::Render followed by
// Film::Develop. Only `backend` performs rendering work.
OfflineRenderResult RenderOfflineScene(OfflineBackend *backend,
                                       sparkium::Scene *scene,
                                       sparkium::Camera *camera,
                                       sparkium::Film *film,
                                       uint32_t frames);

std::unique_ptr<OfflineBackend> CreateCpuBackend();
#if defined(LONGMARCH_CUDA_ENABLED)
std::unique_ptr<OfflineBackend> CreateCudaBackend();
#endif

}  // namespace sparkium::backends
