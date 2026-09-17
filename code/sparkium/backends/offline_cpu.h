#pragma once
// CPU rendering backend of the offline path tracer.
//
// It runs the portable shading core (backends/core, a transcription of
// code/sparkium/shaders) on the host in worker threads: the graphics device is
// only used as the scene's resource store, so no GPU work is required to
// produce an image. The per-pixel work is byte-for-byte the same code the CUDA
// backend runs in its kernels.

#include "sparkium/backends/offline_backend.h"

namespace sparkium::backends {

class CpuOfflineBackend final : public OfflineBackend {
 public:
  static constexpr const char *kName = "cpu";

  const char *Name() const override {
    return kName;
  }

  void Dispatch(const OfflineScene &scene, const RenderSettings &settings, OfflineFilm &film) override;

  std::string DeviceDescription() const override;

  // Number of worker threads used for the next dispatch (for reporting).
  uint32_t ThreadCount() const;
};

}  // namespace sparkium::backends
