#pragma once
// CUDA rendering backend of the offline path tracer.
//
// The backend keeps a device mirror of the flattened scene and runs the path
// tracing and the film resolve in CUDA kernels (offline_cuda_kernels.cu). The
// kernels call the portable shading core (backends/core) that the CPU backend
// also runs, so both backends execute the same shading math and only differ in
// the execution unit.
//
// Only compiled when LONGMARCH_CUDA_ENABLED is defined; the CPU backend and the
// shared core stay CUDA-free.

#include "sparkium/backends/offline_backend.h"

#if defined(LONGMARCH_CUDA_ENABLED)

#include <memory>

namespace sparkium::backends {

class CudaOfflineBackend final : public OfflineBackend {
 public:
  static constexpr const char *kName = "cuda";

  CudaOfflineBackend();
  ~CudaOfflineBackend() override;

  const char *Name() const override {
    return kName;
  }

  // Mirrors the flattened scene into device memory (no-op when nothing
  // changed).
  void Sync(const OfflineScene &scene) override;
  void Dispatch(const OfflineScene &scene, const RenderSettings &settings, OfflineFilm &film) override;
  std::vector<uint8_t> Develop(const OfflineFilm &film, const ToneMappingSettings &settings) const override;

  std::string DeviceDescription() const override;

  // Device memory currently owned by the backend, for tests and reporting.
  size_t DeviceMemoryBytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sparkium::backends

#endif  // LONGMARCH_CUDA_ENABLED
