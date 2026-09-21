#include "sparkium/backend/render_backend.h"

namespace sparkium::backend {
std::unique_ptr<Backend> CreateGraphicsBackend(const RendererSettings &settings);
#ifdef SPARKIUM_NATIVE_ENABLED
std::unique_ptr<Backend> CreateCpuBackend(const RendererSettings &settings);
#endif
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
std::unique_ptr<Backend> CreateCudaBackend(const RendererSettings &settings);
#endif
std::unique_ptr<Backend> CreateBackend(const RendererSettings &settings) {
  switch (settings.backend) {
    case RenderBackend::Graphics:
      return CreateGraphicsBackend(settings);
#ifdef SPARKIUM_NATIVE_ENABLED
    case RenderBackend::CPU:
      return CreateCpuBackend(settings);
#endif
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
    case RenderBackend::CUDA:
      return CreateCudaBackend(settings);
#endif
    default:
      throw std::invalid_argument("rendering backend is not built");
  }
}
}  // namespace sparkium::backend
