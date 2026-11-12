#include "grassland/rhi/backend/backend_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <algorithm>

namespace grassland::rhi::backend {
CUDAExternalSemaphore::~CUDAExternalSemaphore() {
  Reset();
}

bool CUDAExternalSemaphore::Import(const cudaExternalSemaphoreHandleDesc &desc) {
  CUDADeviceScope scope(device_);
  return scope && CheckCUDA(cudaImportExternalSemaphore(&semaphore_, &desc), "Import CUDA timeline");
}

void CUDAExternalSemaphore::Reset() {
  if (!semaphore_)
    return;
  CUDADeviceScope scope(device_);
  if (scope)
    CheckCUDA(cudaDestroyExternalSemaphore(semaphore_), "Destroy CUDA timeline");
  semaphore_ = nullptr;
}

bool CUDAExternalSemaphore::Wait(const void *owner, cudaStream_t stream, const std::vector<Event> &events) const {
  std::vector<cudaExternalSemaphore_t> semaphores;
  std::vector<cudaExternalSemaphoreWaitParams> params;
  // Validate the entire set before enqueueing any wait. Coalesce duplicate timelines.
  for (const auto &event : events) {
    auto source = QueryInterfacePtr<IBackendCUDATimeline>(event.timeline);
    if (!source || source->CUDAOwner() != owner || !source->CUDASemaphore()) {
      ReportError(-1, "CUDA waits require exported timelines from this device.");
      return false;
    }
    auto semaphore = source->CUDASemaphore();
    auto found = std::find(semaphores.begin(), semaphores.end(), semaphore);
    if (found != semaphores.end()) {
      auto &value = params[found - semaphores.begin()].params.fence.value;
      value = std::max(value, static_cast<unsigned long long>(event.value));
    } else {
      semaphores.push_back(semaphore);
      cudaExternalSemaphoreWaitParams param{};
      param.params.fence.value = event.value;
      params.push_back(param);
    }
  }
  if (semaphores.empty())
    return true;
  CUDADeviceScope scope(device_);
  return scope && CheckCUDA(cudaWaitExternalSemaphoresAsync(semaphores.data(), params.data(),
                                                            static_cast<unsigned int>(semaphores.size()), stream),
                            "Wait for timelines on CUDA stream");
}

bool CUDAExternalSemaphore::Signal(cudaStream_t stream, uint64_t value) const {
  if (!semaphore_ || !value)
    return false;
  CUDADeviceScope scope(device_);
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = value;
  return scope && CheckCUDA(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream), "Signal CUDA timeline");
}
}  // namespace grassland::rhi::backend
#endif
