#include "grassland/rhi/backend/backend_timeline.h"

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
#if defined(LONGMARCH_CUDA_RUNTIME)
bool Timeline::ImportCUDA(const cudaExternalSemaphoreHandleDesc &desc) {
  CUDADeviceScope scope(device_->GetCUDADeviceIndex());
  return scope && CheckCUDA(cudaImportExternalSemaphore(&cuda_semaphore_, &desc), "Import CUDA timeline");
}

void Timeline::ReleaseCUDA() {
  if (!cuda_semaphore_)
    return;
  CUDADeviceScope scope(device_->GetCUDADeviceIndex());
  if (scope)
    CheckCUDA(cudaDestroyExternalSemaphore(cuda_semaphore_), "Destroy CUDA timeline");
  cuda_semaphore_ = nullptr;
}

bool Timeline::Wait(cudaStream_t stream, Event event) {
  auto source = dynamic_cast<Timeline *>(event.timeline);
  if (!cuda_semaphore_ || !source || source->Owner() != device_ || !source->cuda_semaphore_) {
    ReportError(-1, "CUDA waits require an exported timeline from this device.");
    return false;
  }
  CUDADeviceScope scope(device_->GetCUDADeviceIndex());
  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = event.value;
  return scope && CheckCUDA(cudaWaitExternalSemaphoresAsync(&source->cuda_semaphore_, &params, 1, stream),
                            "Wait for graphics timeline on CUDA stream");
}

Event Timeline::Signal(cudaStream_t stream, uint64_t value) {
  if (!cuda_semaphore_ || !value) {
    ReportError(-1, "CUDA signal requires an exported timeline and a positive value.");
    return {};
  }
  CUDADeviceScope scope(device_->GetCUDADeviceIndex());
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = value;
  if (!scope ||
      !CheckCUDA(cudaSignalExternalSemaphoresAsync(&cuda_semaphore_, &params, 1, stream), "Signal CUDA timeline"))
    return {};
  return {this, value};
}
#endif
}  // namespace grassland::rhi::backend
