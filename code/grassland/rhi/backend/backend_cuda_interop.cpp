#include "grassland/rhi/backend/backend_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend {
bool CheckCUDA(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return true;
  ReportError(static_cast<int32_t>(error), std::string(operation) + ": " + cudaGetErrorString(error));
  return false;
}

CUDADeviceScope::CUDADeviceScope(int device) {
  if (device < 0 || !CheckCUDA(cudaGetDevice(&previous_), "Get CUDA device"))
    return;
  changed_ = previous_ != device;
  valid_ = !changed_ || CheckCUDA(cudaSetDevice(device), "Select CUDA device");
}

CUDADeviceScope::~CUDADeviceScope() {
  if (valid_ && changed_)
    CheckCUDA(cudaSetDevice(previous_), "Restore CUDA device");
}

CUDADeviceScope::operator bool() const {
  return valid_;
}

CUDAMapping::CUDAMapping(int device) : device_(device) {
}

CUDAMapping::~CUDAMapping() {
  Reset();
}

void CUDAMapping::Reset() {
  CUDADeviceScope scope(device_);
  if (!scope)
    return;
  if (pointer_)
    CheckCUDA(cudaFree(pointer_), "Free CUDA mapping");
  if (memory_)
    CheckCUDA(cudaDestroyExternalMemory(memory_), "Destroy CUDA external memory");
  pointer_ = nullptr;
  memory_ = nullptr;
}

bool CUDAMapping::Import(const cudaExternalMemoryHandleDesc &desc, uint64_t size) {
  CUDADeviceScope scope(device_);
  if (!scope || !CheckCUDA(cudaImportExternalMemory(&memory_, &desc), "Import CUDA memory"))
    return false;
  cudaExternalMemoryBufferDesc buffer{};
  buffer.size = size;
  return CheckCUDA(cudaExternalMemoryGetMappedBuffer(&pointer_, memory_, &buffer), "Map CUDA buffer");
}

void *CUDAMapping::Pointer() const {
  return pointer_;
}

bool CUDAMapping::HasMemory() const {
  return memory_ != nullptr;
}

CUDAInteropTimeline::CUDAInteropTimeline(Device *device) : device_(device) {
}

CUDAInteropTimeline::~CUDAInteropTimeline() {
  Reset();
}

void CUDAInteropTimeline::Reset() {
  CUDADeviceScope scope(device_->GetPhysicalDevice().cuda_device);
  if (scope && semaphore_)
    CheckCUDA(cudaDestroyExternalSemaphore(semaphore_), "Destroy CUDA external semaphore");
  semaphore_ = nullptr;
}

Device *CUDAInteropTimeline::Owner() const {
  return device_;
}

bool CUDAInteropTimeline::Import(const cudaExternalSemaphoreHandleDesc &desc) {
  CUDADeviceScope scope(device_->GetPhysicalDevice().cuda_device);
  return scope && CheckCUDA(cudaImportExternalSemaphore(&semaphore_, &desc), "Import CUDA semaphore");
}

bool CUDAInteropTimeline::SignalCUDA(cudaStream_t stream, uint64_t value) {
  CUDADeviceScope scope(device_->GetPhysicalDevice().cuda_device);
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = value;
  return scope && CheckCUDA(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream), "CUDA signal");
}

bool CUDAInteropTimeline::WaitCUDA(cudaStream_t stream, uint64_t value) {
  CUDADeviceScope scope(device_->GetPhysicalDevice().cuda_device);
  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = value;
  return scope && CheckCUDA(cudaWaitExternalSemaphoresAsync(&semaphore_, &params, 1, stream), "CUDA wait");
}
}  // namespace grassland::rhi::backend
#endif
