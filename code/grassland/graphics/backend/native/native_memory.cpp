#include "native_memory.h"

#include <algorithm>
#include <cstring>

#include "native_util.h"

namespace grassland::graphics::backend {

NativeMemory::NativeMemory(bool cuda, size_t size) : cuda_(cuda), size_(size) {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CUdeviceptr ptr;
    CheckCUDA(cuMemAlloc(&ptr, std::max(size, size_t(16))));
    data_ = reinterpret_cast<void *>(ptr);
    CheckCUDA(cuMemsetD8(ptr, 0, std::max(size, size_t(16))));
    return;
  }
#endif
  if (cuda_)
    NativeUnsupported();
  data_ = ::operator new(std::max(size, size_t(16)));
  std::memset(data_, 0, std::max(size, size_t(16)));
}

NativeMemory::~NativeMemory() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    cuMemFree(reinterpret_cast<CUdeviceptr>(data_));
    return;
  }
#endif
  ::operator delete(data_);
}

void NativeMemory::Upload(const void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("native buffer upload");
  if (!size)
    return;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CheckCUDA(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(data_) + offset, p, size));
    return;
  }
#endif
  std::memcpy(static_cast<char *>(data_) + offset, p, size);
}

void NativeMemory::Download(void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("native buffer download");
  if (!size)
    return;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_) {
    CheckCUDA(cuMemcpyDtoH(p, reinterpret_cast<CUdeviceptr>(data_) + offset, size));
    return;
  }
#endif
  std::memcpy(p, static_cast<char *>(data_) + offset, size);
}

}  // namespace grassland::graphics::backend
