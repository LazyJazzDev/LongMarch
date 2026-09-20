#include "sparkium/backend/common/native_memory.h"

#include <algorithm>
#include <cstring>

#include "sparkium/backend/common/native_util.h"
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
#include "sparkium/backend/cuda/cuda_memory.h"
#endif

namespace sparkium::backend {

NativeMemory::NativeMemory(bool cuda, size_t size) : cuda_(cuda), size_(size) {
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  if (cuda_) {
    data_ = AllocateCudaMemory(std::max(size, size_t(16)));
    return;
  }
#endif
  if (cuda_)
    NativeUnsupported();
  data_ = ::operator new(std::max(size, size_t(16)));
  std::memset(data_, 0, std::max(size, size_t(16)));
}

NativeMemory::~NativeMemory() {
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  if (cuda_) {
    FreeCudaMemory(data_);
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
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  if (cuda_) {
    UploadCudaMemory(data_, p, size, offset);
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
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  if (cuda_) {
    DownloadCudaMemory(p, data_, size, offset);
    return;
  }
#endif
  std::memcpy(p, static_cast<char *>(data_) + offset, size);
}

}  // namespace sparkium::backend
