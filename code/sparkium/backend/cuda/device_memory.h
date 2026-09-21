#pragma once
#include <cstddef>

namespace sparkium::backend {
void *AllocateCudaMemory(size_t size);
void FreeCudaMemory(void *data) noexcept;
void UploadCudaMemory(void *destination, const void *source, size_t size, size_t offset);
void DownloadCudaMemory(void *destination, const void *source, size_t size, size_t offset);
}  // namespace sparkium::backend
