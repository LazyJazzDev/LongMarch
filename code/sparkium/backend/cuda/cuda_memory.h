#pragma once
#include <cstddef>
#include <memory>

namespace sparkium::backend::cuda {

class CudaMemory {
 public:
  CudaMemory(size_t size);
  ~CudaMemory();
  CudaMemory(const CudaMemory &) = delete;
  void Upload(const void *, size_t, size_t = 0) const;
  void Download(void *, size_t, size_t = 0) const;

  void *Data() const {
    return data_;
  }

  size_t Size() const {
    return size_;
  }

 private:
  size_t size_;
  void *data_{};
};

}  // namespace sparkium::backend::cuda
