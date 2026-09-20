#pragma once
#include <cstddef>

namespace sparkium::backend {

class NativeMemory {
 public:
  NativeMemory(bool cuda, size_t size);
  ~NativeMemory();
  NativeMemory(const NativeMemory &) = delete;
  void Upload(const void *, size_t, size_t = 0) const;
  void Download(void *, size_t, size_t = 0) const;

  void *Data() const {
    return data_;
  }

  size_t Size() const {
    return size_;
  }

  bool IsCUDA() const {
    return cuda_;
  }

 private:
  bool cuda_;
  size_t size_;
  void *data_{};
};

}  // namespace sparkium::backend
