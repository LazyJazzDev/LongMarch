#pragma once
#include <cstddef>
#include <memory>

namespace sparkium::backend {

class NativeMemory {
 public:
  NativeMemory(bool cuda, size_t size);
  NativeMemory(std::shared_ptr<const void> owner, const void *data, size_t size);
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
  std::shared_ptr<const void> owner_;
};

}  // namespace sparkium::backend
