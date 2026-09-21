#pragma once
#include <cstddef>
#include <memory>

namespace sparkium::backend::cpu {

class CpuMemory {
 public:
  CpuMemory(size_t size);
  CpuMemory(std::shared_ptr<const void> owner, const void *data, size_t size);
  ~CpuMemory();
  CpuMemory(const CpuMemory &) = delete;
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
  std::shared_ptr<const void> owner_;
};

}  // namespace sparkium::backend::cpu
