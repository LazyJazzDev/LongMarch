#include "sparkium/backend/cpu/cpu_memory.h"

#include <algorithm>
#include <cstring>

#include "sparkium/backend/cpu/cpu_util.h"

namespace sparkium::backend::cpu {

CpuMemory::CpuMemory(size_t size) : size_(size) {
  data_ = ::operator new(std::max(size, size_t(16)));
  std::memset(data_, 0, std::max(size, size_t(16)));
}

CpuMemory::CpuMemory(std::shared_ptr<const void> owner, const void *data, size_t size)
    : size_(size),
      data_(const_cast<void *>(data)),
      owner_(std::move(owner)) {
  if (!owner_ || (!data && size))
    throw std::invalid_argument("invalid borrowed scene memory");
}

CpuMemory::~CpuMemory() {
  if (owner_)
    return;
  ::operator delete(data_);
}

void CpuMemory::Upload(const void *p, size_t size, size_t offset) const {
  if (owner_)
    throw std::logic_error("scene memory is read-only");
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("compute buffer upload");
  if (!size)
    return;
  std::memcpy(static_cast<char *>(data_) + offset, p, size);
}

void CpuMemory::Download(void *p, size_t size, size_t offset) const {
  if (offset > size_ || size > size_ - offset)
    throw std::out_of_range("compute buffer download");
  if (!size)
    return;
  std::memcpy(p, static_cast<char *>(data_) + offset, size);
}

}  // namespace sparkium::backend::cpu
