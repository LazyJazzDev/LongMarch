#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi {
void IObject::AddRef() noexcept {
  references_.fetch_add(1, std::memory_order_relaxed);
}

void IObject::Release() noexcept {
  if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1)
    delete this;
}

IObject::~IObject() = default;

bool IObject::IsUniquelyOwned() const noexcept {
  return references_.load(std::memory_order_acquire) == 1;
}
}  // namespace grassland::rhi
