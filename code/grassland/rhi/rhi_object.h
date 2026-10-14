#pragma once
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {
// Factories return one owned reference. Ordinary object arguments are borrowed.
// Release all caller-owned children before their parent; children do not retain it.
class IObject {
 public:
  virtual void AddRef() noexcept = 0;
  virtual void Release() noexcept = 0;

 protected:
  virtual ~IObject();
};
}  // namespace grassland::rhi
