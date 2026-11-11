#pragma once
#include "grassland/rhi/rhi_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::IObject, "898187ec-5050-4d89-b818-ceedf1a2d067")

namespace grassland::rhi {
// Factories return one owned reference. Ordinary object arguments are borrowed.
// Release all caller-owned children before their parent; children do not retain it.
// Caller-contract diagnostics are enabled only when NDEBUG is not defined.
// Release builds require valid arguments, object lifetimes and call ordering.
class IObject {
 public:
  // Success transfers one owned reference. Failure clears a non-null output argument.
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept = 0;

  template <typename Interface>
  Interface *QueryInterface() noexcept {
    void *result = nullptr;
    if (!QueryInterface(uuid<Interface>(), &result))
      return nullptr;
    return static_cast<Interface *>(result);
  }

  virtual void AddRef() noexcept = 0;
  virtual void Release() noexcept = 0;

 protected:
  virtual ~IObject();
};
}  // namespace grassland::rhi
