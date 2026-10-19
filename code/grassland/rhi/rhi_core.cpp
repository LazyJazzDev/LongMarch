#include "grassland/rhi/rhi_core.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi {
ICore::~ICore() = default;

ICore *CreateCore(BackendAPI api) {
  try {
    return backend::CreateCore(api).release();
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
}  // namespace grassland::rhi
