#include "grassland/rhi/rhi_core.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi {
ICore::~ICore() = default;

ICore *CreateCore(BackendAPI api, const CoreDesc &desc) {
  return backend::CreateCore(api, desc);
}
}  // namespace grassland::rhi
