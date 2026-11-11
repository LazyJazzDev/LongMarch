#include "grassland/rhi/rhi_core.h"

#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi {
ICore::~ICore() = default;

ICore *CreateCore(BackendAPI api, const CoreDesc &desc) {
  return backend::CreateCore(api, desc);
}
}  // namespace grassland::rhi

LONGMARCH_RHI_UUID(grassland::rhi::ICore, "299e58a8-f58c-4cea-9df2-7ba69bc52d75")
