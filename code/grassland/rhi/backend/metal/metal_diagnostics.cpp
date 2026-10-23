#include "grassland/rhi/backend/metal/metal_diagnostics.h"

#include "grassland/util/log.h"

namespace grassland::rhi::backend::metal {
void ReportMetalError(NS::Error *error, const char *operation, bool details) {
  if (!error) {
    LogError("{}.", operation);
    return;
  }
  LogError("{}: {} [{}:{}]", operation, error->localizedDescription()->utf8String(), error->domain()->utf8String(),
           error->code());
  if (details && error->userInfo())
    LogError("Metal diagnostic information: {}", error->userInfo()->description()->utf8String());
}
}  // namespace grassland::rhi::backend::metal
