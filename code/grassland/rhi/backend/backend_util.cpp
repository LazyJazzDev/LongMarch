#include "grassland/rhi/backend/backend_util.h"

#include "grassland/util/log.h"

namespace grassland::rhi::backend {
void ReportError(int32_t error_code, const std::string &error_message) {
  LogError("Error {}: {}", error_code, error_message);
}
}  // namespace grassland::rhi::backend
