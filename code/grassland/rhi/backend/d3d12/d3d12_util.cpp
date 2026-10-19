#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
bool Check(HRESULT result, const char *operation) {
  if (SUCCEEDED(result))
    return true;
  ReportError(result, operation);
  return false;
}
}  // namespace grassland::rhi::backend::d3d12
