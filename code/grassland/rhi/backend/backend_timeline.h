#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendTimeline)

namespace grassland::rhi::backend {
class IBackendTimeline : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};

}  // namespace grassland::rhi::backend
