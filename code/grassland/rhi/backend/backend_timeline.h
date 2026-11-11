#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_timeline.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendTimeline,
                        grassland::rhi::IObject,
                        "1eefb6fe-e220-483b-a0da-92f046be5fe6")

namespace grassland::rhi::backend {
class IBackendTimeline : public IObject {
 public:
  virtual IBackendDevice *Owner() const = 0;
};

}  // namespace grassland::rhi::backend
