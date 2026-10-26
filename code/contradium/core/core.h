#pragma once
#include "contradium/core/core_util.h"
#include "grassland/rhi/rhi_device.h"

namespace contradium {

class Core {
 public:
  Core(rhi::IDevice *core);

 private:
  rhi::IDevice *core_;
};

}  // namespace contradium
