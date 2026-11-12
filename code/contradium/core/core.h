#pragma once
#include "contradium/core/core_util.h"
#include "grassland/rhi/rhi_device.h"

namespace contradium {

class Core {
 public:
  Core(rhi::IDevice *device);

 private:
  rhi::IDevice *device_;
};

}  // namespace contradium
