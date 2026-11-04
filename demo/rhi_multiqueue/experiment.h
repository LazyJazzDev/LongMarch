#pragma once
#include "grassland/rhi/rhi.h"

namespace multiqueue_demo {
struct Result {
  bool success{};
  bool distinct_queues{};
  std::string device;
  uint32_t frames{};
  uint64_t mismatched_pixels{};
  std::vector<uint8_t> pixels;
};

Result Run(grassland::rhi::BackendAPI api,
           bool synchronize,
           bool validation,
           uint32_t frames = 8,
           uint32_t copies = 32);
}  // namespace multiqueue_demo
