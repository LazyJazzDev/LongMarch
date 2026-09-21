#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "grassland/graphics/buffer.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/sampler.h"

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

// Compute target ABI: buffer/unsized array = pointer + byte size/element count.
struct CpuSpan {
  void *data{};
  size_t size{};
};

struct CpuImageBinding {
  CpuSpan pixels;
  uint32_t width{}, height{};
  uint32_t unorm{}, padding{};
};

static_assert(sizeof(CpuImageBinding) == 32);

struct CpuBindings {
  std::map<int, std::vector<BufferRange>> buffers;
  std::map<int, std::vector<Image *>> images;
  std::map<int, std::vector<Sampler *>> samplers;
};

}  // namespace sparkium::backend::cpu
