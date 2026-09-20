#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/buffer.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/sampler.h"

namespace grassland::graphics::backend {

// Native target ABI: buffer/unsized array = pointer + byte size/element count.
struct NativeSpan {
  void *data{};
  size_t size{};
};

struct NativeImageBinding {
  NativeSpan pixels;
  uint32_t width{}, height{};
  uint32_t unorm{}, padding{};
};

static_assert(sizeof(NativeImageBinding) == 32);

struct NativeBindings {
  std::map<int, std::vector<BufferRange>> buffers;
  std::map<int, std::vector<Image *>> images;
  std::map<int, std::vector<Sampler *>> samplers;
  std::map<int, AccelerationStructure *> acceleration_structures;
};

}  // namespace grassland::graphics::backend
