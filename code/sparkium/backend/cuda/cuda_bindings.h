#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/buffer.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/sampler.h"

namespace sparkium::backend::cuda {
using namespace grassland;
using namespace grassland::graphics;

// Compute target ABI: buffer/unsized array = pointer + byte size/element count.
struct CudaSpan {
  void *data{};
  size_t size{};
};

struct CudaImageBinding {
  CudaSpan pixels;
  uint32_t width{}, height{};
  uint32_t unorm{}, padding{};
};

static_assert(sizeof(CudaImageBinding) == 32);

struct CudaBindings {
  std::map<int, std::vector<BufferRange>> buffers;
  std::map<int, std::vector<Image *>> images;
  std::map<int, std::vector<Sampler *>> samplers;
  std::map<int, AccelerationStructure *> acceleration_structures;
};

}  // namespace sparkium::backend::cuda
