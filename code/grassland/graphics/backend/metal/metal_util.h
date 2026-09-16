#pragma once
#include <Metal/Metal.hpp>

#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

// Every native owning reference is scoped, including autoreleased encoder objects.
struct MetalPool {
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  ~MetalPool() {
    pool->release();
  }
};
void MetalCheck(const void *object, NS::Error *error, const char *operation);
MTL::PixelFormat MetalFormat(ImageFormat format);

class MetalCore;
class MetalBuffer;
class MetalImage;
class MetalSampler;
class MetalShader;
class MetalComputeProgram;
class MetalProgram;
class MetalCommandContext;
class MetalWindow;
struct MetalStage;

struct MetalBinding {
  ResourceType type;
  int count;
};

}  // namespace grassland::graphics::backend
