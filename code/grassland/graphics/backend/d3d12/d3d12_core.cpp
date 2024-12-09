#include "grassland/graphics/backend/d3d12/d3d12_core.h"

namespace grassland::graphics::backend {

D3D12Core::D3D12Core(const Settings &settings) : Core(settings) {
}

int D3D12Core::CreateBuffer(size_t size,
                            BufferType type,
                            double_ptr<Buffer> pp_buffer) {
  return 0;
}

int D3D12Core::CreateImage(int width,
                           int height,
                           ImageFormat format,
                           double_ptr<Image> pp_image) {
  return 0;
}

}  // namespace grassland::graphics::backend
