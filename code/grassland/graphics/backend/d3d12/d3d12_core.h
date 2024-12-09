#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {

class D3D12Core : public Core {
 public:
  D3D12Core(const Settings &settings);

  int CreateBuffer(size_t size,
                   BufferType type,
                   double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width,
                  int height,
                  ImageFormat format,
                  double_ptr<Image> pp_image) override;

 private:
};

}  // namespace grassland::graphics::backend
