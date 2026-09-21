#pragma once
#include "grassland/graphics/image.h"
#include "sparkium/backend/cuda/cuda_bindings.h"
#include "sparkium/backend/cuda/cuda_memory.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend::cuda {
using namespace grassland;
using namespace grassland::graphics;

class CudaImage final : public Image {
 public:
  CudaImage(int width, int height, ImageFormat format);

  Extent2D Extent() const override {
    return extent_;
  }

  ImageFormat Format() const override {
    return format_;
  }

  void UploadData(const void *data) const override {
    UploadData(data, {0, 0}, extent_);
  }

  void DownloadData(void *data) const override {
    DownloadData(data, {0, 0}, extent_);
  }

  void UploadData(const void *, const Offset2D &, const Extent2D &) const override;
  void DownloadData(void *, const Offset2D &, const Extent2D &) const override;
  void Clear(const ClearValue &value);

  CudaImageBinding Binding() const {
    return {{memory->Data(), memory->Size()}, extent_.width, extent_.height, uint32_t(unorm_), 0};
  }

  std::unique_ptr<CudaMemory> memory;

 private:
  Extent2D extent_;
  ImageFormat format_;
  size_t channels_, external_bytes_;
  bool unorm_;
};

}  // namespace sparkium::backend::cuda
