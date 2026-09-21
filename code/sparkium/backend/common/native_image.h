#pragma once
#include "grassland/graphics/image.h"
#include "sparkium/backend/common/native_bindings.h"
#include "sparkium/backend/common/native_memory.h"
#include "sparkium/scene/scene_definition.h"

namespace sparkium::backend {
using namespace grassland;
using namespace grassland::graphics;

class NativeImage final : public Image {
 public:
  NativeImage(bool cuda, int width, int height, ImageFormat format);
  explicit NativeImage(std::shared_ptr<const TextureData> texture);

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

  NativeImageBinding Binding() const {
    return {{memory->Data(), memory->Size()}, extent_.width, extent_.height, uint32_t(unorm_), 0};
  }

  std::unique_ptr<NativeMemory> memory;

 private:
  Extent2D extent_;
  ImageFormat format_;
  size_t channels_, external_bytes_;
  bool unorm_;
};

}  // namespace sparkium::backend
